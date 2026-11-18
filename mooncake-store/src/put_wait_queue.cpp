#include "put_wait_queue.h"

#include <utility>

#include "master_metric_manager.h"

namespace mooncake {

PutWaitQueue::PutWaitQueue(std::shared_ptr<SpaceAvailableSignal> signal,
                           size_t max_requests, size_t max_bytes)
    : signal_(std::move(signal)),
      max_requests_(max_requests),
      max_bytes_(max_bytes) {
    drain_thread_ = std::thread([this] { DrainLoop(); });
}

PutWaitQueue::~PutWaitQueue() {
    {
        std::lock_guard lock(mutex_);
        stopping_ = true;
    }
    signal_->Wake();
    drain_thread_.join();

    std::deque<std::shared_ptr<Waiter>> remaining;
    {
        std::lock_guard lock(mutex_);
        remaining.swap(waiters_);
        queued_bytes_ = 0;
        signal_->SetArmed(false);
        PublishSizeLocked();
    }
    for (const auto& waiter : remaining) {
        waiter->cancel();
    }
}

bool PutWaitQueue::HasRoomLocked(size_t bytes) const {
    // A limit of zero means unlimited.
    if (max_requests_ != 0 && waiters_.size() >= max_requests_) {
        return false;
    }
    return max_bytes_ == 0 || queued_bytes_ + bytes <= max_bytes_;
}

void PutWaitQueue::ParkLocked(std::shared_ptr<Waiter> waiter) {
    waiter->parked_at = std::chrono::steady_clock::now();
    queued_bytes_ += waiter->bytes;
    waiters_.push_back(std::move(waiter));
    signal_->SetArmed(true);
    MasterMetricManager::instance().inc_put_wait_enqueued();
    PublishSizeLocked();
}

void PutWaitQueue::PublishSizeLocked() const {
    MasterMetricManager::instance().set_put_wait_queue(
        static_cast<int64_t>(waiters_.size()),
        static_cast<int64_t>(queued_bytes_));
}

PutWaitQueue::SubmitResult PutWaitQueue::Submit(Waiter waiter) {
    auto entry = std::make_shared<Waiter>(std::move(waiter));
    {
        std::lock_guard lock(mutex_);
        if (stopping_) {
            return SubmitResult::kRejected;
        }
        if (!waiters_.empty()) {
            if (!HasRoomLocked(entry->bytes)) {
                MasterMetricManager::instance().inc_put_wait_rejected();
                return SubmitResult::kRejected;
            }
            ParkLocked(std::move(entry));
            return SubmitResult::kQueued;
        }
    }

    if (entry->attempt()) {
        return SubmitResult::kDone;
    }

    // Becoming the head, it keeps what its attempt took. Behind others it
    // gives that back first, so that only the head holds partial allocations.
    bool stopping = false;
    bool parked = false;
    {
        std::lock_guard lock(mutex_);
        stopping = stopping_;
        if (!stopping && waiters_.empty()) {
            ParkLocked(entry);
            parked = true;
        }
    }
    if (!stopping && !parked) {
        if (entry->release_partial) {
            entry->release_partial();
        }
        std::lock_guard lock(mutex_);
        stopping = stopping_;
        if (!stopping) {
            ParkLocked(entry);
        }
    }
    if (stopping) {
        entry->cancel();
        return SubmitResult::kDone;
    }
    // Space released between the failed attempt and arming the signal raised
    // no wake-up, and the queue may have emptied while the partial allocations
    // were given back, so retry the head once now.
    signal_->Wake();
    return SubmitResult::kQueued;
}

size_t PutWaitQueue::size() const {
    std::lock_guard lock(mutex_);
    return waiters_.size();
}

void PutWaitQueue::DrainLoop() {
    while (true) {
        const uint32_t seen = signal_->Epoch();
        {
            std::lock_guard lock(mutex_);
            if (stopping_) {
                return;
            }
        }
        DrainOnce();
        signal_->WaitForChange(seen);
    }
}

void PutWaitQueue::DrainOnce() {
    bool counted = false;
    while (true) {
        std::shared_ptr<Waiter> head;
        {
            std::lock_guard lock(mutex_);
            if (stopping_) {
                return;
            }
            if (waiters_.empty()) {
                signal_->SetArmed(false);
                return;
            }
            head = waiters_.front();
        }
        if (!counted) {
            MasterMetricManager::instance().inc_put_wait_drains();
            counted = true;
        }
        MasterMetricManager::instance().inc_put_wait_attempts();
        if (!head->attempt()) {
            return;
        }
        MasterMetricManager::instance().inc_put_wait_admitted();
        MasterMetricManager::instance().inc_put_wait_time_us(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - head->parked_at)
                .count());
        std::lock_guard lock(mutex_);
        // Waiters only join at the back and only this thread removes them,
        // so the head is still in front.
        waiters_.pop_front();
        queued_bytes_ -= head->bytes;
        if (waiters_.empty()) {
            signal_->SetArmed(false);
        }
        PublishSizeLocked();
    }
}

}  // namespace mooncake
