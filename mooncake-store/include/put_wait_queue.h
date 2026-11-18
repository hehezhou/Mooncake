#pragma once

#include <chrono>
#include <cstddef>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

#include "storage_usage.h"

namespace mooncake {

/**
 * Parks PutStart requests that found no memory and admits them in arrival
 * order once space appears.
 *
 * Only the head of the queue is retried, and only after the space signal
 * fires, so a wake-up costs one allocation attempt per admitted waiter plus
 * one for the head that still does not fit, however many waiters are parked.
 * Waiting holds no RPC thread: the caller suspends on its own future.
 */
class PutWaitQueue {
   public:
    struct Waiter {
        // Bytes the waiter asks for, counted against the queue byte limit.
        size_t bytes = 0;
        // Tries to start the put. Returns true once the waiter has its result
        // and false while it still needs space; may hold partial allocations.
        std::function<bool()> attempt;
        // Gives back the partial allocations of a failed attempt. Called when
        // a waiter that tried on arrival is parked behind others, so that only
        // the head of the queue ever holds part of what it asked for.
        // Optional.
        std::function<void()> release_partial;
        // Resolves the waiter when the queue shuts down before admitting it.
        std::function<void()> cancel;
        // Set by the queue when the waiter is parked.
        std::chrono::steady_clock::time_point parked_at;
    };

    enum class SubmitResult {
        // The waiter has its result.
        kDone,
        // The waiter is parked and gets its result later.
        kQueued,
        // The queue is full or shutting down; the caller handles the request
        // without waiting.
        kRejected,
    };

    PutWaitQueue(std::shared_ptr<SpaceAvailableSignal> signal,
                 size_t max_requests, size_t max_bytes);
    ~PutWaitQueue();

    PutWaitQueue(const PutWaitQueue&) = delete;
    PutWaitQueue& operator=(const PutWaitQueue&) = delete;

    // With an empty queue, attempts the put at once on the calling thread and
    // parks it if it still needs space; if others were parked meanwhile, it
    // releases its partial allocations and joins behind them. With waiters
    // present, parks it behind them without attempting, so a newcomer never
    // takes space a parked request is waiting for.
    SubmitResult Submit(Waiter waiter);

    [[nodiscard]] size_t size() const;

   private:
    void DrainLoop();
    // Retries the head until it still needs space or the queue empties.
    void DrainOnce();
    bool HasRoomLocked(size_t bytes) const;
    void ParkLocked(std::shared_ptr<Waiter> waiter);
    void PublishSizeLocked() const;

    const std::shared_ptr<SpaceAvailableSignal> signal_;
    const size_t max_requests_;
    const size_t max_bytes_;

    mutable std::mutex mutex_;
    std::deque<std::shared_ptr<Waiter>> waiters_;
    size_t queued_bytes_ = 0;
    bool stopping_ = false;
    std::thread drain_thread_;
};

}  // namespace mooncake
