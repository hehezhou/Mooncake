#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

namespace mooncake {

struct StorageUsage {
    size_t used_bytes{0};
    size_t capacity_bytes{0};

    [[nodiscard]] double used_ratio() const noexcept {
        if (capacity_bytes == 0) {
            return 0.0;
        }
        return static_cast<double>(used_bytes) /
               static_cast<double>(capacity_bytes);
    }
};

/**
 * Wakes a waiter when space may have become available.
 *
 * NotifySpaceAvailable runs on allocator release paths, which can hold any
 * lock, so it never blocks: while disarmed it is a single atomic load, and
 * while armed it bumps an epoch and wakes a sleeper through the futex behind
 * std::atomic::notify_all. Many releases between two waits collapse into one
 * wake-up.
 */
class SpaceAvailableSignal {
   public:
    void NotifySpaceAvailable() noexcept {
        if (armed_.load()) {
            Wake();
        }
    }

    // Wakes the waiter even while disarmed.
    void Wake() noexcept {
        epoch_.fetch_add(1, std::memory_order_release);
        epoch_.notify_all();
    }

    void SetArmed(bool armed) noexcept { armed_.store(armed); }
    [[nodiscard]] bool IsArmed() const noexcept { return armed_.load(); }

    // Read the epoch before checking for space, then wait on it: a release
    // that lands in between changes the epoch, so the wait returns at once.
    [[nodiscard]] uint32_t Epoch() const noexcept {
        return epoch_.load(std::memory_order_acquire);
    }
    void WaitForChange(uint32_t seen) const noexcept {
        epoch_.wait(seen, std::memory_order_acquire);
    }

   private:
    std::atomic<bool> armed_{false};
    std::atomic<uint32_t> epoch_{0};
};

/**
 * Maintains aggregate usage for one storage domain.
 *
 * Mirrors the old metric-gauge model: allocation updates and reads are plain
 * atomics. Mount/unmount briefly updates used and capacity independently, so
 * readers may observe a best-effort torn pair during topology changes—the same
 * class of race the previous metric-based watermark path accepted.
 */
class StorageUsageTracker {
   public:
    [[nodiscard]] StorageUsage GetUsage() const noexcept {
        return StorageUsage{
            .used_bytes = used_bytes_.load(std::memory_order_relaxed),
            .capacity_bytes = capacity_bytes_.load(std::memory_order_relaxed),
        };
    }

    // Set before any allocator attaches; release paths read it unlocked.
    void SetSpaceAvailableSignal(
        std::shared_ptr<SpaceAvailableSignal> signal) noexcept {
        space_available_signal_ = std::move(signal);
    }

    void AddUsedBytes(size_t bytes) noexcept {
        used_bytes_.fetch_add(bytes, std::memory_order_relaxed);
    }

    void RemoveUsedBytes(size_t bytes) noexcept {
        used_bytes_.fetch_sub(bytes, std::memory_order_relaxed);
        NotifySpaceAvailable();
    }

    void AttachAllocator(size_t used_bytes, size_t capacity_bytes) noexcept {
        used_bytes_.fetch_add(used_bytes, std::memory_order_relaxed);
        capacity_bytes_.fetch_add(capacity_bytes, std::memory_order_relaxed);
        NotifySpaceAvailable();
    }

    void DetachAllocator(size_t used_bytes, size_t capacity_bytes) noexcept {
        used_bytes_.fetch_sub(used_bytes, std::memory_order_relaxed);
        capacity_bytes_.fetch_sub(capacity_bytes, std::memory_order_relaxed);
    }

   private:
    void NotifySpaceAvailable() noexcept {
        if (space_available_signal_) {
            space_available_signal_->NotifySpaceAvailable();
        }
    }

    std::atomic_size_t used_bytes_{0};
    std::atomic_size_t capacity_bytes_{0};
    std::shared_ptr<SpaceAvailableSignal> space_available_signal_;
};

/**
 * RAII registration of one allocator into a StorageUsageTracker.
 *
 * Attach once before the allocator is published. The registration stays
 * immutable for the allocator's lifetime and detaches from the destructor,
 * after in-flight allocate/deallocate calls have dropped their local
 * shared_ptr and the allocator can be destroyed.
 */
class StorageUsageRegistration {
   public:
    StorageUsageRegistration(std::shared_ptr<StorageUsageTracker> tracker,
                             std::atomic_size_t& used_bytes,
                             size_t capacity_bytes) noexcept
        : tracker_(std::move(tracker)),
          used_bytes_(used_bytes),
          capacity_bytes_(capacity_bytes) {
        tracker_->AttachAllocator(used_bytes_.load(std::memory_order_relaxed),
                                  capacity_bytes_);
    }

    ~StorageUsageRegistration() {
        tracker_->DetachAllocator(used_bytes_.load(std::memory_order_relaxed),
                                  capacity_bytes_);
    }

    StorageUsageRegistration(const StorageUsageRegistration&) = delete;
    StorageUsageRegistration& operator=(const StorageUsageRegistration&) =
        delete;

    void AddUsedBytes(size_t bytes) noexcept { tracker_->AddUsedBytes(bytes); }

    void RemoveUsedBytes(size_t bytes) noexcept {
        tracker_->RemoveUsedBytes(bytes);
    }

   private:
    std::shared_ptr<StorageUsageTracker> tracker_;
    std::atomic_size_t& used_bytes_;
    size_t capacity_bytes_;
};

}  // namespace mooncake
