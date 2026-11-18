#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <ylt/coro_rpc/coro_rpc_server.hpp>

#include "common/network.h"
#include "master_client.h"
#include "master_config.h"
#include "rpc_service.h"
#include "rpc_worker_pool.h"
#include "types.h"

namespace mooncake {
namespace {

constexpr uint64_t kSegmentSize = 16 * 1024 * 1024;
constexpr uint64_t kObjectSize = 4 * 1024 * 1024;
// How long a parked put is watched to confirm it is still waiting.
constexpr auto kStillParked = std::chrono::milliseconds(300);
constexpr auto kAdmitted = std::chrono::seconds(15);

using PutStartResult = WrappedMasterService::PutStartResult;
using BatchPutStartResult = WrappedMasterService::BatchPutStartResult;

WrappedMasterServiceConfig Config(bool wait_for_space) {
    WrappedMasterServiceConfig config;
    config.default_kv_lease_ttl = 0;
    config.enable_metric_reporting = false;
    config.put_wait_for_space = wait_for_space;
    // Only removals free memory unless a test turns eviction on.
    config.eviction_ratio = 0.0;
    config.eviction_high_watermark_ratio = 1.0;
    return config;
}

ReplicateConfig OneReplica() {
    ReplicateConfig config;
    config.replica_num = 1;
    return config;
}

template <typename T>
bool Ready(const std::future<T>& future, std::chrono::milliseconds timeout) {
    return future.wait_for(timeout) == std::future_status::ready;
}

// A master serving RPCs from a worker pool with one mounted memory segment,
// and a client whose PutStart calls wait without a timeout.
class PutWaitCluster {
   public:
    explicit PutWaitCluster(const WrappedMasterServiceConfig& config) {
        setenv("MC_PUT_START_RPC_TIMEOUT_MS", "-1", 1);
        const auto ports = getFreeTcpPorts(1);
        EXPECT_EQ(ports.size(), 1u);
        server_ = std::make_unique<coro_rpc::coro_rpc_server>(
            /*thread_num=*/2, ports.front(), "127.0.0.1",
            std::chrono::seconds(0), /*tcp_no_delay=*/true);
        service_ = std::make_unique<WrappedMasterService>(config);
        pool_ = std::make_unique<RpcWorkerPool>(2);
        RegisterRpcService(*server_, *service_, pool_.get());
        auto started = server_->async_start();
        EXPECT_FALSE(started.hasResult());

        Segment segment;
        segment.id = generate_uuid();
        segment.name = "put_wait_segment";
        segment.base = 0x300000000;
        segment.size = kSegmentSize;
        EXPECT_TRUE(service_->MountSegment(segment, owner_id_).has_value());

        client_ = std::make_unique<MasterClient>(generate_uuid());
        EXPECT_EQ(
            client_->Connect("127.0.0.1:" + std::to_string(ports.front())),
            ErrorCode::OK);
    }

    ~PutWaitCluster() {
        // Stop serving first; the service then resolves anything still parked
        // and the server outlives it, as in the master binary.
        server_->stop();
        client_.reset();
        pool_.reset();
        service_.reset();
        server_.reset();
        unsetenv("MC_PUT_START_RPC_TIMEOUT_MS");
    }

    WrappedMasterService& service() { return *service_; }
    MasterClient& client() { return *client_; }
    const UUID& owner_id() const { return owner_id_; }

    // Completes objects until the segment is full and returns their keys.
    std::vector<std::string> Fill() {
        std::vector<std::string> keys;
        for (int i = 0;; ++i) {
            const std::string key = "fill-" + std::to_string(i);
            auto started =
                service_->PutStart(owner_id_, key, kObjectSize, OneReplica());
            if (!started) {
                EXPECT_EQ(started.error(), ErrorCode::NO_AVAILABLE_HANDLE);
                break;
            }
            EXPECT_TRUE(service_
                            ->PutEnd(owner_id_, ObjectMeta{key, std::nullopt},
                                     ReplicaType::MEMORY)
                            .has_value());
            keys.push_back(key);
        }
        EXPECT_FALSE(keys.empty());
        return keys;
    }

    std::future<PutStartResult> PutAsync(const std::string& key,
                                         uint64_t size = kObjectSize) {
        return std::async(std::launch::async, [this, key, size] {
            return client_->PutStart(key, {size}, OneReplica());
        });
    }

   private:
    const UUID owner_id_ = generate_uuid();
    std::unique_ptr<coro_rpc::coro_rpc_server> server_;
    std::unique_ptr<WrappedMasterService> service_;
    std::unique_ptr<RpcWorkerPool> pool_;
    std::unique_ptr<MasterClient> client_;
};

TEST(PutWaitForSpaceTest, FailsAtOnceWhenDisabled) {
    PutWaitCluster cluster(Config(/*wait_for_space=*/false));
    cluster.Fill();

    auto pending = cluster.PutAsync("late");
    ASSERT_TRUE(Ready(pending, std::chrono::seconds(5)));
    auto result = pending.get();
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), ErrorCode::NO_AVAILABLE_HANDLE);
}

TEST(PutWaitForSpaceTest, PutWaitsUntilRemovalFreesSpace) {
    PutWaitCluster cluster(Config(/*wait_for_space=*/true));
    const auto filled = cluster.Fill();

    auto pending = cluster.PutAsync("late");
    EXPECT_FALSE(Ready(pending, kStillParked));

    ASSERT_TRUE(cluster.service().Remove(filled.front()).has_value());
    ASSERT_TRUE(Ready(pending, kAdmitted));
    auto result = pending.get();
    ASSERT_TRUE(result.has_value()) << toString(result.error());
    EXPECT_FALSE(result->empty());
}

TEST(PutWaitForSpaceTest, BatchPutWaitsUntilEveryKeyFits) {
    PutWaitCluster cluster(Config(/*wait_for_space=*/true));
    const auto filled = cluster.Fill();
    ASSERT_GE(filled.size(), 2u);

    auto pending = std::async(std::launch::async, [&cluster] {
        return cluster.client().BatchPutStart(
            {"late-0", "late-1"}, {{kObjectSize}, {kObjectSize}}, OneReplica());
    });
    EXPECT_FALSE(Ready(pending, kStillParked));

    ASSERT_TRUE(cluster.service().Remove(filled[0]).has_value());
    EXPECT_FALSE(Ready(pending, kStillParked));
    ASSERT_TRUE(cluster.service().Remove(filled[1]).has_value());

    ASSERT_TRUE(Ready(pending, kAdmitted));
    auto results = pending.get();
    ASSERT_EQ(results.size(), 2u);
    for (const auto& result : results) {
        ASSERT_TRUE(result.has_value()) << toString(result.error());
    }
}

TEST(PutWaitForSpaceTest, PutWaitsForEvictionToFreeSpace) {
    auto config = Config(/*wait_for_space=*/true);
    config.eviction_ratio = 0.5;
    PutWaitCluster cluster(config);
    cluster.Fill();

    auto pending = cluster.PutAsync("late");
    ASSERT_TRUE(Ready(pending, kAdmitted));
    auto result = pending.get();
    ASSERT_TRUE(result.has_value()) << toString(result.error());
}

TEST(PutWaitForSpaceTest, PutWaitsForOffloadToFreeSpace) {
    auto config = Config(/*wait_for_space=*/true);
    config.enable_offload = true;
    config.offload_on_evict = true;
    config.eviction_ratio = 0.5;
    PutWaitCluster cluster(config);
    ASSERT_TRUE(cluster.service()
                    .MountLocalDiskSegment(cluster.owner_id(),
                                           /*enable_offloading=*/true)
                    .has_value());
    cluster.Fill();

    // Eviction can only queue the objects for offload, which frees nothing.
    auto pending = cluster.PutAsync("late");
    EXPECT_FALSE(Ready(pending, kStillParked));

    std::vector<OffloadTaskItem> tasks;
    const auto deadline = std::chrono::steady_clock::now() + kAdmitted;
    while (tasks.empty() && std::chrono::steady_clock::now() < deadline) {
        auto fetched = cluster.service().OffloadObjectHeartbeat(
            cluster.owner_id(), /*enable_offloading=*/true);
        ASSERT_TRUE(fetched.has_value()) << toString(fetched.error());
        tasks = std::move(fetched.value());
        if (tasks.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }
    ASSERT_FALSE(tasks.empty());
    EXPECT_FALSE(Ready(pending, std::chrono::milliseconds(0)));

    std::vector<StorageObjectMetadata> metadatas;
    for (const auto& task : tasks) {
        metadatas.push_back({.bucket_id = 0,
                             .offset = 0,
                             .key_size = static_cast<int64_t>(task.key.size()),
                             .data_size = task.size,
                             .transport_endpoint = "put_wait_disk"});
    }
    ASSERT_TRUE(cluster.service()
                    .NotifyOffloadSuccess(cluster.owner_id(), tasks, metadatas)
                    .has_value());

    ASSERT_TRUE(Ready(pending, kAdmitted));
    auto result = pending.get();
    ASSERT_TRUE(result.has_value()) << toString(result.error());
}

TEST(PutWaitForSpaceTest, DepartedClientLeavesNoAllocation) {
    PutWaitCluster cluster(Config(/*wait_for_space=*/true));
    const auto filled = cluster.Fill();

    auto closed = std::make_shared<std::atomic<bool>>(false);
    auto future = cluster.service().PutStartWaitingForSpace(
        cluster.owner_id(), "late", kObjectSize, OneReplica(), "default",
        [closed] { return closed->load(); });
    std::this_thread::sleep_for(kStillParked);
    ASSERT_FALSE(future.hasResult());

    closed->store(true);
    ASSERT_TRUE(cluster.service().Remove(filled.front()).has_value());
    const auto deadline = std::chrono::steady_clock::now() + kAdmitted;
    while (!future.hasResult() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_TRUE(future.hasResult());
    auto result = std::move(future).get();
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), ErrorCode::RPC_FAIL);

    // The freed object's space is still there for exactly one more object.
    EXPECT_TRUE(cluster.service()
                    .PutStart(cluster.owner_id(), "replacement", kObjectSize,
                              OneReplica())
                    .has_value());
    auto overflow = cluster.service().PutStart(cluster.owner_id(), "overflow",
                                               kObjectSize, OneReplica());
    ASSERT_FALSE(overflow.has_value());
    EXPECT_EQ(overflow.error(), ErrorCode::NO_AVAILABLE_HANDLE);
}

TEST(PutWaitForSpaceTest, FullQueueRejectsWithNoAvailableHandle) {
    auto config = Config(/*wait_for_space=*/true);
    config.put_wait_queue_max_requests = 1;
    PutWaitCluster cluster(config);
    const auto filled = cluster.Fill();

    auto parked = cluster.PutAsync("parked");
    EXPECT_FALSE(Ready(parked, kStillParked));

    auto rejected = cluster.PutAsync("rejected");
    ASSERT_TRUE(Ready(rejected, std::chrono::seconds(5)));
    auto rejected_result = rejected.get();
    ASSERT_FALSE(rejected_result.has_value());
    EXPECT_EQ(rejected_result.error(), ErrorCode::NO_AVAILABLE_HANDLE);

    ASSERT_TRUE(cluster.service().Remove(filled.front()).has_value());
    ASSERT_TRUE(Ready(parked, kAdmitted));
    auto parked_result = parked.get();
    ASSERT_TRUE(parked_result.has_value()) << toString(parked_result.error());
}

TEST(PutWaitForSpaceTest, PutThatCanNeverFitFailsAtOnce) {
    PutWaitCluster cluster(Config(/*wait_for_space=*/true));
    cluster.Fill();

    auto pending = cluster.PutAsync("huge", 2 * kSegmentSize);
    ASSERT_TRUE(Ready(pending, std::chrono::seconds(5)));
    auto result = pending.get();
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), ErrorCode::NO_AVAILABLE_HANDLE);
}

}  // namespace
}  // namespace mooncake
