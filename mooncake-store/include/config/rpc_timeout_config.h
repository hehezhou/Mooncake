#pragma once

#include <chrono>
#include <optional>

namespace mooncake {

struct RpcTimeoutConfig {
    // An absent override leaves the caller's connection policy unchanged.
    std::optional<std::chrono::milliseconds> request_timeout;
    std::optional<std::chrono::milliseconds> connect_timeout;
    // Overrides request_timeout for PutStart and BatchPutStart; negative
    // disables their timer.
    std::optional<std::chrono::milliseconds> put_start_request_timeout;

    static RpcTimeoutConfig FromEnvironment();
};

}  // namespace mooncake
