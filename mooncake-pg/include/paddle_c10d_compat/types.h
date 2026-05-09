#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <ATen/core/Tensor.h>
#include <paddle_c10d_compat/intrusive_ptr.h>
#include <paddle_c10d_compat/store.h>

namespace c10d {

struct ReduceOp {
  enum RedOpType : uint8_t {
    SUM = 0,
    AVG = 1,
    PRODUCT = 2,
    MIN = 3,
    MAX = 4,
    BAND = 5,
    BOR = 6,
    BXOR = 7,
  };

  ReduceOp() = default;
  ReduceOp(RedOpType op) : op_(op) {}

  operator RedOpType() const { return op_; }

  bool operator==(const RedOpType other) const { return op_ == other; }
  bool operator==(const ReduceOp& other) const { return op_ == other.op_; }

  RedOpType op_ = SUM;
};

constexpr auto kUnsetTimeout = std::chrono::milliseconds(-1);

struct BroadcastOptions {
  int64_t rootRank = 0;
  int64_t rootTensor = 0;
  std::chrono::milliseconds timeout = kUnsetTimeout;
  bool asyncOp = true;
};

struct AllreduceOptions {
  ReduceOp reduceOp = ReduceOp::SUM;
  std::chrono::milliseconds timeout = kUnsetTimeout;
  bool asyncOp = true;
  std::optional<at::Tensor> sparseIndices = std::nullopt;
};

struct ReduceOptions {
  ReduceOp reduceOp = ReduceOp::SUM;
  int64_t rootRank = 0;
  int64_t rootTensor = 0;
  std::chrono::milliseconds timeout = kUnsetTimeout;
  bool asyncOp = true;
};

struct AllgatherOptions {
  std::chrono::milliseconds timeout = kUnsetTimeout;
  bool asyncOp = true;
};

struct GatherOptions {
  int64_t rootRank = 0;
  std::chrono::milliseconds timeout = kUnsetTimeout;
  bool asyncOp = true;
};

struct ScatterOptions {
  int64_t rootRank = 0;
  std::chrono::milliseconds timeout = kUnsetTimeout;
  bool asyncOp = true;
};

struct ReduceScatterOptions {
  ReduceOp reduceOp = ReduceOp::SUM;
  std::chrono::milliseconds timeout = kUnsetTimeout;
  bool asyncOp = true;
};

struct AllToAllOptions {
  std::chrono::milliseconds timeout = kUnsetTimeout;
  bool asyncOp = true;
};

struct BarrierOptions {
  std::vector<int64_t> device_ids;
  std::chrono::milliseconds timeout = kUnsetTimeout;
  std::optional<at::Device> device;
  bool asyncOp = true;
};

struct DistributedBackendOptions {
  c10::intrusive_ptr<phi::distributed::Store> store;
  int group_rank;
  int group_size;
  std::chrono::duration<float> timeout;
  std::string group_id;
  std::vector<int64_t> global_ranks_in_group;
};

}  // namespace c10d
