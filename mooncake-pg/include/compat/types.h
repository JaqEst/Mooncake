#pragma once

// paddle/include/paddle/common/macros.h defines UNUSED, we should avoid it.
#ifdef UNUSED
#undef UNUSED
#endif

#include <chrono>
#include <cstdint>

#include <ATen/core/Tensor.h>
#include <ATen/core/ivalue.h>

#include <compat/store.h>
#include <compat/intrusive_ptr.h>

namespace c10d {

// Other ReduceOps that need different supplementary data can also
// derive from _SupplementBase.
struct ReduceOp : torch::CustomClassHolder {
  // note(crcrpar): RedOpType could be defined outside of `ReduceOp`
  enum RedOpType : uint8_t {
    SUM = 0,
    AVG = 1,
    PRODUCT = 2,
    MIN = 3,
    MAX = 4,
    BAND = 5, // Bitwise AND
    BOR = 6, // Bitwise OR
    BXOR = 7, // Bitwise XOR
    PREMUL_SUM = 8, // Multiply by a user-supplied constant before summing.
    UNUSED = 9
  };

  ReduceOp() = default;

  ReduceOp(RedOpType op) : op_(op) {
    TORCH_INTERNAL_ASSERT(
        op_ != PREMUL_SUM,
        "Use `torch.distributed._make_nccl_premul_sum` to create an instance of ReduceOp with PREMUL_SUM");
  }

  ReduceOp(const ReduceOp& other) = default;
  ReduceOp& operator=(const ReduceOp& other) = default;

  ReduceOp(ReduceOp&& other) = default;
  ReduceOp& operator=(ReduceOp&& other) = default;
  ~ReduceOp() override = default;

  operator RedOpType() const {
    return op_;
  }

  bool operator==(const std::uint8_t other) {
    TORCH_INTERNAL_ASSERT(other < 9, "Invalid other op value");
    return other == op_;
  }

  bool operator==(const ReduceOp::RedOpType other) {
    return *this == static_cast<std::uint8_t>(other);
  }

  // todo(crcrpar): Handle `RedOpType::PREMUL_SUM` with its scaling factor.
  bool operator==(const ReduceOp& other) {
    return *this == other.op_;
  }

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

struct AllreduceCoalescedOptions : AllreduceOptions {};

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

} // namespace c10d
