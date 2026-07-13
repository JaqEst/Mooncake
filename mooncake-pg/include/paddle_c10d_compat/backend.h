#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>
#include <utility>

#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>
#include <paddle_c10d_compat/intrusive_ptr.h>
#include <paddle_c10d_compat/store.h>
#include <paddle_c10d_compat/types.h>
#include <paddle_c10d_compat/work.h>

constexpr auto kBackendDefaultTimeout =
    std::chrono::milliseconds(30 * 60 * 1000);

namespace c10d {

class Backend {
 public:
  struct Options {
    explicit Options(
        std::string backend,
        std::chrono::milliseconds timeout = kBackendDefaultTimeout)
        : timeout(timeout), backend(std::move(backend)) {}
    virtual ~Options() = default;

    std::chrono::milliseconds timeout;
    const std::string backend;
    std::string group_name;
    std::vector<uint64_t> global_ranks_in_group;
  };

  explicit Backend(int rank, int size) : rank_(rank), size_(size) {}
  virtual ~Backend() = default;

  int getRank() const { return rank_; }
  int getSize() const { return size_; }

  int64_t getID() const {
    return reinterpret_cast<std::intptr_t>(this);
  }

  virtual bool supportsCoalescing() const {
    return false;
  }

  virtual const std::string getBackendName() const {
    TORCH_INTERNAL_ASSERT(false, "getBackendName() is not implemented.");
  }

  virtual c10::intrusive_ptr<Options> getBackendOptions() {
    TORCH_CHECK(false, "Backend ", getBackendName(),
                " does not implement getBackendOptions.");
  }

  virtual c10::intrusive_ptr<Work> send(std::vector<at::Tensor>& /* tensors */,
                                        int /* dstRank */, int /* tag */) {
    TORCH_CHECK(false, "Backend ", getBackendName(), " does not support send.");
  }

  virtual c10::intrusive_ptr<Work> recv(std::vector<at::Tensor>& /* tensors */,
                                        int /* srcRank */, int /* tag */) {
    TORCH_CHECK(false, "Backend ", getBackendName(), " does not support recv.");
  }

  virtual c10::intrusive_ptr<Work> recvAnysource(
      std::vector<at::Tensor>& /* tensors */,
      int /* tag */) {
    TORCH_CHECK(false, "Backend ", getBackendName(), " does not support recvAnysource.");
  }

  virtual c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& /* tensors */,
      const BroadcastOptions& /* opts */ = BroadcastOptions()) {
    TORCH_CHECK(false, "Backend ", getBackendName(),
                " does not support broadcast.");
  }

  virtual c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& /* tensors */,
      const AllreduceOptions& /* opts */ = AllreduceOptions()) {
    TORCH_CHECK(false, "Backend ", getBackendName(),
                " does not support allreduce.");
  }

  virtual c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& /* outputTensors */,
      std::vector<at::Tensor>& /* inputTensors */,
      const AllgatherOptions& /* opts */ = AllgatherOptions()) {
    TORCH_CHECK(false, "Backend ", getBackendName(),
                " does not support allgather.");
  }

  virtual c10::intrusive_ptr<Work> _allgather_base(
      at::Tensor& /* outputBuffer */, at::Tensor& /* inputBuffer */,
      const AllgatherOptions& /* opts */ = AllgatherOptions()) {
    TORCH_CHECK(false, "Backend ", getBackendName(),
                " does not support _allgather_base.");
  }

  virtual c10::intrusive_ptr<Work> _reduce_scatter_base(
      at::Tensor& /* outputBuffer */, at::Tensor& /* inputBuffer */,
      const ReduceScatterOptions& /* opts */ = ReduceScatterOptions()) {
    TORCH_CHECK(false, "Backend ", getBackendName(),
                " does not support _reduce_scatter_base.");
  }

  virtual c10::intrusive_ptr<Work> alltoall(
      std::vector<at::Tensor>& /* outputTensors */,
      std::vector<at::Tensor>& /* inputTensors */,
      const AllToAllOptions& /* opts */ = AllToAllOptions()) {
    TORCH_CHECK(false, "Backend ", getBackendName(),
                " does not support alltoall.");
  }

  virtual c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& /* opts */ = BarrierOptions()) {
    TORCH_CHECK(false, "Backend ", getBackendName(),
                " does not support barrier.");
  }

  virtual c10::intrusive_ptr<Work> reduce(
      std::vector<at::Tensor>& /* tensors */,
      const ReduceOptions& /* opts */ = ReduceOptions()) {
    TORCH_CHECK(false, "Backend ", getBackendName(),
                " does not support reduce.");
  }

  virtual c10::intrusive_ptr<Work> gather(
      std::vector<std::vector<at::Tensor>>& /* outputTensors */,
      std::vector<at::Tensor>& /* inputTensors */,
      const GatherOptions& /* opts */ = GatherOptions()) {
    TORCH_CHECK(false, "Backend ", getBackendName(),
                " does not support gather.");
  }

  virtual c10::intrusive_ptr<Work> scatter(
      std::vector<at::Tensor>& /* outputTensors */,
      std::vector<std::vector<at::Tensor>>& /* inputTensors */,
      const ScatterOptions& /* opts */ = ScatterOptions()) {
    TORCH_CHECK(false, "Backend ", getBackendName(),
                " does not support scatter.");
  }

  virtual void shutdown() {}

 protected:
  const int rank_;
  const int size_;
};

}  // namespace c10d
