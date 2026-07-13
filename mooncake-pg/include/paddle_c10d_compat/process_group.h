#pragma once

#include <chrono>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <c10/util/Exception.h>
#include <paddle_c10d_compat/backend.h>
#include <paddle_c10d_compat/store.h>
#include <paddle_c10d_compat/types.h>
#include <paddle_c10d_compat/work.h>

namespace c10d {

class ProcessGroup {
 public:
  enum BackendType : uint8_t {
    UNDEFINED = 0,
    GLOO = 1,
    NCCL = 2,
    UCC = 3,
    MPI = 4,
    XCCL = 5,
    CUSTOM = 6,
  };

  static std::string backendTypeToString(const BackendType& type) {
    switch (type) {
      case BackendType::GLOO:
        return "gloo";
      case BackendType::NCCL:
        return "nccl";
      case BackendType::XCCL:
        return "xccl";
      case BackendType::UCC:
        return "ucc";
      case BackendType::MPI:
        return "mpi";
      case BackendType::UNDEFINED:
        return "undefined";
      case BackendType::CUSTOM:
        return "custom";
      default:
        TORCH_CHECK(false, "Unknown process group backend type.");
    }
  }

  static BackendType strToBackendType(const std::string& backend) {
    if (backend == "undefined") {
      return BackendType::UNDEFINED;
    } else if (backend == "gloo") {
      return BackendType::GLOO;
    } else if (backend == "nccl") {
      return BackendType::NCCL;
    } else if (backend == "xccl") {
      return BackendType::XCCL;
    } else if (backend == "ucc") {
      return BackendType::UCC;
    } else if (backend == "mpi") {
      return BackendType::MPI;
    } else {
      return BackendType::CUSTOM;
    }
  }

  explicit ProcessGroup(int rank, int size)
      : rank_(rank), size_(size), backendType_(BackendType::UNDEFINED) {}

  explicit ProcessGroup(c10::intrusive_ptr<::c10d::Store> store, int rank,
                        int size)
      : store_(std::move(store)),
        rank_(rank),
        size_(size),
        backendType_(BackendType::UNDEFINED) {}

  virtual ~ProcessGroup() = default;

  virtual int getRank() const {
    if (backendType_ == BackendType::UNDEFINED) {
      return rank_;
    }
    return getDefaultBackend()->getRank();
  }

  virtual int getSize() const {
    if (backendType_ == BackendType::UNDEFINED) {
      return size_;
    }
    return getDefaultBackend()->getSize();
  }

  int64_t getID() const { return reinterpret_cast<std::intptr_t>(this); }

  int64_t getBackendID(BackendType backendType) const {
    return reinterpret_cast<std::intptr_t>(getBackend(backendType).get());
  }

  virtual const std::string getBackendName() const {
    return backendTypeToString(backendType_);
  }

  BackendType getBackendType() const { return backendType_; }

  virtual c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& /* tensors */,
      const BroadcastOptions& /* opts */ = BroadcastOptions()) {
    TORCH_CHECK(false, "ProcessGroup ", getBackendName(),
                " does not support broadcast.");
  }

  virtual c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& /* tensors */,
      const AllreduceOptions& /* opts */ = AllreduceOptions()) {
    TORCH_CHECK(false, "ProcessGroup ", getBackendName(),
                " does not support allreduce.");
  }

  virtual c10::intrusive_ptr<Work> reduce(
      std::vector<at::Tensor>& /* tensors */,
      const ReduceOptions& /* opts */ = ReduceOptions()) {
    TORCH_CHECK(false, "ProcessGroup ", getBackendName(),
                " does not support reduce.");
  }

  virtual c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& /* outputTensors */,
      std::vector<at::Tensor>& /* inputTensors */,
      const AllgatherOptions& /* opts */ = AllgatherOptions()) {
    TORCH_CHECK(false, "ProcessGroup ", getBackendName(),
                " does not support allgather.");
  }

  virtual c10::intrusive_ptr<Work> _allgather_base(
      at::Tensor& /* outputBuffer */, at::Tensor& /* inputBuffer */,
      const AllgatherOptions& /* opts */ = AllgatherOptions()) {
    TORCH_CHECK(false, "ProcessGroup ", getBackendName(),
                " does not support _allgather_base.");
  }

  virtual c10::intrusive_ptr<Work> all_gather_single(
      at::Tensor& outputBuffer, at::Tensor& inputBuffer,
      const AllgatherOptions& opts = AllgatherOptions()) {
    return _allgather_base(outputBuffer, inputBuffer, opts);
  }

  virtual c10::intrusive_ptr<Work> gather(
      std::vector<std::vector<at::Tensor>>& /* outputTensors */,
      std::vector<at::Tensor>& /* inputTensors */,
      const GatherOptions& /* opts */ = GatherOptions()) {
    TORCH_CHECK(false, "ProcessGroup ", getBackendName(),
                " does not support gather.");
  }

  virtual c10::intrusive_ptr<Work> scatter(
      std::vector<at::Tensor>& /* outputTensors */,
      std::vector<std::vector<at::Tensor>>& /* inputTensors */,
      const ScatterOptions& /* opts */ = ScatterOptions()) {
    TORCH_CHECK(false, "ProcessGroup ", getBackendName(),
                " does not support scatter.");
  }

  virtual c10::intrusive_ptr<Work> _reduce_scatter_base(
      at::Tensor& /* outputBuffer */, at::Tensor& /* inputBuffer */,
      const ReduceScatterOptions& /* opts */ = ReduceScatterOptions()) {
    TORCH_CHECK(false, "ProcessGroup ", getBackendName(),
                " does not support _reduce_scatter_base.");
  }

  virtual c10::intrusive_ptr<Work> reduce_scatter_single(
      at::Tensor& outputBuffer, at::Tensor& inputBuffer,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) {
    return _reduce_scatter_base(outputBuffer, inputBuffer, opts);
  }

  virtual c10::intrusive_ptr<c10d::Work> reduce_scatter(
      std::vector<at::Tensor>& /* outputTensors */,
      std::vector<std::vector<at::Tensor>>& /* inputTensors */,
      const c10d::ReduceScatterOptions& /* opts */ = c10d::ReduceScatterOptions()) {
    TORCH_CHECK(false, "ProcessGroup ", getBackendName(),
                " does not support reduce_scatter.");
  }

  virtual c10::intrusive_ptr<Work> alltoall(
      std::vector<at::Tensor>& /* outputTensors */,
      std::vector<at::Tensor>& /* inputTensors */,
      const AllToAllOptions& /* opts */ = AllToAllOptions()) {
    TORCH_CHECK(false, "ProcessGroup ", getBackendName(),
                " does not support alltoall.");
  }

  virtual c10::intrusive_ptr<Work> all_to_all_single(
      at::Tensor& outputBuffer, at::Tensor& inputBuffer,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& opts = AllToAllOptions()) {
    return alltoall_base(outputBuffer, inputBuffer, outputSplitSizes,
                         inputSplitSizes, opts);
  }

  virtual c10::intrusive_ptr<Work> alltoall_base(
      at::Tensor& outputBuffer, at::Tensor& inputBuffer,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& opts = AllToAllOptions()) {
    TORCH_CHECK(false, "ProcessGroup ", getBackendName(),
                " does not support all_to_all_base.");
  }

  virtual c10::intrusive_ptr<Work> send(std::vector<at::Tensor>& /* tensors */,
                                        int /* dstRank */, int /* tag */) {
    TORCH_CHECK(false, "ProcessGroup ", getBackendName(),
                " does not support send.");
  }

  virtual c10::intrusive_ptr<Work> recv(std::vector<at::Tensor>& /* tensors */,
                                        int /* srcRank */, int /* tag */) {
    TORCH_CHECK(false, "ProcessGroup ", getBackendName(),
                " does not support recv.");
  }

  virtual c10::intrusive_ptr<Work> recvAnysource(
      std::vector<at::Tensor>& /* tensors */, int /* tag */) {
    TORCH_CHECK(false, "ProcessGroup ", getBackendName(),
                " does not support recvAnysource.");
  }

  virtual c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& /* opts */ = BarrierOptions()) {
    TORCH_CHECK(false, "ProcessGroup ", getBackendName(),
                " does not support barrier.");
  }

  bool hasBackends() { return !deviceTypeToBackendType_.empty(); }

  bool hasBackendForDeviceType(c10::DeviceType deviceType) {
    return deviceTypeToBackendType_.find(deviceType) !=
        deviceTypeToBackendType_.end();
  }

  void setBackend(c10::DeviceType deviceType, BackendType backendType,
                  const std::optional<c10::intrusive_ptr<Backend>>& backend) {
    deviceTypeToBackendType_[deviceType] = backendType;
    deviceTypes_.insert(deviceType);
    if (backendTypeToBackend_.find(backendType) != backendTypeToBackend_.end()) {
      deviceTypeToBackend_[deviceType] = backendTypeToBackend_.at(backendType);
      return;
    }
    if (backend.has_value()) {
      deviceTypeToBackend_[deviceType] = backend.value();
      backendTypeToBackend_[backendType] = backend.value();
    }
  }

  c10::intrusive_ptr<Backend> getDefaultBackend() const {
    auto iter = backendTypeToBackend_.find(backendType_);
    TORCH_CHECK(iter != backendTypeToBackend_.end(),
                "Could not find default backend type ", backendType_,
                " for ProcessGroup ", getBackendName(), ".");
    return iter->second;
  }

  void setDefaultBackend(const BackendType& backendType) {
    backendType_ = backendType;
  }

  void setDefaultBackend(const std::string& backend) {
    backendType_ = strToBackendType(backend);
  }

  c10::intrusive_ptr<Backend> getBackend(c10::DeviceType deviceType) {
    // If there is a backend associated with this device type then return it
    if (deviceTypeToBackend_.find(deviceType) != deviceTypeToBackend_.end()) {
      return deviceTypeToBackend_.at(deviceType);
    }

    // Get the backend type associated with the device
    BackendType backendType{BackendType::UNDEFINED};
    try {
      backendType = deviceTypeToBackendType_.at(deviceType);
    } catch (const std::out_of_range&) {
      TORCH_CHECK(false, "No backend type associated with device type ",
                  deviceType, ".");
    }

    // Check if the backend has already been initialized
    if (backendTypeToBackend_.find(backendType) != backendTypeToBackend_.end()) {
      auto backend = backendTypeToBackend_.at(backendType);
      deviceTypeToBackend_[deviceType] = backend;
      return backend;
    }

    TORCH_CHECK(false, "Could not retrieve or create the backend ",
                backendType, " for device type ", deviceType, ".");
  }

  c10::intrusive_ptr<Backend> getBackend(BackendType backendType) const {
    auto iter = backendTypeToBackend_.find(backendType);
    TORCH_CHECK(iter != backendTypeToBackend_.end(),
                "Could not find backend type ", uint16_t(backendType),
                " for ProcessGroup ", backendTypeToString(backendType), ".");
    return iter->second;
  }

  std::vector<c10::Device> getDeviceTypes() const {
    std::vector<c10::Device> devices;
    devices.reserve(deviceTypes_.size());
    for (auto deviceType : deviceTypes_) {
      devices.emplace_back(deviceType);
    }
    return devices;
  }

  virtual void shutdown() {
    for (auto& backend : backendTypeToBackend_) {
      backend.second->shutdown();
    }
  }

  c10::intrusive_ptr<c10d::Store> getStore() const { return store_; }

 protected:
  c10::intrusive_ptr<c10d::Store> store_;
  const int rank_;
  const int size_;
  BackendType backendType_;

  std::unordered_set<c10::DeviceType> deviceTypes_;
  std::map<c10::DeviceType, BackendType> deviceTypeToBackendType_;
  std::unordered_map<c10::DeviceType, c10::intrusive_ptr<Backend>>
      deviceTypeToBackend_;
  std::unordered_map<BackendType, c10::intrusive_ptr<Backend>>
      backendTypeToBackend_;
};

}  // namespace c10d
