#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <mutex>
#include <utility>
#include <vector>

#include <ATen/ATen.h>
#include <c10/util/Exception.h>

constexpr auto kNoTimeout = std::chrono::milliseconds(0);

namespace c10d {

enum class OpType : std::uint8_t {
  BROADCAST = 0,
  ALLREDUCE = 1,
  REDUCE = 3,
  ALLGATHER = 4,
  _ALLGATHER_BASE = 5,
  GATHER = 7,
  SCATTER = 8,
  REDUCE_SCATTER = 9,
  ALLTOALL_BASE = 10,
  ALLTOALL = 11,
  SEND = 12,
  RECV = 13,
  BARRIER = 15,
  _REDUCE_SCATTER_BASE = 16,
  UNKNOWN = 100,
};

class Work {
 public:
  Work(int rank = -1, OpType opType = OpType::UNKNOWN)
      : rank_(rank), opType_(opType) {}
  virtual ~Work() = default;

  virtual bool isCompleted() {
    std::lock_guard<std::mutex> lock(mutex_);
    return completed_;
  }

  virtual bool isSuccess() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return !exception_;
  }

  virtual std::vector<at::Tensor> result() {
    TORCH_CHECK(false, "Work::result is not implemented.");
  }

  virtual void synchronize() {}

  virtual bool wait(std::chrono::milliseconds timeout = kNoTimeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (timeout == kNoTimeout) {
      cv_.wait(lock, [&] { return completed_; });
    } else {
      cv_.wait_for(lock, timeout, [&] { return completed_; });
      TORCH_CHECK(completed_, "Operation timed out!");
    }
    if (exception_) {
      std::rethrow_exception(exception_);
    }
    lock.unlock();
    synchronize();
    return true;
  }

  OpType retrieveOpType() const { return opType_; }

 protected:
  void finish(std::exception_ptr exception = nullptr) {
    std::unique_lock<std::mutex> lock(mutex_);
    completed_ = true;
    exception_ = std::move(exception);
    lock.unlock();
    cv_.notify_all();
  }

  mutable std::mutex mutex_;
  std::condition_variable cv_;
  bool completed_ = false;
  std::exception_ptr exception_;

  const int rank_;
  OpType opType_;
};

}  // namespace c10d
