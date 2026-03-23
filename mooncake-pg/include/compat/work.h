#pragma once

#include <ATen/ATen.h>
#include <chrono>
#include <mutex>
#include <vector>

constexpr auto kNoTimeout = std::chrono::milliseconds(0);

namespace c10d {

enum class OpType : std::uint8_t {
  BROADCAST = 0,
  ALLREDUCE = 1,
  ALLREDUCE_COALESCED = 2,
  REDUCE = 3,
  ALLGATHER = 4,
  _ALLGATHER_BASE = 5,
  ALLGATHER_COALESCED = 6,
  GATHER = 7,
  SCATTER = 8,
  REDUCE_SCATTER = 9,
  ALLTOALL_BASE = 10,
  ALLTOALL = 11,
  SEND = 12,
  RECV = 13,
  RECVANYSOURCE = 14,
  BARRIER = 15,
  _REDUCE_SCATTER_BASE = 16,
  COALESCED = 17,
  _ALLREDUCE_SPARSE = 18,
  UNKNOWN = 100,
};

// Please do not use Work API, it is going away, to be
// replaced by ivalue::Future.
// Python binding for this class might change, please do not assume
// this will be bound using pybind.
class Work : public torch::CustomClassHolder {
 public:
  Work(
      int rank = -1,
      OpType opType = OpType::UNKNOWN) :
      rank_(rank), opType_(opType) {}

  ~Work() override = default;

  // Checks if request has completed. Non-blocking operation.
  virtual bool isCompleted() {
    std::lock_guard<std::mutex> lock(mutex_);
    return completed_;
  }

  // Returns if the work completed successfully.
  // If false, the exception function can be called to get details.
  virtual bool isSuccess() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return !exception_;
  }

  // Returns exception if isSuccess() returned false.
  virtual std::exception_ptr exception() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return exception_;
  }

  // Returns source rank if this objects represents a recv-from-any.
  virtual int sourceRank() const {
    TORCH_CHECK(
      false,
      "sourceRank() may only be called on work objects "
      "that correspond to a recv or recv-from-any call.");
  }

  // Returns result tensors, if applicable.
  // If work is not supposed to have result, we return empty list.
  virtual std::vector<at::Tensor> result() {
    TORCH_CHECK(false, "result() not implemented.");
  }

  // Ensures that operations on the output tensors that are invoked
  // after this function returns are correctly sequenced after the
  // asynchronous completion of this work.
  //
  // For CUDA tensors, it inserts stream synchronization such that
  // the streams of the caller wait for completion of the
  // asynchronous operations on the destination tensors.
  //
  // For CPU tensors, it is currently a nop.
  //
  // This function should only be used if the caller polls for
  // completion through the `isCompleted` function, it has returned
  // true, and the `isSuccess` function also has returned true.
  //
  virtual void synchronize() {}

  // Waits until request completes. Blocking operation.
  // Throws if the work completed with an exception.
  // Returns false if the work is aborted.
  // Otherwise, it always returns true, indicating the work is completed.
  //
  // Functionally equivalent to:
  //
  //   while (!isCompleted()) { /* nop */ }
  //   auto success = isSuccess();
  //   if (!success) { std::rethrow_exception(exception()); }
  //   return success;
  //
  virtual bool wait(std::chrono::milliseconds timeout = kNoTimeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (timeout == kNoTimeout) {
      // This waits without a timeout.
      cv_.wait(lock, [&] { return completed_; });
    } else {
      // Waits for the user-provided timeout.
      cv_.wait_for(lock, timeout, [&] { return completed_; });
      if (!completed_) {
        // Throw exception if the wait operation timed out and the work was not
        // completed.
        TORCH_CHECK(false, "Operation timed out!");
      }
    }
    if (exception_) {
      std::rethrow_exception(exception_);
    }
    synchronize();
    // Always return true, because abort API is not implemented.
    return true;
  }

  // Blocks the current stream until the work is completed.
  // This is equivalent to synchronize for CUDA tensors but works for both CPU
  // tensors and CUDA tensors by using a spinlock CUDA kernel.
  // This will immediately return.
  // If no stream is active it will throw an error.
  virtual void blockCurrentStream() {
    TORCH_CHECK(false, "Work::blockCurrentStream not implemented.");
  }

  virtual void abort() {
    TORCH_CHECK(false, "Work::abort not implemented.");
  }

  virtual float getDuration() const {
    TORCH_CHECK(false, "This Backend doesn't support getDuration.");
  }

  virtual uint64_t getSequencenumber() const {
    TORCH_CHECK(false, "This Backend doesn't support getSequencenumber.");
  }

  OpType retrieveOpType() const {
    return opType_;
  }

 protected:
  // Completes the work object and optionally sets the exception in a
  // thread-safe manner. Notifies all waiting condition variables as well.
  void finish(std::exception_ptr exception = nullptr) {
    std::unique_lock<std::mutex> lock(mutex_);
    completed_ = true;
    exception_ = std::move(exception);
    lock.unlock();
    cv_.notify_all();
  }

  // Similar to finish, but throws an exception if one is already set or
  // provided by the user.
  void finishAndThrow(std::exception_ptr exception) {
    std::unique_lock<std::mutex> lock(mutex_);
    completed_ = true;
    exception_ = std::move(exception);
    if (exception_) {
      std::rethrow_exception(exception_);
    }
  }

  mutable std::mutex mutex_;
  std::condition_variable cv_;
  bool completed_ = false;
  std::exception_ptr exception_;

  // Current rank of the node.
  const int rank_;

  // Operation type that this work object refers to.
  OpType opType_;
};

} // namespace c10d
