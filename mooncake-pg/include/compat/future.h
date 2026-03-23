#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <memory>

namespace c10 {

enum class TypeKind {
  TensorType,
  ListType,
};

class Type;
using TypePtr = std::shared_ptr<Type>;

class Type : public std::enable_shared_from_this<Type> {
 public:
  virtual ~Type() = default;
  virtual TypeKind kind() const = 0;

  virtual bool equals(const Type& other) const {
    return kind() == other.kind();
  }

  bool operator==(const Type& other) const {
    return equals(other);
  }

  bool operator!=(const Type& other) const {
    return !equals(other);
  }
};

class TensorType : public Type {
 private:
  struct PrivateTag {};
 public:
  static TypePtr get() {
   return std::make_shared<TensorType>(PrivateTag{});
  }

  explicit TensorType(PrivateTag) {}

  TypeKind kind() const override { return TypeKind::TensorType; }
};

class ListType : public Type {
 private:
  struct PrivateTag {};
 public:
  static TypePtr create(const TypePtr& elem_type) {
    return std::make_shared<ListType>(PrivateTag{}, elem_type);
  }

  static TypePtr ofTensors() {
    return create(TensorType::get());
  }

  explicit ListType(PrivateTag, TypePtr elem_type)
    : elem_type_(std::move(elem_type)) {}

  TypeKind kind() const override { return TypeKind::ListType; }

  const TypePtr& getElementType() const { return elem_type_; }

  bool equals(const Type& other) const override {
    if (kind() != other.kind()) return false;
    const auto& other_list = static_cast<const ListType&>(other);

    if (!elem_type_ && !other_list.elem_type_) return true;
    if (!elem_type_ || !other_list.elem_type_) return false;
    return elem_type_->equals(*other_list.elem_type_);
  }

 private:
  TypePtr elem_type_;
};

class IValue {
 public:
  IValue() : is_none_(true) {}
  bool isNone() const { return is_none_; }

 private:
  bool is_none_;
};

namespace ivalue {

class Future {
 public:
  explicit Future(TypePtr type = nullptr)
    : type_(std::move(type)),
      completed_(false),
      has_error_(false) {}

  bool completed() const {
    return completed_.load(std::memory_order_acquire);
  }

  void wait() const {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return completed_.load(); });
  }

  void markCompleted(const IValue& value = IValue()) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      value_ = value;
      completed_.store(true, std::memory_order_release);
    }
    cv_.notify_all();
  }

  bool hasError() const { return has_error_.load(); }
  void setError() { has_error_.store(true); }

  const TypePtr& elementType() const { return type_; }
  const IValue& value() const { return value_; }

 private:
  TypePtr type_;
  IValue value_;
  std::atomic<bool> completed_;
  std::atomic<bool> has_error_;
  mutable std::mutex mutex_;
  mutable std::condition_variable cv_;
};

}  // namespace ivalue
}  // namespace c10
