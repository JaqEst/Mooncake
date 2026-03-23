#pragma once

#include <memory>

namespace c10 {
template<typename T>
using intrusive_ptr = std::shared_ptr<T>;

template<typename T, typename... Args>
inline intrusive_ptr<T> make_intrusive(Args&&... args) {
  return std::make_shared<T>(std::forward<Args>(args)...);
}

template<typename To, typename From>
inline intrusive_ptr<To> static_intrusive_pointer_cast(intrusive_ptr<From> f) {
  return std::static_pointer_cast<To>(f);
}
} // namespace c10
