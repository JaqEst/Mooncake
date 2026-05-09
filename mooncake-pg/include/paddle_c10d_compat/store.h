#pragma once

#include <string>
#include <vector>
#include <utility>
#include <cstdint>

#include <paddle_c10d_compat/intrusive_ptr.h>
#include "paddle/phi/core/distributed/store/store.h"

namespace c10d {
using phi::distributed::Store;

class PrefixStore: public Store {
 public:
  explicit PrefixStore(std::string prefix, c10::intrusive_ptr<Store> store)
      : prefix_(std::move(prefix)), store_(std::move(store)) {}

  int64_t add(const std::string& key, int64_t value) override {
    return store_->add(joinKey(key), value);
  }

  std::vector<uint8_t> get(const std::string& key) override {
    return store_->get(joinKey(key));
  }

  bool check(const std::string& key) override {
    return store_->check(joinKey(key));
  }

  void wait(const std::string& key) override {
    store_->wait(joinKey(key));
  }

  void set(const std::string& key, const std::vector<uint8_t>& value) override {
    store_->set(joinKey(key), value);
  }

  bool deleteKey(const std::string& key) override {
    return store_->deleteKey(joinKey(key));
  }

 protected:
  std::string prefix_;
  c10::intrusive_ptr<Store> store_;

  std::string joinKey(const std::string& key) {
    return prefix_ + "/" + key;
  }
};

} // namespace c10d
