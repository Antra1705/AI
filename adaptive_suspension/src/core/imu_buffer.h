// imu_buffer.h - rolling window of the last 1 second of IMU samples.
//
// Fixed-size circular buffer: O(1) push, no dynamic memory. Samples are always
// read back in time order (at(0) = oldest), so code that compares neighbouring
// samples never compares the newest with the oldest at the wrap point (fixes I-7).
#pragma once
#include "../config.h"
#include "types.h"

class ImuBuffer {
 public:
  enum { CAPACITY = cfg::IMU_BUFFER_SIZE };

  void clear() {
    head_ = 0;
    count_ = 0;
  }

  void push(const ImuSample& s) {
    data_[head_] = s;
    head_ = (head_ + 1) % CAPACITY;
    if (count_ < CAPACITY) ++count_;
  }

  int size() const { return count_; }
  bool full() const { return count_ == CAPACITY; }

  // i = 0 is the oldest sample, size() - 1 the newest.
  const ImuSample& at(int i) const { return data_[(head_ - count_ + i + CAPACITY) % CAPACITY]; }
  const ImuSample& newest() const { return at(count_ - 1); }

 private:
  ImuSample data_[CAPACITY];
  int head_ = 0;   // next slot to write
  int count_ = 0;
};
