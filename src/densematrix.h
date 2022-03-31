/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <assert.h>
#include <cstdint>
#include <iostream>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <vector>

#include "matrix.h"
#include "real.h"

namespace fasttext {

class Vector;

class DenseMatrix : public Matrix {
 protected:
  std::vector<real> data_;
  void uniformThread(real, int, int32_t);

 public:
  DenseMatrix();
  explicit DenseMatrix(int64_t, int64_t);
  explicit DenseMatrix(int64_t m, int64_t n, real* dataPtr);
  DenseMatrix(const DenseMatrix&) = default;
  DenseMatrix(DenseMatrix&&) noexcept;
  DenseMatrix& operator=(const DenseMatrix&) = delete;
  DenseMatrix& operator=(DenseMatrix&&) = delete;
  virtual ~DenseMatrix() noexcept override = default;

  inline real* data() {
    return data_.data();
  }
  inline const real* data() const {
    return data_.data();
  }

  inline const real& at(int64_t i, int64_t j) const {
    assert(i * n_ + j < data_.size());
    return data_[i * n_ + j];
  };
  inline real& at(int64_t i, int64_t j) {
    return data_[i * n_ + j];
  };
  inline void eraseRow(int64_t row) {
    assert(row * n_ + n_ < data_.size());
    data_.erase(data_.begin() + row * n_, data_.begin() + row * n_ + n_);
  }
  inline void eraseRows(std::vector<int64_t> rows) {
    size_t rm_index = 0;
    int64_t count = 0;
    int64_t deleted = 0;
    data_.erase(
      std::remove_if(std::begin(data_), std::end(data_), [&](real& elem)
      {
          if (rm_index < rows.size() && int((&elem - &data_[0]) / n_) == rows[rm_index])
          {
            count++;
            if(count == n_) {
              rm_index++;
              count = 0;
              deleted++;
            }
            if(rm_index % 10000 == 0) {
              std::cerr << "\rRemoved " << rm_index << " rows..." << std::flush;
            }
            return true;
          }
          return false;
      }),
      std::end(data_)
    );
    m_ = m_ - deleted;
  }

  inline int64_t rows() const {
    return m_;
  }
  inline int64_t cols() const {
    return n_;
  }
  void zero();
  void uniform(real, unsigned int, int32_t);

  void multiplyRow(const Vector& nums, int64_t ib = 0, int64_t ie = -1);
  void divideRow(const Vector& denoms, int64_t ib = 0, int64_t ie = -1);

  real l2NormRow(int64_t i) const;
  void l2NormRow(Vector& norms) const;

  real dotRow(const Vector&, int64_t) const override;
  void filterRows(const std::vector<int64_t>) override;
  void addVectorToRow(const Vector&, int64_t, real) override;
  void addRowToVector(Vector& x, int32_t i) const override;
  void addRowToVector(Vector& x, int32_t i, real a) const override;
  void save(std::ostream&) const override;
  void load(std::istream&) override;
  void dump(std::ostream&) const override;

  class EncounteredNaNError : public std::runtime_error {
   public:
    EncounteredNaNError() : std::runtime_error("Encountered NaN.") {}
  };
};
} // namespace fasttext
