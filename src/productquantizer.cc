/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "productquantizer.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>

namespace fasttext {

real distL2(const real* x, const real* y, int32_t d) {
  real dist = 0;
  for (auto i = 0; i < d; i++) {
    auto tmp = x[i] - y[i];
    dist += tmp * tmp;
  }
  return dist;
}

ProductQuantizer::ProductQuantizer(int32_t dim, int32_t dsub)
    : dim_(dim),
      nsubq_(dim / dsub),
      dsub_(dsub),
      centroids_(dim * ksub_),
      rng(seed_) {
  lastdsub_ = dim_ % dsub;
  if (lastdsub_ == 0) {
    lastdsub_ = dsub_;
  } else {
    nsubq_++;
  }
}

const real* ProductQuantizer::get_centroids(int32_t m, uint8_t i) const {
  if (m == nsubq_ - 1) {
    return &centroids_[m * ksub_ * dsub_ + i * lastdsub_];
  }
  return &centroids_[(m * ksub_ + i) * dsub_];
}

real* ProductQuantizer::get_centroids(int32_t m, uint8_t i) {
  if (m == nsubq_ - 1) {
    return &centroids_[m * ksub_ * dsub_ + i * lastdsub_];
  }
  return &centroids_[(m * ksub_ + i) * dsub_];
}

real ProductQuantizer::assign_centroid(
    const real* x,
    const real* c0,
    uint8_t* code,
    int32_t d) const {
  const real* c = c0;
  real dis = distL2(x, c, d);
  code[0] = 0;
  for (auto j = 1; j < ksub_; j++) {
    c += d;
    real disij = distL2(x, c, d);
    if (disij < dis) {
      code[0] = (uint8_t)j;
      dis = disij;
    }
  }
  return dis;
}

void ProductQuantizer::Estep(
    const real* x,
    const real* centroids,
    uint8_t* codes,
    int32_t d,
    int32_t n) const {
  for (auto i = 0; i < n; i++) {
    assign_centroid(x + i * d, centroids, codes + i, d);
  }
}

void ProductQuantizer::MStep(
    const real* x0,
    real* centroids,
    const uint8_t* codes,
    int32_t d,
    int32_t n) {
  std::vector<int32_t> nelts(ksub_, 0);
  memset(centroids, 0, sizeof(real) * d * ksub_);
  const real* x = x0;
  for (auto i = 0; i < n; i++) {
    auto k = codes[i];
    real* c = centroids + k * d;
    for (auto j = 0; j < d; j++) {
      c[j] += x[j];
    }
    nelts[k]++;
    x += d;
  }

  real* c = centroids;
  for (auto k = 0; k < ksub_; k++) {
    real z = (real)nelts[k];
    if (z != 0) {
      for (auto j = 0; j < d; j++) {
        c[j] /= z;
      }
    }
    c += d;
  }

  std::uniform_real_distribution<> runiform(0, 1);
  for (auto k = 0; k < ksub_; k++) {
    if (nelts[k] == 0) {
      int32_t m = 0;
      while (runiform(rng) * (n - ksub_) >= nelts[m] - 1) {
        m = (m + 1) % ksub_;
      }
      memcpy(centroids + k * d, centroids + m * d, sizeof(real) * d);
      for (auto j = 0; j < d; j++) {
        int32_t sign = (j % 2) * 2 - 1;
        centroids[k * d + j] += sign * eps_;
        centroids[m * d + j] -= sign * eps_;
      }
      nelts[k] = nelts[m] / 2;
      nelts[m] -= nelts[k];
    }
  }
}

void ProductQuantizer::kmeans(const real* x, real* c, int32_t n, int32_t d) {
  std::vector<int32_t> perm(n, 0);
  std::iota(perm.begin(), perm.end(), 0);
  std::shuffle(perm.begin(), perm.end(), rng);
  for (auto i = 0; i < ksub_; i++) {
    memcpy(&c[i * d], x + perm[i] * d, d * sizeof(real));
  }
  auto codes = std::vector<uint8_t>(n);
  for (auto i = 0; i < niter_; i++) {
    Estep(x, c, codes.data(), d, n);
    MStep(x, c, codes.data(), d, n);
  }
}

/**
 * @brief
 * Training process for a Prodcut-Quantization. The process mainly training 
 * a k-means model on an one-dimentsion vector, precisely, mainly executed 
 * on an l2-norm vector for embedding-matrix.
 *
 * @param x
 * The pointer points to the first element of a `Vector` or `Matrix` object.
 * 
 * When doing PQ for l2-norm vector of embedding-matrix, then PQ target is 
 * `n` numbers of elements in `Vector`, each element value represents l2-norm 
 * corresponding one embedding vector in embedding-matrix, and during the 
 * k-means process in PQ, each element will be handled as an one-dimension 
 * vector.
 *
 * When doing PQ for embedding-matrix, the PQ target is a `Matrix` object. 
 * In this case, the minimum data unit is each embedding-vector.
 *
 * @param n
 * The number of records using to execute prodcut-quantization process. 
 * `n` is usually same with the embedding vector number.
 */
void ProductQuantizer::train(int32_t n, const real* x) {
  /// For each subvector collection, he number of items waiting for 
  /// k-means should be larger than target k-means cluster number.
  if (n < ksub_) {
    throw std::invalid_argument(
        "Matrix too small for quantization, must have at least " +
        std::to_string(ksub_) + " rows");
  }
  /// TODO:
  /// `perm` arrange an "id" for each data-unit. `std::iota` will assign 
  /// these data-units' id as [0, 1, 2, ..., n-1].
  ///
  /// In the case of embedding-matrix's l2-norm vector PQ, each element in 
  /// l2-norm vector is an l2-norm value of one embedding vector, which itself 
  /// is an data-unit, and will be seem as an one-dimentional vector during PQ.
  ///
  /// In the case of embedding-matrix PQ, each data-unit is an embedding-vector.
  ///
  /// As a summary, both PQ case will as `n` data-units, and so `n` ids for each 
  /// data unit, `n` equals the number of embedding-vectors. 
  std::vector<int32_t> perm(n, 0);
  std::iota(perm.begin(), perm.end(), 0);
  /// `dsub_` means "dimension of subquantizers/subvectors/subspaces", which 
  /// is the dimension of each sub-vector(sub-space) in product-quantization 
  /// process. 
  /// 
  /// One important point is, the dimension of subvectors can only smaller or 
  /// equal with the dimension of original vactors, so `dsub_` always smaller 
  /// or equal with `dim_`.
  ///
  /// Another important point is, `dim_` should be divided evenly by `dsub_`.
  ///
  /// In the case of l2-norm vector prodcut-quantization, `dsub_` can only 
  /// equals to 1.
  /// In the case of embedding matrix prodcut-quantization, `dsub_` smaller 
  /// or equal with embedding size (embedding-dimension), and could make 
  /// `dim_` be divided evenly by it. 
  auto d = dsub_;
  /// `np` means "number of points" for each iteration EM training of k-means, 
  /// similiar with the notion of "batch-size". But the difference is these 
  /// "batch" is not a subset splited from full data, but sampled from the full 
  /// data. 
  ///
  /// For example, suppose we has `n` embedding vectors, and we want training 
  /// a k-means model for each subquantizer by EM algorithm. If we set, for 
  /// each EM training iteration we the training data size is `np`, what 
  /// fastText do is sampling `np` samples from full data with size `n` by:
  ///   * Shuffling input original matrix (or vector in l2-norm PQ case).
  ///     
  ///     For l2-norm vector case, the shuffing-unit is each element in the 
  ///     `Vector` instance since each element represent a l2-norm value 
  ///     of one embedding-vector, which will be seem as an one-dimentional 
  ///     vector during PQ (and k-means in PQ).
  ///
  ///     For embedding-matrix PQ case, since in `Matrix` instance, the data 
  ///     organized as:
  ///         [[embedding-0-block][embedding-1-block]...[embedding-n-block]]
  ///     and for each vector-element-block, the size is embedding-size.
  ///     So each element group with size as embedding-size will be the 
  ///     minimum suffle-unit.
  /// 
  ///     During PQ process, each embedding will be splitted to several 
  ///     subvectors corresponding each subquantizer. 
  ///
  ///   * Using first `np` sample from shuffled data as training data for 
  ///     current EM training iteration.
  ///
  /// And, the max value of `np` can not larger than `n`, which is the number 
  /// of embedding vectors. The extreme case we only execute single iteration 
  /// EM training algorithm and feeding all data as a "huge" batch with size 
  /// as `n`.
  auto np = std::min(n, max_points_);
  /// `xslice` is the container for holding each training-iteration's training 
  /// data.
  /// A l2-norm vector product-quantization example, suggest we have n embedding 
  /// vectors' l2-norm vector, and each EM k-means training-iteration's number 
  /// of data `np`, and since during PQ for l2-norm vector of embedding0matrix, 
  /// each element will be seem as an one-dimentional vector, so `dsub_` equals 
  /// to one, so an `vector` container with size equals to `np * dsub_` will 
  /// satisfy the training data's requirement.
  auto xslice = std::vector<real>(np * dsub_);
  /// `nsubq_` means "number of subquantizers", which is same with 
  /// subvector number.
  ///
  /// The following for-loop block iterates along each subvector and training 
  /// corresponding subquantizer.
  ///
  /// One important fact is, when executing PQ on embedding-matrix's l2-norm 
  /// vector, the target of PQ is not conventional vector, but some scalars 
  /// which represent certain embedding-vectors l2-norm value. At this time, 
  /// `ProductQuantizer::train` will regards these scalars as one-dimensional  
  /// vectors and executing k-means on them during PQ process. And at the 
  /// same time, since each "vector" is an one-dimension vector, so the number 
  /// of subquantizer/subvector/subspace can only be 1!
  ///
  /// When executing PQ on embedding-matrix, each embedding vector will be 
  /// splitted to several subvectors, so we will have several subquantizers 
  /// to iterate to traing.
  for (auto m = 0; m < nsubq_; m++) { 
    /// Mark the last subquantizer. 
    if (m == nsubq_ - 1) {
      d = lastdsub_;
    }
    /// Shuffle the k-means training data-units by shuffling their "id" saved 
    /// `perm`.
    ///
    /// With certain data-unit's id and the number of current subquantizer, we 
    /// can extract the data we needs to exetute k-means from this method input 
    /// data pointed by `x`. The extracted k-means training data will be put 
    /// into `xslice`, and the training data extracting details will be coded in 
    /// `for (auto j = 0; j < np; j++)` loop.
    if (np != n) {
      std::shuffle(perm.begin(), perm.end(), rng);
    }
    /// The following for-loop shows, for current subquantizer, how to extract 
    /// each k-means training subvector from sampled-input-data pointed by `x` 
    /// and then put extracted subvector input training-data-holder `xslice` 
    /// one-by-one.
    /// 
    /// In each iteration, there are some using variables, following are breifly 
    /// review:
    ///     `x`: The pointer pointing the first element of input `Vector` or 
    ///          input `Matrix`.
    ///     `dim_`: Dimention of target product-quantizing vectors. For 
    ///             embedding-matrix l2-norm vector PQ case, it `dim_` should 
    ///             be 1. For embedding-matrix  PQ case, `dim_` should equal 
    ///             with embedding-dim.
    ///     `dsub_`: PQ's subvector dimension. When PQ target is an l2-norm 
    ///              `Vector`, then `dsub_` can only be 1. When PQ target is 
    ///              embedding-matrix, then `dsub_` could be any value that 
    ///              `dim_` could be divided evenly by it.
    ///     `m`: Current handling subquantizer's number (or id).
    ///     `np`: `np` defines how many samples we holp sampling from the input 
    ///           data pointed by `x`.
    ///     `j`: Current extracting PQ training sample among `np` samples.
    ///     `xslice`: The PQ training samples holder for current training 
    ///               subquantizer.
    ///
    /// The one PQ training sample's extracting process includes following steps:
    ///     1. Decide in `xslice`, which place we should put current extracting 
    ///        sample at. ps: `xslice.data()` represents the pointer points the 
    ///        first element of `xslice`. 
    ///        Since each extracted sample (subvector) has dimension `dsub_`/`d`, 
    ///        so for j-th sample(subvector) among `np`, the location of this 
    ///        this subvector's first element should be located at the address 
    ///        `xslice.data() + j * d`.
    ///
    ///     2. Get current sample/subvector's corresponding range in original 
    ///        vector, in detail, we should get target subvector's start point 
    ///        address in corresponding original vector, and how many element 
    ///        will be extracted as subvector from the start point.
    ///
    ///        For each subquantizer, there is a corresponding subvector, for 
    ///        each subvector, there is a coresponding original vector. 
    ///
    ///        For our current j-th subvector, it's corresponding original vector's 
    ///        first element address in `Vector.data_` or `Matrix.data_` should be 
    ///        `x + perm[j] * dim_`, which is also current subvector's extracting 
    ///        start point. From the starting point address, we will extract 
    ///        `d`/`dsub_` elements as current subvectors.
    ///    
    ///     3. Using `memcpy` to copy the located/extracted subvectors to its 
    ///        corresponding location in PQ training data holder. The PQ training
    ///        data holder organizes data in following style:
    ///            [[subvector_0][subvector_1]...[subvector_np]]
    ///        each subvector's size is `d`/`dsub_`.
    for (auto j = 0; j < np; j++) {
      memcpy(
          xslice.data() + j * d,
          x + perm[j] * dim_ + m * dsub_,
          d * sizeof(real));
    }
    /// Execute k-means to train PQ.
    kmeans(xslice.data(), get_centroids(m, 0), np, d);
  }
}

real ProductQuantizer::mulcode(
    const Vector& x,
    const uint8_t* codes,
    int32_t t,
    real alpha) const {
  real res = 0.0;
  auto d = dsub_;
  const uint8_t* code = codes + nsubq_ * t;
  for (auto m = 0; m < nsubq_; m++) {
    const real* c = get_centroids(m, code[m]);
    if (m == nsubq_ - 1) {
      d = lastdsub_;
    }
    for (auto n = 0; n < d; n++) {
      res += x[m * dsub_ + n] * c[n];
    }
  }
  return res * alpha;
}

void ProductQuantizer::addcode(
    Vector& x,
    const uint8_t* codes,
    int32_t t,
    real alpha) const {
  auto d = dsub_;
  const uint8_t* code = codes + nsubq_ * t;
  for (auto m = 0; m < nsubq_; m++) {
    const real* c = get_centroids(m, code[m]);
    if (m == nsubq_ - 1) {
      d = lastdsub_;
    }
    for (auto n = 0; n < d; n++) {
      x[m * dsub_ + n] += alpha * c[n];
    }
  }
}

void ProductQuantizer::compute_code(const real* x, uint8_t* code) const {
  auto d = dsub_;
  for (auto m = 0; m < nsubq_; m++) {
    if (m == nsubq_ - 1) {
      d = lastdsub_;
    }
    assign_centroid(x + m * dsub_, get_centroids(m, 0), code + m, d);
  }
}

void ProductQuantizer::compute_codes(const real* x, uint8_t* codes, int32_t n)
    const {
  for (auto i = 0; i < n; i++) {
    compute_code(x + i * dim_, codes + i * nsubq_);
  }
}

void ProductQuantizer::save(std::ostream& out) const {
  out.write((char*)&dim_, sizeof(dim_));
  out.write((char*)&nsubq_, sizeof(nsubq_));
  out.write((char*)&dsub_, sizeof(dsub_));
  out.write((char*)&lastdsub_, sizeof(lastdsub_));
  out.write((char*)centroids_.data(), centroids_.size() * sizeof(real));
}

void ProductQuantizer::load(std::istream& in) {
  in.read((char*)&dim_, sizeof(dim_));
  in.read((char*)&nsubq_, sizeof(nsubq_));
  in.read((char*)&dsub_, sizeof(dsub_));
  in.read((char*)&lastdsub_, sizeof(lastdsub_));
  centroids_.resize(dim_ * ksub_);
  for (auto i = 0; i < centroids_.size(); i++) {
    in.read((char*)&centroids_[i], sizeof(real));
  }
}

} // namespace fasttext
