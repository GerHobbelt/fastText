/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstring>
#include <istream>
#include <ostream>
#include <random>
#include <vector>

#include "real.h"
#include "vector.h"

namespace fasttext {

class ProductQuantizer {
 protected:
  /// `nbits_` means how many bit will used to represent the number of 
  /// centroids for (each) sub-vector-space in product-quantization process. 
  /// Here `nbits_ = 8` implies an `int8` data-type can records all centroid 
  /// id for one sub-space in PQ-codebook.
  ///
  /// So, one important point is,  the value of `nbits_` decides how many 
  /// clusters in each sub-vector-space during k-means training.
  ///
  /// In this way, even if each element in original vector is `float32`, we can 
  /// represent these each element with an `int8` variable, each `int8` variable 
  /// is an centroid id which corresponding to an centriod vector 
  /// (cluster-center-vector) recorded in codebook.
  const int32_t nbits_ = 8;
  /// 1 is 01, `nbits_ = 8`, so `1 << nbits_` is `1 << 8`, and the result is 
  /// `100000000`, which is 256 in decimalism. `ksub_` represents the centroids 
  /// number for each sub-vector-spaces in product-quantization.
  const int32_t ksub_ = 1 << nbits_;
  const int32_t max_points_per_cluster_ = 256;
  const int32_t max_points_ = max_points_per_cluster_ * ksub_;
  const int32_t seed_ = 1234;
  /// K-means EM training iteration times.
  const int32_t niter_ = 25;
  const real eps_ = 1e-7;

  /// TODO: ?
  /// Dimention of target product-quantizing vectors.
  ///
  /// When executing PQ on embedding-matrix's l2-norm vector, each element in 
  /// this l2-norm vector will be seem and handle as an one-dimentional vector 
  /// during PQ (and k-means in PQ) process, so in this case `dim_` equals to 1.
  ///
  /// When executing PQ on embedding-matrix, `dim_` should be same with the 
  /// embedding size (embedding dimention).
  int32_t dim_;
  /// Refer to paper, `nsubq_` means "number of sub-quantizer", this is 
  /// also the number of sub-vectors (sub-spaces) splitted from the 
  /// original vector. 
  int32_t nsubq_;
  /// `dsub_` means "dimension of subquantizers/subvectors/subspaces", which 
  /// is the dimension of each sub-vector(sub-space) in product-quantization 
  /// process. 
  /// 
  /// One import point is, the dimension of subvectors can only smaller or 
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
  int32_t dsub_;  
  int32_t lastdsub_;

  std::vector<real> centroids_;

  std::minstd_rand rng;

 public:
  ProductQuantizer() {}
  ProductQuantizer(int32_t, int32_t);

  real* get_centroids(int32_t, uint8_t);
  const real* get_centroids(int32_t, uint8_t) const;

  real assign_centroid(const real*, const real*, uint8_t*, int32_t) const;
  void Estep(const real*, const real*, uint8_t*, int32_t, int32_t) const;
  void MStep(const real*, real*, const uint8_t*, int32_t, int32_t);
  void kmeans(const real*, real*, int32_t, int32_t);
  void train(int, const real*);

  real mulcode(const Vector&, const uint8_t*, int32_t, real) const;
  void addcode(Vector&, const uint8_t*, int32_t, real) const;
  void compute_code(const real*, uint8_t*) const;
  void compute_codes(const real*, uint8_t*, int32_t) const;

  void save(std::ostream&) const;
  void load(std::istream&);
};

} // namespace fasttext
