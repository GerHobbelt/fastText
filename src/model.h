/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "matrix.h"
#include "real.h"
#include "utils.h"
#include "vector.h"

namespace fasttext {

class Loss;

class Model {
 protected:
  /// `wi_` means "w input", which represents the parameters of 
  /// the model's input layer. 
  std::shared_ptr<Matrix> wi_;
  /// `wo_` means "w ouput", which represents the parameters of 
  /// the model's output layer.
  std::shared_ptr<Matrix> wo_; 
  std::shared_ptr<Loss> loss_;
  bool normalizeGradient_; // If normalize the gradients

 public:
  Model(
      std::shared_ptr<Matrix> wi,
      std::shared_ptr<Matrix> wo,
      std::shared_ptr<Loss> loss,
      bool normalizeGradient);
  Model(const Model& model) = delete;
  Model(Model&& model) = delete;
  Model& operator=(const Model& other) = delete;
  Model& operator=(Model&& other) = delete;

  /**
   * @brief
   * The name "State" is sort like "hiden state", but actually this is not 
   * "hidden", `State` just holding some infomation such as gradients, 
   * parameters, loss value, the rule of normalization gradients, etc.
   */
  class State {
   private:
    real lossValue_;
    int64_t nexamples_;

   public:
    Vector hidden;
    Vector output;
    /// NOTE: 
    /// `grad` isn't for the gradient of parameters matrix mapping hidden 
    /// layer to output layer, it's used for saving the gradients of the 
    /// parameters matrix mapping model input to model hidden layer, by 
    /// averaging all input tokens' embedding vectors, which is also the 
    /// parameters matrix saving each tokens embedding vectors.
    Vector grad;
    std::minstd_rand rng;

    State(int32_t hiddenSize, int32_t outputSize, int32_t seed);
    real getLoss() const;
    void incrementNExamples(real loss);
  };

  void predict(
      const std::vector<int32_t>& input,
      int32_t k,
      real threshold,
      Predictions& heap,
      State& state) const;
  void update(
      const std::vector<int32_t>& input,
      const std::vector<int32_t>& targets,
      int32_t targetIndex,
      real lr,
      State& state);
  void computeHidden(const std::vector<int32_t>& input, State& state) const;

  real std_log(real) const;

  static const int32_t kUnlimitedPredictions = -1;
  static const int32_t kAllLabelsAsTarget = -1;
};

} // namespace fasttext
