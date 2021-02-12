/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "loss.h"
#include "utils.h"

#include <cmath>

namespace fasttext {

constexpr int64_t SIGMOID_TABLE_SIZE = 512;
constexpr int64_t MAX_SIGMOID = 8;
constexpr int64_t LOG_TABLE_SIZE = 512;

bool comparePairs(
    const std::pair<real, int32_t>& l,
    const std::pair<real, int32_t>& r) {
  return l.first > r.first;
}

/**
 * @brief Standart log function based on `e`.
 */
real std_log(real x) {
  return std::log(x + 1e-5);
}

Loss::Loss(std::shared_ptr<Matrix>& wo) : wo_(wo) {
  t_sigmoid_.reserve(SIGMOID_TABLE_SIZE + 1);
  for (int i = 0; i < SIGMOID_TABLE_SIZE + 1; i++) {
    real x = real(i * 2 * MAX_SIGMOID) / SIGMOID_TABLE_SIZE - MAX_SIGMOID;
    t_sigmoid_.push_back(1.0 / (1.0 + std::exp(-x)));
  }

  t_log_.reserve(LOG_TABLE_SIZE + 1);
  for (int i = 0; i < LOG_TABLE_SIZE + 1; i++) {
    real x = (real(i) + 1e-5) / LOG_TABLE_SIZE;
    t_log_.push_back(std::log(x));
  }
}

real Loss::log(real x) const {
  if (x > 1.0) {
    return 0.0;
  }
  int64_t i = int64_t(x * LOG_TABLE_SIZE);
  return t_log_[i];
}

real Loss::sigmoid(real x) const {
  if (x < -MAX_SIGMOID) {
    return 0.0;
  } else if (x > MAX_SIGMOID) {
    return 1.0;
  } else {
    int64_t i =
        int64_t((x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2);
    return t_sigmoid_[i];
  }
}

/**
 * @brief
 * Abstract `predict` method for base class. For multi-category problem, the 
 * prediction process is composed with 2 steps.
 * The first step is `computeOutput` which will compute the output of the model, 
 * in fastText case with softmax loss, this may be the logics vector.
 * The second step is finding which label has best chance to be the right 
 * prediction result with `findKBest` and `std::sort_heap` process.
 */
void Loss::predict(
    int32_t k,
    real threshold,
    Predictions& heap,
    Model::State& state) const {
  computeOutput(state);
  findKBest(k, threshold, heap, state.output);
  std::sort_heap(heap.begin(), heap.end(), comparePairs);
}

/**
 * @brief
 * Find top k most possible prediction results(labels), the process 
 * could be executed in multi-threading mode.
 *
 * @param k k for top-k most possible prediction results.
 * @param threshold The least value that a logic's corresponding could 
 *   be seleted as an potential prediction result.
 * @param heap Holding the results of top-k possible labels, which is a 
 *   struct as `std::vector< std::pair<real, int32_t> >`.
 * @param output Holding the model output, which is the softmax function 
 *   result of logics vector.
 */
void Loss::findKBest(
    int32_t k,
    real threshold,
    Predictions& heap,
    const Vector& output) const {
  for (int32_t i = 0; i < output.size(); i++) {
    if (output[i] < threshold) {
      continue;
    }
    /// `heap` holds top-k possible prediction results, so k is its max 
    /// size, unless the value of log(current_softmax) is larger than the 
    /// least value in `heap`.
    if (heap.size() == k && std_log(output[i]) < heap.front().first) {
      continue;
    }
    /// Push back the new potential prediction results and its correponding 
    /// log(softmax_value).
    heap.push_back(std::make_pair(std_log(output[i]), i));
    /// Sorting the just pushed element.
    std::push_heap(heap.begin(), heap.end(), comparePairs);
    /// Removing extra prediction results which possibiliy is ranking 
    /// larger than k.
    if (heap.size() > k) {
      std::pop_heap(heap.begin(), heap.end(), comparePairs);
      heap.pop_back();
    }
  }
}

BinaryLogisticLoss::BinaryLogisticLoss(std::shared_ptr<Matrix>& wo)
    : Loss(wo) {}

real BinaryLogisticLoss::binaryLogistic(
    int32_t target,
    Model::State& state,
    bool labelIsPositive,
    real lr,
    bool backprop) const {
  real score = sigmoid(wo_->dotRow(state.hidden, target));
  if (backprop) {
    real alpha = lr * (real(labelIsPositive) - score);
    state.grad.addRow(*wo_, target, alpha);
    wo_->addVectorToRow(state.hidden, target, alpha);
  }
  if (labelIsPositive) {
    return -log(score);
  } else {
    return -log(1.0 - score);
  }
}

void BinaryLogisticLoss::computeOutput(Model::State& state) const {
  Vector& output = state.output;
  output.mul(*wo_, state.hidden);
  int32_t osz = output.size();
  for (int32_t i = 0; i < osz; i++) {
    output[i] = sigmoid(output[i]);
  }
}

OneVsAllLoss::OneVsAllLoss(std::shared_ptr<Matrix>& wo)
    : BinaryLogisticLoss(wo) {}

real OneVsAllLoss::forward(
    const std::vector<int32_t>& targets,
    int32_t /* we take all targets here */,
    Model::State& state,
    real lr,
    bool backprop) {
  real loss = 0.0;
  int32_t osz = state.output.size();
  for (int32_t i = 0; i < osz; i++) {
    bool isMatch = utils::contains(targets, i);
    loss += binaryLogistic(i, state, isMatch, lr, backprop);
  }

  return loss;
}

NegativeSamplingLoss::NegativeSamplingLoss(
    std::shared_ptr<Matrix>& wo,
    int neg,
    const std::vector<int64_t>& targetCounts)
    : BinaryLogisticLoss(wo), neg_(neg), negatives_(), uniform_() {
  real z = 0.0;
  for (size_t i = 0; i < targetCounts.size(); i++) {
    z += pow(targetCounts[i], 0.5);
  }
  for (size_t i = 0; i < targetCounts.size(); i++) {
    real c = pow(targetCounts[i], 0.5);
    for (size_t j = 0; j < c * NegativeSamplingLoss::NEGATIVE_TABLE_SIZE / z;
         j++) {
      negatives_.push_back(i);
    }
  }
  uniform_ = std::uniform_int_distribution<size_t>(0, negatives_.size() - 1);
}

real NegativeSamplingLoss::forward(
    const std::vector<int32_t>& targets,
    int32_t targetIndex,
    Model::State& state,
    real lr,
    bool backprop) {
  assert(targetIndex >= 0);
  assert(targetIndex < targets.size());
  int32_t target = targets[targetIndex];
  real loss = binaryLogistic(target, state, true, lr, backprop);

  for (int32_t n = 0; n < neg_; n++) {
    auto negativeTarget = getNegative(target, state.rng);
    loss += binaryLogistic(negativeTarget, state, false, lr, backprop);
  }
  return loss;
}

int32_t NegativeSamplingLoss::getNegative(
    int32_t target,
    std::minstd_rand& rng) {
  int32_t negative;
  do {
    negative = negatives_[uniform_(rng)];
  } while (target == negative);
  return negative;
}

HierarchicalSoftmaxLoss::HierarchicalSoftmaxLoss(
    std::shared_ptr<Matrix>& wo,
    const std::vector<int64_t>& targetCounts)
    : BinaryLogisticLoss(wo),
      paths_(),
      codes_(),
      tree_(),
      osz_(targetCounts.size()) {
  buildTree(targetCounts);
}

void HierarchicalSoftmaxLoss::buildTree(const std::vector<int64_t>& counts) {
  tree_.resize(2 * osz_ - 1);
  for (int32_t i = 0; i < 2 * osz_ - 1; i++) {
    tree_[i].parent = -1;
    tree_[i].left = -1;
    tree_[i].right = -1;
    tree_[i].count = 1e15;
    tree_[i].binary = false;
  }
  for (int32_t i = 0; i < osz_; i++) {
    tree_[i].count = counts[i];
  }
  int32_t leaf = osz_ - 1;
  int32_t node = osz_;
  for (int32_t i = osz_; i < 2 * osz_ - 1; i++) {
    int32_t mini[2] = {0};
    for (int32_t j = 0; j < 2; j++) {
      if (leaf >= 0 && tree_[leaf].count < tree_[node].count) {
        mini[j] = leaf--;
      } else {
        mini[j] = node++;
      }
    }
    tree_[i].left = mini[0];
    tree_[i].right = mini[1];
    tree_[i].count = tree_[mini[0]].count + tree_[mini[1]].count;
    tree_[mini[0]].parent = i;
    tree_[mini[1]].parent = i;
    tree_[mini[1]].binary = true;
  }
  for (int32_t i = 0; i < osz_; i++) {
    std::vector<int32_t> path;
    std::vector<bool> code;
    int32_t j = i;
    while (tree_[j].parent != -1) {
      path.push_back(tree_[j].parent - osz_);
      code.push_back(tree_[j].binary);
      j = tree_[j].parent;
    }
    paths_.push_back(path);
    codes_.push_back(code);
  }
}

real HierarchicalSoftmaxLoss::forward(
    const std::vector<int32_t>& targets,
    int32_t targetIndex,
    Model::State& state,
    real lr,
    bool backprop) {
  real loss = 0.0;
  int32_t target = targets[targetIndex];
  const std::vector<bool>& binaryCode = codes_[target];
  const std::vector<int32_t>& pathToRoot = paths_[target];
  for (int32_t i = 0; i < pathToRoot.size(); i++) {
    loss += binaryLogistic(pathToRoot[i], state, binaryCode[i], lr, backprop);
  }
  return loss;
}

void HierarchicalSoftmaxLoss::predict(
    int32_t k,
    real threshold,
    Predictions& heap,
    Model::State& state) const {
  dfs(k, threshold, 2 * osz_ - 2, 0.0, heap, state.hidden);
  std::sort_heap(heap.begin(), heap.end(), comparePairs);
}

void HierarchicalSoftmaxLoss::dfs(
    int32_t k,
    real threshold,
    int32_t node,
    real score,
    Predictions& heap,
    const Vector& hidden) const {
  if (score < std_log(threshold)) {
    return;
  }
  if (heap.size() == k && score < heap.front().first) {
    return;
  }

  if (tree_[node].left == -1 && tree_[node].right == -1) {
    heap.push_back(std::make_pair(score, node));
    std::push_heap(heap.begin(), heap.end(), comparePairs);
    if (heap.size() > k) {
      std::pop_heap(heap.begin(), heap.end(), comparePairs);
      heap.pop_back();
    }
    return;
  }

  real f = wo_->dotRow(hidden, node - osz_);
  f = 1. / (1 + std::exp(-f));

  dfs(k, threshold, tree_[node].left, score + std_log(1.0 - f), heap, hidden);
  dfs(k, threshold, tree_[node].right, score + std_log(f), heap, hidden);
}

SoftmaxLoss::SoftmaxLoss(std::shared_ptr<Matrix>& wo) : Loss(wo) {}

/**
 * @brief
 * Compute output for softmax function, saving the result in `state`.
 */
void SoftmaxLoss::computeOutput(Model::State& state) const {
  Vector& output = state.output;
  /// In softmax loss case, the output equals the hidden vector (which is 
  /// the average of all input text's token's id embedding vector, the ids  
  /// are word id and char n-gram bucket ids) multiply with matrix (parameters) 
  /// saved in `wo_`, this will mapping the hidden vector to an logics vector 
  /// which has size of 1 * label_num, each element can be understood as 
  /// an unnormalized score of each label's chance to be the right prediction.
  output.mul(*wo_, state.hidden);
  /// Initialize max score as index zero corresponding logit in logits vector. 
  real max = output[0], z = 0.0;
  int32_t osz = output.size(); /// Output vector size, which is also label number.
  /// Iterate along elements in logics vector `output`, and get the max logic value.
  for (int32_t i = 0; i < osz; i++) {
    max = std::max(output[i], max);
  }
  /// Here is the softmax function calculation process, which will using logics 
  /// to calculate sofemax for each label, sometimes the formular is 
  /// exp(i_th_logic - mean_logic) / sum_of( exp(i_th_logic - mean_logic) ), but 
  /// in fastText case, the author didn't use "mean_logic" but "max_logic" to 
  /// normalize each independant logic
  for (int32_t i = 0; i < osz; i++) {
    output[i] = exp(output[i] - max);
    z += output[i];
  }
  for (int32_t i = 0; i < osz; i++) {
    output[i] /= z;
  }
}

/**
 * @brief
 * Calculate softmax loss value.
 */
real SoftmaxLoss::forward(
    const std::vector<int32_t>& targets,
    int32_t targetIndex,
    Model::State& state,
    real lr,
    bool backprop) {
  computeOutput(state);

  assert(targetIndex >= 0);
  assert(targetIndex < targets.size());
  int32_t target = targets[targetIndex];

  if (backprop) {
    int32_t osz = wo_->size(0);
    for (int32_t i = 0; i < osz; i++) {
      real label = (i == target) ? 1.0 : 0.0;
      real alpha = lr * (label - state.output[i]);
      state.grad.addRow(*wo_, i, alpha);
      wo_->addVectorToRow(state.hidden, i, alpha);
    }
  }
  return -log(state.output[target]);
};

} // namespace fasttext
