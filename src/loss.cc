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

/**
 * @brief
 * Compare each potential prediction results saving in `Predictions` 
 * instance(which is a `std::vector< std::pair<real, int32_t> >` struct). 
 * The comparision target is, with heap ranking algorithm we can put the 
 * most impossible prediction result at the top of heap tree struction, so 
 * we can remove that element(result) with `std::pop_heap`.
 *
 * @param l Left node in heap tree, but this is just a comparision, 
 *   right or left is not important.
 * @param r Right node in heap tree, but this is just a comparision, 
 *   right or left is not important.
 */
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

/**
 * @brief
 * Initializing `wo_` with given parameter `wo`, which is a `Matrix` object 
 * represents matrix than will mapping hidden layer to logits vector (each 
 * element represent not normalized possibility of each label) by 
 * matrix multiplication, so one dim of `wo_` should be same with embedding 
 * dim, and the other dim of `wo_` should be same with the number of labels.
 */
Loss::Loss(std::shared_ptr<Matrix>& wo) : wo_(wo) {
  t_sigmoid_.reserve(SIGMOID_TABLE_SIZE + 1);
  for (int i = 0; i < SIGMOID_TABLE_SIZE + 1; i++) {
    /// TODO: 
    /// Figure out what's following line doing and what's `t_sigmoid` meaning, 
    /// and what's `SIGMOID_TABLE_SIZE`, `MAX_SIGMOID` and `LOG_TABLE_SIZE` 
    /// using for.
    /// My guess is, these three `constexpr` variables will be useful when we 
    /// decide to compress the model with Product-Quantilization (a kind of 
    /// vector quantilization approach), there may have some reference in 
    /// serching engine's PQ method, you know, in PQ, there is a notion sort 
    /// of "booking size".
    /// Can also ref to 
    /// http://ethen8181.github.io/machine-learning/deep_learning/multi_label/product_quantization.html
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
 * For multi-category problem, the prediction process is composed with 2 steps:
 * 1. Execute `computeOutput` which will compute the output of the model, 
 *    in fastText case with softmax loss, this may be the logics vector.
 * 2. Finding which label(categores) has best chance to be the right prediction 
 *    result with `findKBest` and `std::sort_heap` process.
 *    2.1. `findKBest` will given top `k` possible categories which 
 *         model-inference-score is higher than `threshold`, So it's possible that 
 *         it returns less than `k` results since all other categories' 
 *         model-inference-score is less than `threshold`.
 *    2.2. The top-k candidates retured by `findKBest` are not sorted, and 
 *         `std::sort_heap` will sort them with heap algorithm.
 *
 * Can also refer to the annotation of `FastText::predict` and `Model::predict`. 
 *
 * @param k Top k most possible prediction categories.
 * @param threshold The least prediction score threshould, only the categories 
 *   which model-inference-score is higher than `threshold` will consider as the 
 *   candidate prediction results.
 * @param heap The heap data-structure which will be useful to sort prediction 
 *   prediction results according their scores with heap-algorithms and removing 
 *   relative-low score prediction canditates
 */
void Loss::predict(
    int32_t k,
    real threshold,
    Predictions& heap,
    Model::State& state) const {
  computeOutput(state);
  findKBest(k, threshold, heap, state.output);
  /// Re-sorting top-k possible results saving in `std::vector` as a heap.
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
 *   result of logits vector.
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
    /// Building a heap bsaed on `std::vector` instance and a element comparision 
    /// rule, which will automaticaly relocate the last element we just pushed back 
    /// to an appropriate locate in heap's tree structure, for chinese, can ref to 
    /// https://www.jianshu.com/p/65fdd3099238.
    std::push_heap(heap.begin(), heap.end(), comparePairs);
    /// Removing extra prediction results which possibiliy is ranking 
    /// larger than k, this can by done on heap structure by using 
    /// `std::pop_heap` and `std::vector`'s `pop_back` method.  
    /// `std::pop_heap` will put the most impossible element (in this case, that 
    /// will be the first element in heap, based on compare rule in `comparePairs`) 
    /// at the end of `std::vector` instance saving heap info, and after that, 
    /// `pop_back` method will remove that last element, the left elements will 
    /// still forming a heap.
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
  /// saved in `wo_`, this will mapping the hidden vector to an logits vector 
  /// which has size of 1 * label_num, each element can be understood as 
  /// an unnormalized score of each label's chance to be the right prediction.
  output.mul(*wo_, state.hidden);
  /// Initialize max score as index zero corresponding logit in logits vector. 
  real max = output[0], z = 0.0;
  int32_t osz = output.size(); /// Output vector size, which is also label number.
  /// Iterate along elements in logits vector `output`, and get the max logit value.
  for (int32_t i = 0; i < osz; i++) {
    max = std::max(output[i], max);
  }
  /// Here is the softmax function calculation process, which will using logits 
  /// to calculate sofemax for each label, sometimes the formular is 
  /// exp(i_th_logit - mean_logit) / sum_of( exp(i_th_logit - mean_logit) ), but 
  /// in fastText case, the author didn't use "mean_logit" but "max_logit" to 
  /// normalize each independant logit
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
 * Calculate softmax loss value, the parameters are nearly same with the 
 * parameter of `Model::update`.
 * The parameter `targets` and `targetIndex` maybe hard to understand, this 
 * confusion can be figured out by refering  to handling progress in 
 * `FastText::supervised`.
 *
 * @param targets Target label's index, since in fastText case, there has 
 *   multiple labels and each sample can have not only one target label during 
 *   training, so target label ids can be put into a `std::vector<int32_t>`, 
 *   each element in `targets` represents one label id for current training sample. 
 * @param targetIndex Thought for each sample, MAYBE we have several labels, but 
 *   during each time training, we only use one of this labels, which means, is 
 *   we have more than one labels for current sample, we will choose one of these 
 *   lables and assign its corrsponding element in one-hot encoding vector to 1, 
 *   and all other lables‘ corrsponding element in one-hot encoding vector to 0. 
 *   According `FastText::supervised`, the target labels choosing strategy is 
 *   using uniform random choosing. 
 * @param state The data structure using to save some hidden state info, such as 
 *   hidden vector(which is just the hidden layer of the model, caculater by 
 *   averaging each input token-id's embedding vector), embedding dim, etc.
 * @param lr Learning-rate for SGD algorithm.
 * @param backprop If execute back-propogate process to update parameters.
 *
 * TODO: Make sure the understanding about `targets` and `targetIndex` is not wrong. 
 */
real SoftmaxLoss::forward(
    const std::vector<int32_t>& targets,
    int32_t targetIndex,
    Model::State& state,
    real lr,
    bool backprop) {
  /// Compute hidden states (hidden layer), saving results in `state`.
  computeOutput(state);

  /// Check if `targetIndex` value illegal. The min target label id should 
  /// larger or equal than zero, but smaller than labels number.
  assert(targetIndex >= 0); 
  assert(targetIndex < targets.size());
  /// TODO: What's `target` meaning?
  int32_t target = targets[targetIndex];

  if (backprop) {
    /// `wo_->size()` is equal with the number of labels in current task. 
    int32_t osz = wo_->size(0);
    /// NOTE: 
    /// The detail of parameters updating process can refer to paper 
    /// "word2vec Parameter Learning Explained".
    ///
    /// What's the following codes do is, iterate along each label and setting 
    /// all labels to 0 except `targetIndex` pointing label to 1 (this process 
    /// could also understood as iterate along a dynamically building one-hot 
    /// encoding label vector), and then iteratively update parameters accoding 
    /// to SGD.
    /// 
    /// Conclusion 1:
    /// In the page 2 of the ref paper, the text about "...where v0 wj is the j-th 
    /// column of the matrix W'" and formula (2)~(4) shows that, during the 
    /// back propogation process, about the gradient of the parameter matrix using 
    /// to map hidden layer to output layer, for one certain label's corresponding logit, 
    /// it's only related with one certain row in this matrix, this row will multiplied 
    /// with hidden layer(hidden vector) and output the logit for this certain label.
    ///
    /// Based on Conclusion 1, since each gradient vector corresponding to each row 
    /// of parameter matrix (which mapping hidden layer to output layer) is independent 
    /// with each other, we can iteratively update row of parameter matrix (which 
    /// mapping hidden layer to output layer) in a for loop, in each loop, we will 
    /// calculate corresponding hidden-to-output parameter matrix row's gradient, 
    /// multiply with learning-rate (the result saving in `alpha`), and finally plus 
    /// result to corresponding hidden-to-output parameter matrix row to update that 
    /// row's parameters.
    for (int32_t i = 0; i < osz; i++) {
      real label = (i == target) ? 1.0 : 0.0;
      /// NOTE: 
      /// Accoding to formula (5)~(11) in paper "word2vec Parameter Learning Explained", 
      /// we can figure out why `alpha` calculated in this way.
      /// Specifically speaking, the update equation is parameter - lr * gradient
      /// since `Vector` object only has `addRow` method, we should process this 
      /// equation as parameter + (-1 * lr * gradient), and 
      /// gradient = `state.output[i] - label`, so the actually update formula should 
      /// be "parameter + (-1 * lr * (`state.output[i] - label`))", which is equal with 
      /// "parameter + lr * (`label - state.output[i]`))". The detail can ref to 
      /// formula (10) in paper "word2vec Parameter Learning Explained". 
      /// BUT, code reader maybe confuse with why just call `addRow` to add `alpha` 
      /// directly to corresponding parameter matrix row without multiplying with  
      ///
      /// Tips:
      /// 1. `label - state.output[i]` in `alpha` is "e_j" in above paper's formula (8). 
      real alpha = lr * (label - state.output[i]);
      /// Adds (alpha * i-th-row-of-wo_) to `state.grad`.
      /// NOTE: 
      /// Here code-reader may have no idea about why calculate `state.grad` in this way, 
      /// and where and how will using `state.grad`.
      /// 
      /// This two points of confusing is reasonable, since `::forward` method only 
      /// finish part of gradient calculation and cache intermediate result temporally 
      /// in `state.grad`, and `state.grad` is used for only saving the gradient of 
      /// parameters matrix mapping input to hidden layer.
      /// 
      /// The main job of `Loss::forwart` includes:
      ///   1. Calculate but not saving hidden-to-output layer parameters matrix gradient, 
      ///      and update hidden-to-output parameters matrix with that gradient
      ///   2. Get loss function value, which is the log form of target-label likehood. 
      ///   3. Cache the intermediate result in `state.grad` which will be helpful to get 
      ///      the final result of input-to-hidden layer parameters matrix gradient, in this 
      ///      way we can improve computational efficency
      ///
      /// As above discussed, there are three points we should clear: 
      ///   1. The computation of hidden-to-output layer parameters matrix 
      ///      gradients and the updating of hidden-to-output layer parameters have  
      ///      been put together in `Loss::forward`.
      ///   2. The intermediate result of the computataion of input-to-hidden layer 
      ///      parameters matrix gradient has been put in `Loss::forward`, while the 
      ///      rest part of this gradient computation and input-to-hidden layer 
      ///      parameters updating have beem put in `Model::update`.
      /// 
      /// There are 2 advantages for this splitting design:
      ///   1. higher computataional efficency since we can cache some useful variables 
      ///      for future computation. 
      ///   2. We can unify each Loss function's interface and when we need add a new 
      ///      loss function, we can just developing a class satisfy these interface 
      ///      requirements.
      /// 
      /// See detail in https://app.yinxiang.com/fx/19541afc-f298-4511-a90c-d9cd56b06e0b 
      state.grad.addRow(*wo_, i, alpha); /// `state.grad` is a `Vector` object.
      /// NOTE: 
      /// This is `Matrix::addVectorToRow`, NOT `Matrix::addRowToVector`!!!
      ///
      /// Following line updates parameter matrix (which mapping hidden layer to 
      /// output layer) following SGD algorithm, ref to above paper's formula (10), 
      /// which will adds: 
      ///   alpha * state.hidden 
      ///     == leaning-rate * (e * hidden-layer) 
      ///     == leaning-rate * ( (label - state.output[i]) * hidden-layer )
      /// to `wo_` (which is parameter matrix mapping hidden layer to output layer). 
      wo_->addVectorToRow(state.hidden, i, alpha);
    }
  }
  /// Return loss function value, which is the log form of target-label likehood.
  return -log(state.output[target]);
};

} // namespace fasttext
