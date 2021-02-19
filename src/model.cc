/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "model.h"
#include "loss.h"
#include "utils.h"

#include <algorithm>
#include <stdexcept>

namespace fasttext {

/**
 * @brief 
 * Initializing current state's info, includes:
 *  `hiddenSize` for each token's embedding size;
 *  `outputSize` for output logits vector size, which is same with label number;
 *  `lossValue_` for the sum loss value for all trained sample, so initialized as zero, 
 *  `nexamples_` for totally trained samples number for now, so initialized as zero, 
 *  `grad` for gradients，which initialized as an `Vector` instance with `hiddenSize` as size.
 *  `rng`, for random seed, this will helping randomly choosing on label from each sample's 
 *    one-hot encoding label, the chosen label could be zero or one, the details could 
 *    refer to `FastText::supervised`. `rng` will initialized by `seed` parameter.
 */
Model::State::State(int32_t hiddenSize, int32_t outputSize, int32_t seed)
    : lossValue_(0.0),
      nexamples_(0),
      hidden(hiddenSize),
      output(outputSize),
      grad(hiddenSize),
      rng(seed) {}

/**
 * @brief
 * Calculate loss value as the accumulated loss value of all trained samples' 
 * divided by the number of all trained samples. 
 */
real Model::State::getLoss() const {
  return lossValue_ / nexamples_;
}

/**
 * @brief
 * Doing incremental update for some value, `lossValue_` for accumulated loss values 
 * for all trained samples, and `nexamples_` for number of all trained samples. 
 */
void Model::State::incrementNExamples(real loss) {
  lossValue_ += loss;
  nexamples_++;
}

/**
 * @brief
 * Initializing model with predefined parameters and loss functions.
 */
Model::Model(
    std::shared_ptr<Matrix> wi,
    std::shared_ptr<Matrix> wo,
    std::shared_ptr<Loss> loss,
    bool normalizeGradient)
    : wi_(wi), wo_(wo), loss_(loss), normalizeGradient_(normalizeGradient) {}

/**
 * @brief
 * Compute "hidden state" or "hidden vector" of fastText model (which is nearly 
 * same as CBOW or skip-gram), and putting computation result into `State` instance. 
 */
void Model::computeHidden(const std::vector<int32_t>& input, State& state)
    const {
  Vector& hidden = state.hidden;
  /// Set hidden vector to zero to clear historical cumputational results.
  hidden.zero();
  /// The core computational process is: 
  ///   iterate along each element in input, which is an `std::vector<int32_t>` 
  ///   instance, each element is an token (word id or char n-gram bucket id).
  ///     accumulate each token's embedding vector to `hidden`
  ///
  ///   This "loop-accumulate" style process can help us getting the result of 
  ///   the hidden vector, which is the same as using "multi-hot" form of input 
  ///   vector multiple with embedding matrix, but "loop-accumulate" style only 
  ///   using query and add operator without multiply operator (actually the 
  ///   multiply operate will executed on a lot of zero, it's a huge waste of 
  ///   computational resource), so the performance is much better. 
  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    hidden.addRow(*wi_, *it);
  }
  /// Above process gets the sum of all input tokens' embedding vectors, here we 
  /// divide that value with `input.size()`, which is equal with the input tokens' 
  /// number, so now we get the average of all input tokens' embedding vectors. 
  hidden.mul(1.0 / input.size());
}

/**
 * @brief
 * The predict or inference process. This process consists of two stages. 
 * The first stage is calculation of "hidden vector" mentioned in 
 * `Model::computeHidden`, this gives out an average embedding vector. 
 * The second stage is inference step with given hidden average embedding 
 * vector. This process is executed by `loss_`, which is an Loss instance, 
 * such as a `SoftmaxLoss` instance. The reason of this design is the 
 * second stage will be execute both during inference and forward-propogate 
 * during training, so put it into loss instance can let this step has 
 * reusability.
 *
 * @param input The input of the model, which is a `std::vector<int32_t>` 
 *   instance, each element is a token, which is one of text's word and 
 *   char n-gram bucket id, these ids will be used to query each token's 
 *   embedding vector in embedding vector dict(matrix).
 * @param k Top k most possible prediction categories.
 * @param threshold The least prediction score threshould, only the categories 
 *   which model-inference-score is higher than `threshold` will consider as the 
 *   candidate prediction results.
 * @param heap The heap data-structure which will be useful to sort prediction 
 *   prediction results according their scores with heap-algorithms and removing 
 *   relative-low score prediction canditates.
 * @param state This used to holding "hidden state" info, see details in 
 *   `Model::State::State`. 
 */
void Model::predict(
    const std::vector<int32_t>& input,
    int32_t k,
    real threshold,
    Predictions& heap,
    State& state) const {
  if (k == Model::kUnlimitedPredictions) {
    k = wo_->size(0); // output size
  } else if (k <= 0) {
    throw std::invalid_argument("k needs to be 1 or higher!");
  }
  heap.reserve(k + 1);
  /// Compute hidden vector composed by average value of all input taken ids, 
  /// and saving result in `state` object.
  computeHidden(input, state);
  
  /// Using the hidden vector value saved in `state` to do the inference, 
  /// finding which label has the best chance to be the correct prediction.
  loss_->predict(k, threshold, heap, state);
}

/**
 * @brief 
 * Update model parameters base on SGD algorithm, the parameters are nearly 
 * same with the parameter of `SoftmaxLoss::forward`.
 *
 * @param input Certain text's token ids, which include word id and several 
 *   char n-gram bucket ids.
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
 */
void Model::update(
    const std::vector<int32_t>& input,
    const std::vector<int32_t>& targets,
    int32_t targetIndex,
    real lr,
    State& state) {
  if (input.size() == 0) {
    return;
  }
  /// Compute hidden state (hidden layer, dim info, etc), saving results in `state`.
  computeHidden(input, state); 

  /// Re-initializing all gradients to zero, this is critical in 
  /// multi-threading training mode.
  Vector& grad = state.grad;
  grad.zero();
  /// Calculate loss value for current training sample, updating the parameters 
  /// matrix mapping hidden to output layer, and caching intermediate results 
  /// during calculation of the gradients of parameters matrix mapping input to 
  /// hidden layer (which is also the matrix saving each tokens embedding vectors).
  real lossValue = loss_->forward(targets, targetIndex, state, lr, true);
  /// Update the incremental recording variables.
  state.incrementNExamples(lossValue);

  /// TODO: Figure out the gradient-normalizing meaning.
  if (normalizeGradient_) {
    grad.mul(1.0 / input.size());
  }
  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    /// NOTE:
    /// Following annotation will contain sth refer to the paper 
    /// "word2vec Parameter Learning Explained".
    ///
    /// For now, the i-th row value saved in `grad` is equal to 
    /// wo[i] · leaning-rate * e_i == EH_i, ref formula (12), now we multiply 
    /// EH_i with x_k, which is 1.0 since each x_k is multi-hot encoding vector, 
    /// with input token ids corresponding index be 1.0 and others are 0.0, all 
    /// gradient related with these zero input will be zero, so we just pay 
    /// attention with these x_i which value are 1.0. 
    /// The detail of formula EH_i * x_k == EH_i * 1.0 can ref to formula (14) 
    /// in paper. 
    wi_->addVectorToRow(grad, *it, 1.0);
  }
}

real Model::std_log(real x) const {
  return std::log(x + 1e-5);
}

} // namespace fasttext
