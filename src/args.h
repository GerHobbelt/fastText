/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <istream>
#include <ostream>
#include <string>
#include <unordered_set>
#include <vector>

namespace fasttext {

/// There are two training mode: Supervised and Unsupervised.
/// In unsupervised training, the targe is training word  
/// embeddings with language model, there are two ways, 
/// `cbow` for continuous bag of word, `sg` for skip-gram; 
/// And `sup` represents supervised training in text labeling task.
enum class model_name : int { cbow = 1, sg, sup };
/// `hs` for Hierarchical Softmax, `ns` for Negative Sampling. 
/// `softmax` for Softmax, `ova` for One Vs All.
/// TODO: Figure out what is One Vs All Loss.
enum class loss_name : int { hs = 1, ns, softmax, ova };
enum class metric_name : int {
  f1score = 1,
  f1scoreLabel,
  precisionAtRecall,
  precisionAtRecallLabel,
  recallAtPrecision,
  recallAtPrecisionLabel
};

class Args {
 protected:
  std::string boolToString(bool) const;
  std::string modelToString(model_name) const;
  std::string metricToString(metric_name) const;
  std::unordered_set<std::string> manualArgs_;

 public:
  Args();
  /// The `input` may represent different stuff in different running scenario. 
  /// For example, in supervised or unsupervised training scenario, the `input` 
  /// represent training-data path; in product-quantize compression scenario, 
  /// `input` represents the pretrained full-size model's path which will be 
  /// compressed by PQ approach.
  std::string input;
  std::string output;
  double lr;
  int lrUpdateRate;
  int dim;
  int ws;
  int epoch;
  int minCount;
  int minCountLabel;
  int neg;
  int wordNgrams;
  loss_name loss;
  model_name model;
  int bucket;
  int minn;
  int maxn;
  int thread;
  double t;
  std::string label;
  int verbose;
  std::string pretrainedVectors;
  bool saveOutput;
  int seed;

  bool qout;
  bool retrain;
  bool qnorm;
  size_t cutoff;
  size_t dsub;

  std::string autotuneValidationFile;
  std::string autotuneMetric;
  int autotunePredictions;
  int autotuneDuration;
  std::string autotuneModelSize;

  void parseArgs(const std::vector<std::string>& args);
  void printHelp();
  void printBasicHelp();
  void printDictionaryHelp();
  void printTrainingHelp();
  void printAutotuneHelp();
  void printQuantizationHelp();
  void save(std::ostream&);
  void load(std::istream&);
  void dump(std::ostream&) const;
  bool hasAutotune() const;
  bool isManual(const std::string& argName) const;
  void setManual(const std::string& argName);
  std::string lossToString(loss_name) const;
  metric_name getAutotuneMetric() const;
  std::string getAutotuneMetricLabel() const;
  double getAutotuneMetricValue() const;
  int64_t getAutotuneModelSize() const;

  /// The default upper limit of model size
  static constexpr double kUnlimitedModelSize = -1.0;
};
} // namespace fasttext
