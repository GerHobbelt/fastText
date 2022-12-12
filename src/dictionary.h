/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <istream>
#include <memory>
#include <ostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "args.h"
#include "real.h"
#include "language.h"

namespace fasttext {

class Dictionary {
 protected:
  static const int32_t MAX_VOCAB_SIZE = 30000000;
  static const int32_t MAX_LINE_SIZE = 1024;

  int32_t find(const std::string&) const;
  int32_t find(const std::string&, uint32_t h) const;
  void initTableDiscard();
  void initNgrams();
  void reset(std::istream&) const;
  void pushHash(std::vector<int32_t>&, int32_t) const;
  void addSubwords(std::vector<int32_t>&, const std::string&, int32_t) const;

  std::shared_ptr<Args> args_;
  std::vector<int32_t> word2int_;
  std::vector<entry> words_;
  std::vector<int64_t> invalid_; // list of invalid words to remove from matrix

  std::vector<real> pdiscard_;
  int32_t size_;    // size of dictionary, i.e. nwords_ + nlabels_
  int32_t nwords_;  // number of unique words in vocabulary
  int32_t nlabels_; // number of unique labels in vocabulary
  int64_t ntokens_; // number of tokens encountered during training, i.e. sum of word.count for word in words_

  int64_t pruneidx_size_;
  std::unordered_map<int32_t, int32_t> pruneidx_;

 public:
  static const std::string EOS;
  static const std::string BOW;
  static const std::string EOW;

  static const int32_t SKIP_EOS = 0x01;
  static const int32_t SKIP_OOV = 0x02;
  static const int32_t SKIP_FRQ = 0x04;
  static const int32_t SKIP_LNG = 0x08;

  explicit Dictionary(std::shared_ptr<Args>);
  explicit Dictionary(std::shared_ptr<Args>, std::istream&);
  explicit Dictionary(std::shared_ptr<Args>, std::istream&, std::shared_ptr<Language>);
  int32_t nwords() const;
  int32_t nlabels() const;
  int64_t ntokens() const;
  int32_t getId(const std::string&) const;
  int32_t getId(const std::string&, uint32_t h) const;
  entry_type getType(int32_t) const;
  entry_type getType(const std::string&) const;
  bool discard(int32_t, real) const;
  bool checkValidWord(const std::string&, std::shared_ptr<Language>);
  std::string getWord(int32_t) const;
  int64_t getTokenCount(int32_t) const;
  const std::vector<int32_t>& getSubwords(int32_t) const;
  const std::vector<int32_t> getSubwords(const std::string&) const;
  void getSubwords(
      const std::string&,
      std::vector<int32_t>&,
      std::vector<std::string>&) const;
  void computeSubwords(
      const std::string&,
      std::vector<int32_t>&,
      std::vector<std::string>* substrings = nullptr) const;
  void addWordNgrams(
      std::vector<int32_t>& line,
      const std::vector<int32_t>& hashes,
      int32_t n) const;
  void addWordNgrams(
      std::vector<int32_t>& line, 
      const std::vector<int32_t>& hashes,
      int32_t n, 
      int32_t k, 
      std::minstd_rand& rng) const;
  uint32_t hash(const std::string& str) const;
  void add(const std::string&);
  bool readWord(std::istream&, std::string&) const;
  void readFromFile(std::istream&);
  void update(std::shared_ptr<Dictionary>, bool discardOovWords);
  std::string getLabel(int32_t) const;
  void save(std::ostream&) const;
  void load(std::istream&);
  void load(std::istream&, std::shared_ptr<Language>);
  std::vector<int64_t> getCounts(entry_type) const;
  int32_t getLine(std::istream&, std::vector<int32_t>&, std::vector<int32_t>&)
      const;
  int32_t getLineTokens(std::istream&, std::vector<int32_t>&, std::vector<int32_t>&, std::vector<std::string>&)
	const;
  int32_t getLine(std::istream&, std::vector<int32_t>&, std::minstd_rand&)
      const;
  int32_t getLine(std::istream&, std::vector<int32_t>&, std::vector<int32_t>&, std::vector<int32_t>&, std::minstd_rand&, int32_t flags = 0)
    const;
  std::vector<int64_t> getInvalidWords() {
    return invalid_;
  }
  int getInvalidSize() {
    return invalid_.size();
  }
  void clearInvalidWords() {
    invalid_.clear();
  }
  void clip(int32_t, std::shared_ptr<Language>);
  void threshold(int64_t, int64_t);
  void prune(std::vector<int32_t>&);
  bool isPruned() {
    return pruneidx_size_ >= 0;
  }
  void dump(std::ostream&) const;
  void init();
};

} // namespace fasttext
