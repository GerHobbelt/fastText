/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "dictionary.h"

#include <assert.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

namespace fasttext {

const std::string Dictionary::EOS = "</s>";
const std::string Dictionary::BOW = "<";
const std::string Dictionary::EOW = ">";

Dictionary::Dictionary(std::shared_ptr<Args> args)
    : args_(args),
      // Initialized `word2int_` as a vector with size as `MAX_VOCAB_SIZE` 
      // and each element value as -1, this helps to map word/word's hash 
      // to word int32 id 
      word2int_(MAX_VOCAB_SIZE, -1),
      size_(0),
      nwords_(0),
      nlabels_(0),
      ntokens_(0),
      pruneidx_size_(-1) {}

Dictionary::Dictionary(std::shared_ptr<Args> args, std::istream& in)
    : args_(args),
      size_(0),
      nwords_(0),
      nlabels_(0),
      ntokens_(0),
      pruneidx_size_(-1) {
  load(in);
}

int32_t Dictionary::find(const std::string& w) const {
  return find(w, hash(w));
}

/**
 * @brief
 * Given a token(word) and a int, value, which could be some meaningful value 
 * or word's certain hash-value(TODO: How to calculate hash value of this word?), 
 * calculate the 'bucket id' of this word based on its hash value with a given 
 * bucket size, which is also the word vocab size, defined by `word2int_`, 
 * `word2int_` is a predefined, fix-sized vector, which size is word-vocab size, 
 * each element's index in `word2int_` corresponds to a bucket id, the value 
 * of the element represents if there already certain word located in certain 
 * bucket, if not, the element of that bucket index should be -1. 
 * 
 * Briefly, we can understand this as a hash-bucket, push each word to different 
 * bucket based on their hash value, the bucket id is the word id. 
 * BUT, this is a SPECIAL HASH_BUCKET STRATEGY, different word COULD NOT BE put in 
 * same bucket, 
 */
int32_t Dictionary::find(const std::string& w, uint32_t h) const {
  int32_t word2intsize = word2int_.size();
  int32_t id = h % word2intsize;
  // If it's an exists bucket id, which means:
  //   1. Current calculated bucket id has been occupied, 
  // and
  //   2. TODO: The word occupied current bucket is not same with current word
  // Then shift current id by plus 1, and re-judgement the above 2 conditions, 
  // until finding an empty bucket~
  //
  // QA: 
  // * Why `words_[word2int_[id]]` represent current exists word's 
  //   corresponding entry?
  // * Since the strategy is based on "plus 1 shift" mode, maybe?
  while (word2int_[id] != -1 && words_[word2int_[id]].word != w) {
    id = (id + 1) % word2intsize;
  }
  return id;
}

/**
 * @brief
 * Adds new or exists word into dictionary following certain rule
 */
void Dictionary::add(const std::string& w) {
  // Gets word-vocab bucket id
  int32_t h = find(w);
  // Update token count
  ntokens_++;
  // If it's the first time this word appearance, then:
  if (word2int_[h] == -1) {
    // Initial a new `entry` instance saving this word's info
    entry e;
    e.word = w;
    e.count = 1;
    e.type = getType(w);
    // Saving this word's `entry` into `words_`, which is word-vocab 
    words_.push_back(e);
    // Assign the latest appearenced word's order to its corresponding 
    // word-bucket's value 
    word2int_[h] = size_++;
  } else {
    // Update current word's appearance count, i.e., word frequency 
    words_[word2int_[h]].count++;
  }
}

int32_t Dictionary::nwords() const {
  return nwords_;
}

int32_t Dictionary::nlabels() const {
  return nlabels_;
}

int64_t Dictionary::ntokens() const {
  return ntokens_;
}

const std::vector<int32_t>& Dictionary::getSubwords(int32_t i) const {
  assert(i >= 0);
  assert(i < nwords_);
  return words_[i].subwords;
}

const std::vector<int32_t> Dictionary::getSubwords(
    const std::string& word) const {
  int32_t i = getId(word);
  if (i >= 0) {
    return getSubwords(i);
  }
  std::vector<int32_t> ngrams;
  if (word != EOS) {
    computeSubwords(BOW + word + EOW, ngrams);
  }
  return ngrams;
}

void Dictionary::getSubwords(
    const std::string& word,
    std::vector<int32_t>& ngrams,
    std::vector<std::string>& substrings) const {
  int32_t i = getId(word);
  ngrams.clear();
  substrings.clear();
  if (i >= 0) {
    ngrams.push_back(i);
    substrings.push_back(words_[i].word);
  }
  if (word != EOS) {
    computeSubwords(BOW + word + EOW, ngrams, &substrings);
  }
}

bool Dictionary::discard(int32_t id, real rand) const {
  assert(id >= 0);
  assert(id < nwords_);
  if (args_->model == model_name::sup) {
    return false;
  }
  return rand > pdiscard_[id];
}

int32_t Dictionary::getId(const std::string& w, uint32_t h) const {
  int32_t id = find(w, h);
  return word2int_[id];
}

int32_t Dictionary::getId(const std::string& w) const {
  int32_t h = find(w);
  return word2int_[h];
}

entry_type Dictionary::getType(int32_t id) const {
  assert(id >= 0);
  assert(id < size_);
  return words_[id].type;
}

entry_type Dictionary::getType(const std::string& w) const {
  return (w.find(args_->label) == 0) ? entry_type::label : entry_type::word;
}

std::string Dictionary::getWord(int32_t id) const {
  assert(id >= 0);
  assert(id < size_);
  return words_[id].word;
}

// The correct implementation of fnv should be:
// h = h ^ uint32_t(uint8_t(str[i]));
// Unfortunately, earlier version of fasttext used
// h = h ^ uint32_t(str[i]);
// which is undefined behavior (as char can be signed or unsigned).
// Since all fasttext models that were already released were trained
// using signed char, we fixed the hash function to make models
// compatible whatever compiler is used.
uint32_t Dictionary::hash(const std::string& str) const {
  uint32_t h = 2166136261;
  for (size_t i = 0; i < str.size(); i++) {
    h = h ^ uint32_t(int8_t(str[i]));
    h = h * 16777619;
  }
  return h;
}

/**
 * @brief
 * Extracting each word's char n-gram feature.
 * 
 * note: 
 *   1. Here the word has been preprocessed, which means is has been appended 
 *      starting and ending sign, "<" and ">".
 *   2. A tip about cpp grammar, here the last parameter is `std::vector<std::string>*`, 
 *      It seems we can just call this method with only 1st and 2nd parameters given, 
 *      at that case, the compiler will infer that the last parameter is an `nullptr` or sth? 
 */
void Dictionary::computeSubwords(
    const std::string& word,
    std::vector<int32_t>& ngrams,
    std::vector<std::string>* substrings) const {
  for (size_t i = 0; i < word.size(); i++) {
    std::string ngram;
    // TODO: Figure why this char filtering rule.
    if ((word[i] & 0xC0) == 0x80) {
      continue;
    }
    for (size_t j = i, n = 1; j < word.size() && n <= args_->maxn; n++) {
      ngram.push_back(word[j++]);
      while (j < word.size() && (word[j] & 0xC0) == 0x80) {
        ngram.push_back(word[j++]);
      }
      if (n >= args_->minn && !(n == 1 && (i == 0 || j == word.size()))) {
        int32_t h = hash(ngram) % args_->bucket;
        pushHash(ngrams, h);
        if (substrings) {
          substrings->push_back(ngram);
        }
      }
    }
  }
}

/**
 * @brief
 * Before now, we have a token-vocab, with very-low frequency token dropped, 
 * but each token is predefined as input with ' ' as sperator. Generally these 
 * token are some words (and labels in surpervise leaning case), according the 
 * paper "Enriching Word Vectors with Subword Information", the word's 
 * morphologically info can be mining better with char-level n-gram feature, 
 * and int this method, there are some simple preprocessing, and 
 * `Dictionary::computeSubwords` helps extracting char n-gram info for each word 
 * in vocab.
 */
void Dictionary::initNgrams() {
  for (size_t i = 0; i < size_; i++) {
    // Adds starting and ending sign for each word, which are "<" and ">".
    std::string word = BOW + words_[i].word + EOW;
    words_[i].subwords.clear();
    // From this place we can see, for each word, the char n-gram saving in 
    // a `std::vector` using 
    // "{1, ${1st word char-ngram}, 2, ${2nd word char-ngram}, ..., n, ${n-th word char-ngram}}" 
    // as format, and see detail about how extracting each word's char n-gram in 
    // `Dictionary::computeSubwords`. 
    words_[i].subwords.push_back(i);
    if (words_[i].word != EOS) {
      computeSubwords(word, words_[i].subwords);
    }
  }
}

bool Dictionary::readWord(std::istream& in, std::string& word) const {
  int c;
  std::streambuf& sb = *in.rdbuf();
  word.clear();
  while ((c = sb.sbumpc()) != EOF) {
    // the ' ' is using to split each tokens(words), since the input data format of 
    // fastText is a label(), and tokens(words) tokenized from a sentence, each of them 
    // splitted with ' '.
    if (c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\v' ||
        c == '\f' || c == '\0') {
      // This condition means the streambuf get sth like '${WORD}\n', so which means
      // we get the last word, or the end of the line(sentence), so we put '${WORD}' 
      // into `word` and initiative not change the 'pointer' of streambuf to next, 
      // so at the next iteration calling `readWord` (means calls 'next' to 'buff-pointer'), 
      // the 'buff-pointer' will point the '\n' with `word` as an empty string.
      if (word.empty()) {
        if (c == '\n') {
          word += EOS;
          return true;
        }
        continue;
      } else {
        if (c == '\n')
          // `sungetc` is sort of file pointer's 'get next' function.
          // This is handling the dirty data format, for example, if we get '${WORD}\n\n', 
          // the first '\n' will be attached with the last word of the line(sentence) as `EOD`, 
          // but the second '\n' is a dirtyn char so we do nothing about it and get the next 
          // stream-buff pointer.
          sb.sungetc();
        return true;
      }
    }
    word.push_back(c);
  }
  // trigger eofbit
  in.get();
  return !word.empty();
}

void Dictionary::readFromFile(std::istream& in) {
  std::string word;
  int64_t minThreshold = 1;
  while (readWord(in, word)) {
    add(word);
    // Log
    if (ntokens_ % 1000000 == 0 && args_->verbose > 1) {
      std::cerr << "\rRead " << ntokens_ / 1000000 << "M words" << std::flush;
    }
    // If current word-vocab is 75% saturated
    if (size_ > 0.75 * MAX_VOCAB_SIZE) {
      minThreshold++;
      // At this place, using same low-frequency threshold to judge if 
      // word or label is "stop word" since its very low frequency
      threshold(minThreshold, minThreshold);
    }
  }
  // The threshold used in `threshold(minThreshold, minThreshold);` is 
  // just used for dynamical control word vocab scale during continiously 
  // add word into dictionary which can control memory and some other computional 
  // resource using , the following finally `threshold` calling set the final 
  // scale of the word-vocab and label vocab 
  threshold(args_->minCount, args_->minCountLabel);
  initTableDiscard();
  initNgrams();
  if (args_->verbose > 0) {
    std::cerr << "\rRead " << ntokens_ / 1000000 << "M words" << std::endl;
    std::cerr << "Number of words:  " << nwords_ << std::endl;
    std::cerr << "Number of labels: " << nlabels_ << std::endl;
  }
  if (size_ == 0) {
    throw std::invalid_argument(
        "Empty vocabulary. Try a smaller -minCount value.");
  }
}

/**
 * @brief
 * This helps to dynamical located some "stop words" (and "stop labels") for 
 * low-frequency words(low-frequency labels) and remove them (i.e. pruning word vocab).
 * This operation will be triggered according some "monitor" variables, 
 * if these variables satisfies certain "saturate" condition, then vocab-pruning 
 * operation will be triggered. After vocab being pruned, these "monitor" variables 
 * value will be reset accoding to new pruned word vocab and waiting next saturate. 
 *
 * So, in briefly, this method helps dynamically dropping words or labels 
 * with very low (absolute) frequency.
 *
 * @param t Word vocab size limit
 * @param tl Label vocab size limit
 */
void Dictionary::threshold(int64_t t, int64_t tl) {
  // Sort all word's in current vocab, high-frequency words have high order
  sort(words_.begin(), words_.end(), [](const entry& e1, const entry& e2) {
    if (e1.type != e2.type) {
      return e1.type < e2.type;
    }
    return e1.count > e2.count;
  });
  // Remove low-frequency words and low-frequency labels
  words_.erase(
      remove_if(
          words_.begin(),
          words_.end(),
          [&](const entry& e) {
            return (e.type == entry_type::word && e.count < t) ||
                (e.type == entry_type::label && e.count < tl);
          }),
      words_.end());
  // stl vector api
  words_.shrink_to_fit();
  // Reset some "quota" counting variables(includes `size_`, `nwords_`, 
  // `nlabels_`, `word2int_`). These variables are reset by: 
  // 
  // Firstly, assign them certain initialization values, and then they 
  // will waiting for next time's saturated which will triggering the 
  // "pruning" operation on word vocab.
  size_ = 0;
  nwords_ = 0;
  nlabels_ = 0;
  std::fill(word2int_.begin(), word2int_.end(), -1);
  // Secondly, iterate alone "pruned" word-vocab and update these variables 
  // at each iteration according the same rule defined in `Dictionary::add`. 
  for (auto it = words_.begin(); it != words_.end(); ++it) {
    int32_t h = find(it->word);
    word2int_[h] = size_++;
    if (it->type == entry_type::word) {
      nwords_++;
    }
    if (it->type == entry_type::label) {
      nlabels_++;
    }
  }
  // After above steps, these variables(`size_`, `nwords_`, `nlabels_`, `word2int_`) 
  // will continiously updated as iteration in `Dictionary::readFromFile` 
  // until they saturated again and trigger "pruning" operation again.
}

/**
 * @brief
 * This helps deciding how discarding or sampling the token according its appearance 
 * proportion among total token appearance times, and with a function, we can 
 * convert this idea to a score save in `pdiscard_` for each token. 
 * NOTE: 
 * This is only an initialize step, there will have some following updates based
 * on current base-value
 *
 * Cheatsheet: `args_->t` represents "sampling threshold".
 * TODO: Figure out sampling for what?
 */
void Dictionary::initTableDiscard() {
  pdiscard_.resize(size_);
  for (size_t i = 0; i < size_; i++) {
    real f = real(words_[i].count) / real(ntokens_);
    pdiscard_[i] = std::sqrt(args_->t / f) + args_->t / f;
  }
}

std::vector<int64_t> Dictionary::getCounts(entry_type type) const {
  std::vector<int64_t> counts;
  for (auto& w : words_) {
    if (w.type == type) {
      counts.push_back(w.count);
    }
  }
  return counts;
}

void Dictionary::addWordNgrams(
    std::vector<int32_t>& line,
    const std::vector<int32_t>& hashes,
    int32_t n) const {
  for (int32_t i = 0; i < hashes.size(); i++) {
    uint64_t h = hashes[i];
    for (int32_t j = i + 1; j < hashes.size() && j < i + n; j++) {
      h = h * 116049371 + hashes[j];
      pushHash(line, h % args_->bucket);
    }
  }
}

void Dictionary::addSubwords(
    std::vector<int32_t>& line,
    const std::string& token,
    int32_t wid) const {
  if (wid < 0) { // out of vocab
    if (token != EOS) {
      computeSubwords(BOW + token + EOW, line);
    }
  } else {
    if (args_->maxn <= 0) { // in vocab w/o subwords
      line.push_back(wid);
    } else { // in vocab w/ subwords
      const std::vector<int32_t>& ngrams = getSubwords(wid);
      line.insert(line.end(), ngrams.cbegin(), ngrams.cend());
    }
  }
}

void Dictionary::reset(std::istream& in) const {
  if (in.eof()) {
    in.clear();
    in.seekg(std::streampos(0));
  }
}

int32_t Dictionary::getLine(
    std::istream& in,
    std::vector<int32_t>& words,
    std::minstd_rand& rng) const {
  std::uniform_real_distribution<> uniform(0, 1);
  std::string token;
  int32_t ntokens = 0;

  reset(in);
  words.clear();
  while (readWord(in, token)) {
    int32_t h = find(token);
    int32_t wid = word2int_[h];
    if (wid < 0) {
      continue;
    }

    ntokens++;
    if (getType(wid) == entry_type::word && !discard(wid, uniform(rng))) {
      words.push_back(wid);
    }
    if (ntokens > MAX_LINE_SIZE || token == EOS) {
      break;
    }
  }
  return ntokens;
}

int32_t Dictionary::getLine(
    std::istream& in,
    std::vector<int32_t>& words,
    std::vector<int32_t>& labels) const {
  std::vector<int32_t> word_hashes;
  std::string token;
  int32_t ntokens = 0;

  reset(in);
  words.clear();
  labels.clear();
  while (readWord(in, token)) {
    uint32_t h = hash(token);
    int32_t wid = getId(token, h);
    entry_type type = wid < 0 ? getType(token) : getType(wid);

    ntokens++;
    if (type == entry_type::word) {
      addSubwords(words, token, wid);
      word_hashes.push_back(h);
    } else if (type == entry_type::label && wid >= 0) {
      labels.push_back(wid - nwords_);
    }
    if (token == EOS) {
      break;
    }
  }
  addWordNgrams(words, word_hashes, args_->wordNgrams);
  return ntokens;
}

void Dictionary::pushHash(std::vector<int32_t>& hashes, int32_t id) const {
  if (pruneidx_size_ == 0 || id < 0) {
    return;
  }
  if (pruneidx_size_ > 0) {
    if (pruneidx_.count(id)) {
      id = pruneidx_.at(id);
    } else {
      return;
    }
  }
  hashes.push_back(nwords_ + id);
}

std::string Dictionary::getLabel(int32_t lid) const {
  if (lid < 0 || lid >= nlabels_) {
    throw std::invalid_argument(
        "Label id is out of range [0, " + std::to_string(nlabels_) + "]");
  }
  return words_[lid + nwords_].word;
}

void Dictionary::save(std::ostream& out) const {
  out.write((char*)&size_, sizeof(int32_t));
  out.write((char*)&nwords_, sizeof(int32_t));
  out.write((char*)&nlabels_, sizeof(int32_t));
  out.write((char*)&ntokens_, sizeof(int64_t));
  out.write((char*)&pruneidx_size_, sizeof(int64_t));
  for (int32_t i = 0; i < size_; i++) {
    entry e = words_[i];
    out.write(e.word.data(), e.word.size() * sizeof(char));
    out.put(0);
    out.write((char*)&(e.count), sizeof(int64_t));
    out.write((char*)&(e.type), sizeof(entry_type));
  }
  for (const auto pair : pruneidx_) {
    out.write((char*)&(pair.first), sizeof(int32_t));
    out.write((char*)&(pair.second), sizeof(int32_t));
  }
}

void Dictionary::load(std::istream& in) {
  words_.clear();
  in.read((char*)&size_, sizeof(int32_t));
  in.read((char*)&nwords_, sizeof(int32_t));
  in.read((char*)&nlabels_, sizeof(int32_t));
  in.read((char*)&ntokens_, sizeof(int64_t));
  in.read((char*)&pruneidx_size_, sizeof(int64_t));
  for (int32_t i = 0; i < size_; i++) {
    char c;
    entry e;
    while ((c = in.get()) != 0) {
      e.word.push_back(c);
    }
    in.read((char*)&e.count, sizeof(int64_t));
    in.read((char*)&e.type, sizeof(entry_type));
    words_.push_back(e);
  }
  pruneidx_.clear();
  for (int32_t i = 0; i < pruneidx_size_; i++) {
    int32_t first;
    int32_t second;
    in.read((char*)&first, sizeof(int32_t));
    in.read((char*)&second, sizeof(int32_t));
    pruneidx_[first] = second;
  }
  initTableDiscard();
  initNgrams();

  int32_t word2intsize = std::ceil(size_ / 0.7);
  word2int_.assign(word2intsize, -1);
  for (int32_t i = 0; i < size_; i++) {
    word2int_[find(words_[i].word)] = i;
  }
}

void Dictionary::init() {
  initTableDiscard();
  initNgrams();
}

void Dictionary::prune(std::vector<int32_t>& idx) {
  std::vector<int32_t> words, ngrams;
  for (auto it = idx.cbegin(); it != idx.cend(); ++it) {
    if (*it < nwords_) {
      words.push_back(*it);
    } else {
      ngrams.push_back(*it);
    }
  }
  std::sort(words.begin(), words.end());
  idx = words;

  if (ngrams.size() != 0) {
    int32_t j = 0;
    for (const auto ngram : ngrams) {
      pruneidx_[ngram - nwords_] = j;
      j++;
    }
    idx.insert(idx.end(), ngrams.begin(), ngrams.end());
  }
  pruneidx_size_ = pruneidx_.size();

  std::fill(word2int_.begin(), word2int_.end(), -1);

  int32_t j = 0;
  for (int32_t i = 0; i < words_.size(); i++) {
    if (getType(i) == entry_type::label ||
        (j < words.size() && words[j] == i)) {
      words_[j] = words_[i];
      word2int_[find(words_[j].word)] = j;
      j++;
    }
  }
  nwords_ = words.size();
  size_ = nwords_ + nlabels_;
  words_.erase(words_.begin() + size_, words_.end());
  initNgrams();
}

void Dictionary::dump(std::ostream& out) const {
  out << words_.size() << std::endl;
  for (auto it : words_) {
    std::string entryType = "word";
    if (it.type == entry_type::label) {
      entryType = "label";
    }
    out << it.word << " " << it.count << " " << entryType << std::endl;
  }
}

} // namespace fasttext
