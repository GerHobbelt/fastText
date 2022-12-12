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

Dictionary::Dictionary(std::shared_ptr<Args> args, std::istream& in, std::shared_ptr<Language> lang)
    : args_(args),
      size_(0),
      nwords_(0),
      nlabels_(0),
      ntokens_(0),
      pruneidx_size_(-1) {
  load(in, lang);
}

int32_t Dictionary::find(const std::string& w) const {
  return find(w, hash(w));
}

/**
 * @brief
 * Given a token(word) and a int, value, which could be some meaningful value 
 * or word's certain hash-value(TODO: How to calculate hash value of this word?), 
 * calculate the 'no-collision id' of this word based on its hash value with a 
 * given "bucket size"(which is `word2intsize`), which is also 
 * least-word-vocab-size (since real word vocab size could be larger than it since 
 * the id-shift strategy to avoid id-collision), defined by `word2int_`.
 * `word2int_` is a predefined, fix-sized vector, which size is word-vocab size, 
 * each element's index in `word2int_` corresponds to certain word's id, and we can 
 * get the detail info of that word by using this index to query the `entry` in 
 * `Dictionary::words_`. 
 * The value of the element represents if there already certain word located in 
 * certain bucket, if not, the element of that bucket index should be -1. 
 * 
 * Briefly speaking, we can understand this as a hash-bucket, push each word to 
 * different bucket based on their hash value, the bucket id is the word id. 
 * BUT, this is a SPECIAL HASH_BUCKET STRATEGY, different word COULD NOT BE put in 
 * same bucket. 
 *
 * @param w Input word, could be a totally new word or an appeared word.
 * @param h Just represent an parameter to generate raw word id, is this 
 *   new generated id has collision with the id of an apeared different word, 
 *   than continuously using a shift-strategy to generate a new id until 
 *   getting a no-collision id or getting the conclusion that this word had
 *   appeared and beem allocated a id.
 */
int32_t Dictionary::find(const std::string& w, uint32_t h) const {
  int32_t word2intsize = word2int_.size();
  int32_t id = h % word2intsize;

  /// If it's an exists bucket id, which means:
  ///   1. Current calculated bucket id has been occupied. 
  ///   2. TODO: The word occupied current bucket is not same with current word.
  /// 
  /// Then shift current id by plus 1, and re-judgement the above 2 conditions, 
  /// until finding an empty bucket~
  ///
  /// QA: 
  /// * Why `words_[word2int_[id]]` represent current exists word's 
  ///   corresponding entry?
  /// * Since the strategy is based on "plus 1 shift" mode, maybe?
  ///
  /// TIPS:
  /// 1. `word2int_[id] != -1` means current calculated id has been occupied by 
  ///    some word, this word could be same or not same with current one.
  /// 2. `words_[word2int_[id]].word != w` means the word which occupied current 
  ///    calcualted id is not same with the current word, which means current word 
  ///    still has possibility be a new word, which means we should continuously 
  ///    try to assign one no-collision id for current word until we get one or 
  ///    make sure it is an appeared word.
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
  /// Gets word id
  int32_t h = find(w);
  /// Update the processed tokens count, include counting duplicate tokens.
  ntokens_++;
  /// If it's first time this token(word or label) appearance, then doing sth.
  /// NOTE:
  /// Actually the condition `word2int_[h] == -1` mush happen, since in 
  /// `Dictionary::find`, we know if the token is an new token(word or label), 
  /// we will finally find an `id` satisfies `word2int_[id] == -1`.
  if (word2int_[h] == -1) {
    /// Initialize a new `entry` instance saving this new-token(word or label)'s 
    /// info, includes sth such as this token's text, token's count, 
    /// token(word or label) type, token's char n-gram info (this is only for 
    /// word token, not for label token), etc.
    entry e;
    /// Record this new word's text
    e.word = w;
    /// Initialize this new word's count, since it's the first time this 
    /// word's appearance, so the count should initialized as 1.
    e.count = 1;
    /// NOTE:
    /// In fastText, there are two kinds of tokens, word-token and label-token, 
    /// they could be distincted by the fact that there is  "__label__" sign in 
    /// label token. So the naming of "words" sometimes mean "tokens", which 
    /// includes both words or labels but not only refer to words.
    /// The following line assign current token's type.
    e.type = getType(w);
    /// NOTE:
    /// Following line pushs back this new-token's `entry` object into `words_`, 
    /// the `words_` could be understoold as a token vocab which includs both 
    /// word-tokens and label-tokens info.
    /// After this pushing-back operation, we need to record this just-pushed-back 
    /// token's index in `words_`, which is the current size of the `word_`, equals 
    /// to `size_ + 1`, we will record word's token-vocab index to token-id's 
    /// corresponding value in `word2int_`, which is `word2int_[h]`, `h` is token-id.
    ///
    /// Each token's info will be build and put into `Dictionary::words_` during 
    /// the `Dictionary` building with `Dictionary::add`. This includes several steps:
    /// 1. Judging if current token is a new one, if it's a new token, executing 
    ///    following steps.
    /// 2. Getting the token's id from token's raw text by `Dictionary::find`.
    /// 3. Building current new token's `entry` object and push it back into 
    ///    token-vocab `Dictionary::words_`.
    /// 4. Using `Dictionary::word2int_` to record this new token's index in token 
    ///    vocab `Dictionary::words_`. The element value in `Dictionary::word2int_` 
    ///    which index equal to current token-id should be current token's 
    ///    correponding index in `Dictionary::words_`
    ///
    /// After buiding `Dictionary`, when we needs getting a raw token's detail info 
    /// (for example int training/inference stage), we just needs:
    /// 1. Mapping token's raw text to token-id.
    /// 2. Getting token-vocab-index by indexing the element with token-id from 
    ///    `Dictionary::word2int_`.
    /// 3. Getting token `entry` object by indexing token-vocab-index from 
    ///    ` Dictionary::words_`.
    words_.push_back(e);
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

/**
 * @brief 
 * Input sample's word-token and label-token's discarding rule. 
 * If not in surpervised training mod (which means training word-vector with 
 * language model), word-tokens will be randomly dropped if an generated 
 * random number is larget than `rand`.
 * Note this dropping will execute on word-token unit, not on char n-gram! 
 *
 * @param id One input token-vocab-index, the notion of `token-vocab-index` 
 *   ref to annotations in `Dictionary::words_`.
 * @param rand This is threshold control a random word dropping strategy 
 *   which can increase model's robustness by introduce randomness during 
 *   self-surpervised training mode. 
 */
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

/**
 * @brief
 * Judge if the input token is really a word or a label.
 *
 * @param w The input token, could be a word of a label with "__label__" sign. 
 */
entry_type Dictionary::getType(const std::string& w) const {
  return (w.find(args_->label) == 0) ? entry_type::label : entry_type::word;
}

std::string Dictionary::getWord(int32_t id) const {
  assert(id >= 0);
  assert(id < size_);
  return words_[id].word;
}

int64_t Dictionary::getTokenCount(int32_t id) const {
  assert(id >= 0);
  assert(id < size_);
  return words_[id].count;
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
    // TODO: Figure out why using this char filtering rule.
    if ((word[i] & 0xC0) == 0x80) {
      continue;
    }
    // Note, here we have a "max char n-gram" notion, i.e. `args_->maxn`, which means 
    // for each word, we will calculate char 1-gram, ... , n-gram(n == `args_->maxn`) 
    // unless word length smaller than `args_->maxn`.
    for (size_t j = i, n = 1; j < word.size() && n <= args_->maxn; n++) {
      ngram.push_back(word[j++]);
      // First extract char 1-gram, which means putting each char in `ngram`.
      // 
      // Note, we do not use char n-gram text itself to represent char n-gram feature, 
      // but using some int id to represent char n-gram.
      // For char 1-gram case (which is char itself), we will using `char` to `int32_t` 
      // implicit conversion result as each 1-gram id and push back into `ngram`.
      // For char n-gram (n larger than 2), we will first calculate a hash value as 
      // n-gram id for each n-gram and push back into `ngram` with `Dictionary::pushHash` 
      // according certain char n-gram pruning rule.
      while (j < word.size() && (word[j] & 0xC0) == 0x80) {
        ngram.push_back(word[j++]);
      }
      if (n >= args_->minn && !(n == 1 && (i == 0 || j == word.size()))) {
        // Execute hash-bucketing on `Dictionary::hash` result of each n-gram text. 
        // Using the mod of `Dictionary::hash` result on `args_->bucket` as 
        // current n-gram's hash-bucket id. 
        // Cheatsheet:
        //   1. args_->bucket: Hash bucket number for char n-gram. 
        int32_t h = hash(ngram) % args_->bucket;
        pushHash(ngrams, h);
        // TODO: Firgure out `substring` meaning
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

/**
 * @brief 
 * Reading a single token from input stream, the input stream using " " to split
 * each single token.
 * NOTE:
 * Although the method's name is `readWord`, but actually not every token is word, 
 * it could also be a label.
 */
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

void Dictionary::update(std::shared_ptr<Dictionary> dict, bool discardOovWords) {
  int32_t updated = 0, added = 0;

  /* Do not increment ntokens_ as it is used for the epoch calculation.
   * Do not alter the discard table by updating word counts.
   * Rare words are discarded less frequently and never if count=1.
   */
  for (int32_t i = 0; i < dict->nwords_; i++) {
    const std::string& w = dict->words_[i].word;
    int32_t h = find(w);
    if (word2int_[h] == -1) {
      if (!discardOovWords) {
        entry e;
        e.word = w;
        e.count = 1;
        e.type = getType(w);
        words_.push_back(e);
        word2int_[h] = size_++;
        added++;
      }
    } else {
      updated++;
    }
  }
  
  if (args_->model == model_name::sent2vec) {
    assert(words_[0].word == "<PLACEHOLDER>");
    words_[0].count = 1e+18;
  }
  
  threshold(1, 0);
  initTableDiscard();
  initNgrams();

  if (args_->model == model_name::sent2vec) {
    assert(words_[0].word == "<PLACEHOLDER>");
    words_[0].count = 0;
  }
  
  std::cerr << "Pretrained words: " << updated << " updated, " << added << " added" << std::endl;
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

  if (args_->model == model_name::sent2vec) {
    int32_t h = find("<PLACEHOLDER>");
    entry e;
    e.word = "<PLACEHOLDER>";
    e.count = 1e+18;
    e.type = entry_type::word;
    words_.push_back(e);
    word2int_[h] = size_++;
  }

  // The threshold used in `threshold(minThreshold, minThreshold);` is 
  // just used for dynamical control word vocab scale during continiously 
  // add word into dictionary which can control memory and some other computional 
  // resource using , the following finally `threshold` calling set the final 
  // scale of the word-vocab and label vocab 
  threshold(args_->minCount, args_->minCountLabel);
  initTableDiscard();
  initNgrams();

  if (args_->model == model_name::sent2vec) {
    assert(words_[0].word == "<PLACEHOLDER>");
    words_[0].count = 0;
  }

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

void Dictionary::addWordNgrams(
    std::vector<int32_t>& line,
    const std::vector<int32_t>& hashes,
    int32_t n, 
    int32_t k, 
    std::minstd_rand& rng) const {
  int32_t num_discarded = 0;
  std::vector<bool> discard;
  const int32_t size = hashes.size(); 
  discard.resize(size, false);
  std::uniform_int_distribution<> uniform(1, size);
  while (num_discarded < k && size - num_discarded > 2) {
    int32_t token_to_discard = uniform(rng);
    if (!discard[token_to_discard]) {
      discard[token_to_discard] = true;
      num_discarded++;
    }
  }
  for (int32_t i = 0; i < size; i++) {
    if (discard[i]) continue;
    uint64_t h = hashes[i];
    for (int32_t j = i + 1; j < size && j < i + n; j++) {
      if (discard[j]) break;
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

/**
 * @brief 
 * Getting one sample from input stream, and convert the text line into 
 * token-vocab-indexs, with which we can get this sample's tokens (maybe 
 * word-token or label-token) from token-vocab `Dictionary::words_` by 
 * indexing the element in it with token-vocab-indexs. Than executing 
 * some token-dropping strategy and push the left token's token-vocab-index 
 * into `words` and return incremental updated processed tokens number.
 *
 * Here is the detail process: 
 * 1. `Dictionary::find` will map token's raw text to token-id.
 * 2. Getting token-vocab-index by indexing the element which index equal 
 *    with token-id from `Dictionary::word2int_`. 
 * 3. Getting token entry object by indexing token-vocab-index from 
 *    `Dictionary::words_`.
 *
 * If still not understand, could refer more details in `Dictionary::add` 
 * annotation.
 *
 * NOTE: 
 * Here the program do not handling labels, the labels will be handeled 
 * in reload version of `Dictionary::getLine`.
 *
 * @param in Input stream
 * @param words Part of outputs' holder, holding extract and kept tokens' 
 *   token-vocab-index.
 * @param rng Random dropping hitting threshold used by `Dictionary::discard`.
 */
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
    /// Above process 1, `h` is token-id
    int32_t h = find(token);
    /// Above process 2, `wid` is token-vocab-index, which is token's 
    /// corrponding index in token-vocab `Dictionary::words_`.
    int32_t wid = word2int_[h];
    if (wid < 0) {
      continue;
    }

    /// Incremental counting the processed tokens (not matter duplicated or not) 
    /// during training process.
    ntokens++;

    /// Check if current token-vocab-index illegal and if current token hitting 
    /// discarding condition or random discarding strategy. If not, push current 
    /// token's token-vocab-index into result holder `words`. 
    if (getType(wid) == entry_type::word && !discard(wid, uniform(rng))) {
      words.push_back(wid);
    }
    /// Cutting of for some extremet long input token sequences.
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
      labels.push_back(wid);
    }
    if (token == EOS) {
      break;
    }
  }
  addWordNgrams(words, word_hashes, args_->wordNgrams);
  return ntokens;
}

int32_t Dictionary::getLine(
    std::istream& in,
    std::vector<int32_t>& words,
    std::vector<int32_t>& word_hashes,
    std::vector<int32_t>& labels,
    std::minstd_rand& rng,
    int32_t flags) const {
  std::uniform_real_distribution<> uniform(0, 1);
  std::string token;
  int32_t ntokens = 0;

  reset(in);
  words.clear();
  word_hashes.clear();
  labels.clear();
  while (readWord(in, token)) {
    if (flags & SKIP_EOS) {
      if (token == EOS) {
        break;
      }
    }
    uint32_t h = hash(token);
    int32_t wid = getId(token, h);
    if (flags & SKIP_OOV) {
      if (wid < 0) {
        continue;
      }
    }
    entry_type type = wid < 0 ? getType(token) : getType(wid);

    ntokens++;
    if (type == entry_type::word) {
      if (flags & SKIP_FRQ) {
        if (discard(wid, uniform(rng))) {
          continue;
        }
      }
      words.push_back(wid);
      word_hashes.push_back(h);
    } else if (type == entry_type::label && wid >= 0) {
      labels.push_back(wid);
    }
    if (flags & SKIP_LNG) {
      if (ntokens > MAX_LINE_SIZE) {
        break;
      }
    }
    if (token == EOS) {
       break;
    }
  }
  return ntokens;
}

int32_t Dictionary::getLineTokens(
		std::istream& in,
		std::vector<int32_t>& words,
		std::vector<int32_t>& labels,
		std::vector<std::string>& tokens) const {
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
			tokens.push_back(token);
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

/**
 * @brief
 * Decide if put char n-gram hash bucket id into a container or just 
 * prune this hash-bucket according some rule.
 * 
 * TODO: Figure why
 *
 * @param hashes Char n-gram hash id saving container
 * @param id Char n-gram hash id (hash bucket id), for n >= 2.
 */
void Dictionary::pushHash(std::vector<int32_t>& hashes, int32_t id) const {
  if (pruneidx_size_ == 0 || id < 0) {
    return;
  }
  if (pruneidx_size_ > 0) {
    /// `std::unordered_map` only contain unique key, so `pruneidx_.count(id)` can only 
    /// be 1 or 0. 
    /// QA: What's `pruneidx_` using for?
    /// My guess is `pruneidx_` saving the info about how to pruning char n-gram hash 
    /// bucket id, the "pruning" means merger serveral char n-gram id to an single one, 
    /// which saved in `pruneidx_.at(id)` 
    if (pruneidx_.count(id)) {
      /// TODO: What's `pruneidx_.at(id)` mean? 
      id = pruneidx_.at(id);
    } else {
      return;
    }
  }
  /// TODO: Figure out what's `nwords_` meaning.
  hashes.push_back(nwords_ + nlabels_ + id);
}

std::string Dictionary::getLabel(int32_t lid) const {
  if (lid < 0 || lid >= nlabels_) {
    throw std::invalid_argument(
        "Label id is out of range [0, " + std::to_string(nlabels_) + "]");
  }
  return words_[lid + nwords_].word;
}

bool Dictionary::checkValidWord(const std::string& word, std::shared_ptr<Language> lang) {
  // Checks for bigrams, trigrams and higher and increases the maximum length for a word in a language accordingly
  char* multigram_mark = std::getenv("MULTIGRAM_MARK");
  int multiplier = 1;
  int n = std::count(word.begin(), word.end(), multigram_mark[0]);
  multiplier += n;
  if(word.size() > (lang->max() * multiplier)) return false;
  if(!lang->isWord(word)) return false;
  if(lang->isDuplicate(word)) return false; // Also checks for profanity and stopwords
  if(lang->isWeb(word)) return false;
  if(lang->isUUID(word)) return false;

  return true;
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

void Dictionary::load(std::istream& in, std::shared_ptr<Language> lang) {
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
    lang->addWord(e);
  }
  std::cerr << "Read " << size_ << " words." << std::endl;
  std::cerr << "Size: " << size_ << std::endl;
  std::cerr << "NWords: " << nwords_ << std::endl;
  for (int32_t i = 0; i < lang->words.size(); i++) {
    if(checkValidWord(lang->words[i].word, lang)) {
      words_.push_back(lang->words[i]);
    } else {
      size_--;
      nwords_--;
      invalid_.push_back(i);
    }
    if (i % 100000 == 0) {
      std::cerr << "\rParsed " << i << " words" << std::flush;
    }
  }
  std::cerr << "\n\nKept " << size_ << " valid words.\n" << std::endl;
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
  std::cerr << "Dictionary loaded!" << std::endl;
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

void Dictionary::clip(int32_t max_size, std::shared_ptr<Language> lang) {
  if (lang->words.size() < max_size) {
    std::cerr << "Nothing to clip!" << std::endl;
    return;
  }
  std::vector<entry>::iterator last = words_.end();
  for(int32_t i = max_size; i < lang->words.size(); i++) {
    size_--;
    nwords_--;
    invalid_.push_back(i);
  }
  words_.erase(words_.begin() + max_size, words_.end());
  std::cerr << "\rRemoved " << lang->words.size() - max_size << " words from vocabulary." << std::endl;
}

void Dictionary::prune(std::vector<int32_t>& idx) {
  std::vector<int32_t> words, ngrams;
  for (auto it = idx.cbegin(); it != idx.cend(); ++it) {
    if (*it < nwords_ + nlabels_) {
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
      pruneidx_[ngram - nwords_ - nlabels_] = j;
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
