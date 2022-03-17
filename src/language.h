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
#include <unordered_set>
#include <vector>

namespace fasttext {

typedef int32_t id_type;
enum class entry_type : int8_t { word = 0, label = 1 };

struct entry {
  std::string word;
  int64_t count;
  entry_type type;
  std::vector<int32_t> subwords;
};

struct lang {
    uint8_t MAXLEN;
    std::unordered_set<std::string> PROFANITY;
    std::unordered_set<std::string> STOPWORDS;
};

class Language {

    std::unordered_map<std::string, uint8_t> LONGEST_WORDS = {{"en", 28}, {"it", 26}, {"de", 32}, {"fr", 25}, {"es", 23}, {"pt", 29}};
    // STRICT_PUNCT is forbidden at the beginning and end of the word
    const std::unordered_set<std::string> STRICT_PUNCT{"&", "-"};
    // PUNCT is forbidden anywhere inside the word
    const std::vector<char*> PUNCT = {"!", "\"", "#", "$", "%", "\\", "\'", "(", ")", "*", "+", ",", ".", "/", ":", ";", "<", "=", ">", "?", "@", "[", "]", "^", "_", "`", "{", "|", "}", "~"};

 protected:
    lang lang_;
    std::unordered_set<std::string> dict_;
    std::unordered_set<std::string> load(const std::string&);

 public:

    Language(const std::string& lang);
    std::vector<entry> words;

    uint8_t max() {
      return lang_.MAXLEN;
    };
    void init(const std::string& lang);
    void addWord(const entry);
    bool isWord(const std::string);        // check that there is at least a letter
    bool isDuplicate(std::string);         // check that the word isn't already present after stripping punctuation and lowercasing,
                                           // also check that individual tokens are present in the model after splitting on punctuation?
    bool isProfanity(const std::string);   // dictionary-based check that word is not profanity
    bool isStopword(const std::string);    // check that word is not a stopword
    bool isWeb(const std::string);         // check that word is not an URL and that it's not an HTML tag
    bool isUUID(const std::string);        // check that word is not a UUID (apparently very common)

};

} // namespace fasttext
