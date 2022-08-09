#include "language.h"
#include "assert.h"

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/uuid/string_generator.hpp>
// #include <boost/regex.hpp>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string.h>
#include <regex>

namespace fasttext {

    Language::Language(const std::string& language) : lang_() {
        init(language);
    };

    void Language::init(const std::string& language) {
        assert(LONGEST_WORDS.find(language) != LONGEST_WORDS.end());
        lang_.MAXLEN = LONGEST_WORDS[language];
        lang_.PROFANITY = load(std::getenv("PROFANITY_PATH") + language + ".txt");
        std::cerr << "Loaded " << lang_.PROFANITY.size() << " profanity words" << std::endl;
        lang_.STOPWORDS = load(std::getenv("STOPWORDS_PATH") + language + ".txt");
        std::cerr << "Loaded " << lang_.STOPWORDS.size() << " stopwords" << std::endl;
        std::cerr << "Loaded language " << language << " with MAXLEN " << unsigned(lang_.MAXLEN) << std::endl;
    }

    std::unordered_set<std::string> Language::load(const std::string& filename) {
        std::unordered_set<std::string> set;
        std::string line;
        std::ifstream ifs(filename);
        if(ifs.is_open()) {
            while(getline(ifs, line)) {
                set.insert(line);
            }
        } else {
            throw std::invalid_argument("Cannot load " + filename + "!");
        }
        ifs.close();
        return set;
    }

    void Language::addWord(const entry e) {
        dict_.insert(e.word);
        words.push_back(e);
    }

    bool Language::isWord(const std::string word) {
        for(int8_t i = 0; i < word.size(); i++) {
            if(std::isalpha(word[i])){return true;}
        }
        return false;
    }

    bool Language::isProfanity(const std::string word) {
        return lang_.PROFANITY.find(word) != lang_.PROFANITY.end();
    }

    bool Language::isStopword(const std::string word) {
        return lang_.STOPWORDS.find(word) != lang_.STOPWORDS.end();
    }

    bool Language::isDuplicate(std::string word) {
        std::string original = word;
        std::transform(word.begin(), word.end(), word.begin(), [](unsigned char c){return std::tolower(c);});
        std::string s = word;
        for (uint8_t i = 0; i < PUNCT.size(); i++) {
            word.erase(std::remove(word.begin(), word.end(), *PUNCT[i]), word.end());
        }
        if (word.size() > 0 && STRICT_PUNCT.find(std::string(1, word.front())) != STRICT_PUNCT.end()) {
            word.erase(0, 1);
        }
        if (word.size() > 0 && STRICT_PUNCT.find(std::string(1, word.back())) != STRICT_PUNCT.end()) {
            word.erase(word.size() - 1);
        }
        if (word.empty()) return true;
        if (isProfanity(word)) {
            return true;
        }
        if (isStopword(word)) return true;
        if (word == original) return false;
        if (dict_.find(word) != dict_.end()) return true;
        // Check that splitting the words on full stops doesn't return a set of already known tokens
        // Very expensive but worth the effort.
        std::string delimiter = ".";
        size_t pos = 0;
        std::string token;
        bool flag = true;
        int its = 0;
        while ((pos = s.find(delimiter)) != std::string::npos) {
            its++;
            token = s.substr(0, pos);
            if (!token.empty() && dict_.find(token) == dict_.end() && !isStopword(token) && !isProfanity(token) && isWord(token)) {
                flag = false;
                break;
            }
            s.erase(0, pos + delimiter.length());
        }
        if(flag and its > 0) {
            // std::cerr << original << " is a composition of already known words!" << std::endl;
            return true;
        }
        return false;
    }

    bool Language::isWeb(const std::string word) {
        try {
            // std::regex_match(word, std::regex("^(https?:\/\/)?([\da-z-]+\\.)+([a-z\\.]{2,6})([\/\w \\.-]*)*\/?$"));
            return std::regex_match(word, std::regex("^(https?:\/\/)?([\da-z-]+\\.)+([a-z\\.]{2,6})([\/\w \\.-]*)*\/?$"));
        } catch (const std::regex_error& e) {
            std::cerr << "REGEX ERROR: " << e.what() << std::endl;
            std::cerr << "REGEX ERROR CODE: " << e.code() << std::endl;
            exit(0);
        }
        std::cerr << "PASSED" << std::endl;
        exit(0);
    }

    bool Language::isUUID(const std::string word) {
        using namespace boost::uuids;
        try {
            auto result = string_generator()(word); 
            return result.version() != uuid::version_unknown;
        } catch(...) {
            return false;
        }
    }

}