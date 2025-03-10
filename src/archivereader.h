/**
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "utils.h"

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filter/lzma.hpp>

namespace fasttext {
namespace impl {

class ArchiveReader {
public:
  ArchiveReader(const std::string& filename)
    : filename_(filename),
      src_(filename_, std::ios_base::in | std::ios_base::binary),
      dont_close_src_(false) {
    if (!src_.is_open()) {
      throw std::invalid_argument(filename_ + " cannot be opened");
    }

    reset();
  }

  ArchiveReader(std::istream& src) {
    impl_.push(src);
    dont_close_src_ = true;
  }

  virtual ~ArchiveReader() {
    while (!impl_.empty()) {
      impl_.pop();
    }
    if (!dont_close_src_ && src_.is_open()) {
      src_.close();
    }
  }

  void reset() {
    src_.clear();
    src_.seekg(std::streampos(0));
    while (!impl_.empty()) {
      impl_.pop();
    }
    if (utils::endsWith(filename_, ".xz")) {
      impl_.push(boost::iostreams::lzma_decompressor());
    }
    else if (utils::endsWith(filename_, ".gz")) {
      impl_.push(boost::iostreams::gzip_decompressor());
    }
    impl_.push(src_);
  }

  inline std::istream& stream() {
    return impl_;
  }

  inline bool eof() {
    return impl_.eof();
  }

private:
  std::string filename_;
  boost::iostreams::filtering_istream impl_;
  std::ifstream src_;
  bool dont_close_src_;
};

}
}
