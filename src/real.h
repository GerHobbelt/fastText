/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#if defined(FASTTEXT_Q16) || defined(FASTTEXT_Q32) || defined(FASTTEXT_F16) || defined(FASTTEXT_BF16)
#include "qnum.hpp"
#include "flex.hpp"
#endif

namespace fasttext {

#if defined(FASTTEXT_Q16)
typedef qnum::qspace_number_t<int16_t, 3, 0, true> real;
#elif defined(FASTTEXT_Q32)
typedef qnum::qspace_number_t<int32_t, 3, 0, true> real;
#elif defined(FASTTEXT_F16)
typedef flex::flexfloat<5,10> real;
#elif defined(FASTTEXT_BF16)
typedef flex::flexfloat<8,7> real;
#else
typedef float real;
#endif
}
