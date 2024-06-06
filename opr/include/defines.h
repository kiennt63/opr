#pragma once

namespace opr {

enum status
{
    ok = 0,
    err
};

using char_t     = char;
using int8_t     = signed char;
using int16_t    = signed short;
using int32_t    = signed int;
using int64_t    = signed long;
using uint8_t    = unsigned char;
using uint16_t   = unsigned short;
using uint32_t   = unsigned int;
using uint64_t   = unsigned long;
using float32_t  = float;
using float64_t  = double;
using float128_t = long double;

}  // namespace opr
