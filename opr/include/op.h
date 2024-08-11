#pragma once

#include "defines.h"

#include <iostream>

namespace opr {

template <typename t>
class op {
public:
    status exec() { return static_cast<t*>(this)->exec(); }
};

}  // namespace opr
