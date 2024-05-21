#pragma once

#include "defines.h"

namespace opr {

template <typename t>
class op_factory_crtp
{
public:
    std::unique_ptr<op<cpu::add_op>> create_add_op()
    {
        return static_cast<t*>(this)->create_add_op();
    } status create_mul_op() { return static_cast<t*>(this)->create_mul_op(); }
};

}  // namespace opr
