#pragma once

#include <memory>

#include "add_cpu.h"
#include "op_factory.h"

namespace opr {

class op_factory_cpu : public opr::op_factory_crtp<op_factory_cpu>
{
public:
    std::unique_ptr<op<cpu::add_op>> create_add_op() { return std::make_unique<op<cpu::add_op>>(); }
};

}  // namespace opr
