#pragma once

#include "logger.h"
#include "op.h"

namespace opr::cpu {

class add_op : public op<add_op>
{
public:
    status exec();
};

}  // namespace opr::cpu
