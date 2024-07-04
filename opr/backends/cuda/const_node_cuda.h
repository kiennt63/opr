#pragma once

#include "node.h"

namespace opr {

class const_node_cuda : public node
{
public:
    const_node_cuda(int id, const tensor_shape& shape, int value);
    virtual ~const_node_cuda();
    status exec() override;
};

}  // namespace opr
