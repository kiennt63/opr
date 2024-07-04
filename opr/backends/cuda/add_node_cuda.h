#pragma once

#include "node.h"

namespace opr {

class add_node_cuda : public node
{
public:
    add_node_cuda(int id, const tensor_shape& shape);
    virtual ~add_node_cuda();
    status exec() override;
};

}  // namespace opr
