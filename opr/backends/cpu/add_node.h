#pragma once

#include "node.h"

namespace opr {

class add_node : public node
{
public:
    add_node(int id, const tensor_shape& shape);
    virtual ~add_node();
    status exec() override;
};

}  // namespace opr
