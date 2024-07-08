#pragma once

#include "node.h"

namespace opr {

class const_node : public node
{
public:
    const_node(int id, const tensor_shape& shape, int value);
    const_node(int id, const tensor_shape& shape, void* value);
    virtual ~const_node();
    status exec() override;
};

}  // namespace opr
