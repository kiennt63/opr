#pragma once

#include "node.h"

namespace opr {

class subtract_node : public node
{
public:
    subtract_node(int id, const tensor_shape& shape);
    virtual ~subtract_node();
    status exec() override;
};

}  // namespace opr
