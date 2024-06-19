#pragma once

#include "node.h"

namespace opr {

class const_node : public node
{
public:
    const_node(int id, int value);
    virtual ~const_node();
    status exec() override;
};

}  // namespace opr
