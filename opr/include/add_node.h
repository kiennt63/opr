#pragma once

#include "node.h"

namespace opr {

class add_node : public node
{
public:
    add_node(int id);
    status exec() override;
};

}  // namespace opr
