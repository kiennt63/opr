#pragma once

#include <memory>
#include <vector>

#include "defines.h"

namespace opr {

class node;

using node_ptr = std::shared_ptr<node>;

class node
{
public:
    node(int id) : id(id), visited(false) {}
    virtual ~node() {}
    int id;
    std::vector<int*> inputs;
    int* output;
    bool visited;
    std::vector<node_ptr> dependencies;

    status virtual exec() = 0;
};

}  // namespace opr
