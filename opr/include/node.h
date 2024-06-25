#pragma once

#include <memory>
#include <vector>

#include "defines.h"
#include "tensor.h"

namespace opr {

class node;

using node_ptr = std::shared_ptr<node>;

class node
{
public:
    node(int id, const tensor_shape& shape) : id(id), visited(false), shape_(shape) {}
    virtual ~node() {}
    int id;
    std::vector<std::shared_ptr<tensor>> input;
    std::shared_ptr<tensor> output;
    bool visited;
    std::vector<node_ptr> dependencies;
    status virtual exec() = 0;

private:
    tensor_shape shape_;
};

}  // namespace opr
