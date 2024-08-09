#pragma once

#include <unordered_map>

#include "defines.h"
#include "node.h"

namespace opr {

class graph : public node
{
public:
    graph() = delete;
    graph(int id, const tensor_shape& shape);
    // add a node to graph
    status add_node(node_ptr node);

    // add dependency edge from node v to w
    status add_depe(int from, int to);

    status gen_dot(const std::string& filename);

    status exec();

    status finalize();

private:
    // perform topological sort
    status topo_sort();

    // helper func to topo sort
    void topo_sort_helper(node_ptr node);

    // map of node id to node pointer
    std::unordered_map<int, node_ptr> nodes_;
    std::vector<int> order_;
};

}  // namespace opr
