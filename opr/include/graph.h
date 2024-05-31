#pragma once

#include <list>
#include <stack>
#include <unordered_set>

#include "defines.h"
#include "logger.h"
#include "node.h"

namespace opr {

class graph
{
public:
    // add a node to graph
    status add_node(node_ptr node);

    // add dependency edge from node v to w
    status add_dep(int from, int to);

    status exec();

private:
    // perform topological sort
    status topo_sort();

    // helper func to topo sort
    void topo_sort_helper(node_ptr node, std::vector<int>& order);

    // map of node id to node pointer
    std::unordered_map<int, node_ptr> nodes_;
};

}  // namespace opr
