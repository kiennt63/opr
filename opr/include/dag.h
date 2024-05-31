#pragma once

#include <list>
#include <stack>
#include <unordered_set>

#include "defines.h"
#include "logger.h"

namespace opr {

class dag
{
public:
    // add a vertex to graph
    status add_vert(int v);
    // add directed edge from vertex v to w
    status add_edge(int v, int w);

    // perform topological sort
    status topo_sort();

private:
    void topo_sort_helper(int v, std::unordered_set<int>& visited, std::stack<int>& stack);

    std::unordered_map<int, std::list<int>> adjacents_;
};

}  // namespace opr
