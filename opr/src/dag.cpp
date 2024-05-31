#include "dag.h"
#include <sstream>

namespace opr {

status dag::add_vert(int v)
{
    adjacents_[v];
    return status::ok;
}

status dag::add_edge(int v, int w)
{
    adjacents_[v].push_back(w);
    return status::ok;
}

void dag::topo_sort_helper(int v, std::unordered_set<int>& visited, std::stack<int>& stack)
{
    visited.insert(v);

    for (auto i = adjacents_[v].begin(); i != adjacents_[v].end(); ++i)
    {
        // i is not visited
        if (visited.find(*i) == visited.end())
        {
            topo_sort_helper(*i, visited, stack);
        }
    }

    stack.push(v);
}

status dag::topo_sort()
{
    std::stack<int> stack;
    std::unordered_set<int> visited;

    for (auto i = adjacents_.begin(); i != adjacents_.end(); ++i)
    {
        if (visited.find(i->first) == visited.end())
        {
            topo_sort_helper(i->first, visited, stack);
        }
    }

    std::stringstream ss;
    while (!stack.empty())
    {
        ss << stack.top() << " ";
        stack.pop();
    }
    log_inf("topo sort: {}", ss.str());

    return status::ok;
}

}  // namespace opr
