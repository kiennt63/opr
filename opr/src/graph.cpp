#include "graph.h"

namespace opr {

status graph::add_node(node_ptr node)
{
    if (nodes_.find(node->id) == nodes_.end())
    {
        nodes_[node->id] = node;
    }
    return status::ok;
}

status graph::add_dep(int from, int to)
{
    nodes_[from]->dependencies.push_back(nodes_[to]);
    nodes_[from]->input.push_back(nodes_[to]->output);
    return status::ok;
}

void graph::topo_sort_helper(node_ptr node, std::vector<int>& order)
{
    node->visited = true;
    for (auto&& dep : node->dependencies)
    {
        if (!dep->visited)
        {
            topo_sort_helper(dep, order);
        }
    }

    order.push_back(node->id);
}

status graph::topo_sort()
{
    std::vector<int> order;

    for (auto& pair : nodes_)
    {
        if (!pair.second->visited)
        {
            topo_sort_helper(pair.second, order);
        }
    }

    for (int id : order)
    {
        node_ptr node = nodes_[id];
        node->exec();
    }

    return status::ok;
}

status graph::exec()
{
    check_err(topo_sort() == status::ok, "topo sort return fail");

    return status::ok;
}

}  // namespace opr
