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

status graph::add_depe(int from, int to)
{
    nodes_[from]->dependencies.push_back(nodes_[to]);
    nodes_[from]->input.push_back(nodes_[to]->output);
    return status::ok;
}

void graph::topo_sort_helper(node_ptr node)
{
    node->visited = true;
    for (auto&& dep : node->dependencies)
    {
        if (!dep->visited)
        {
            topo_sort_helper(dep);
        }
    }

    order_.push_back(node->id);
}

status graph::topo_sort()
{
    for (auto& pair : nodes_)
    {
        if (!pair.second->visited)
        {
            topo_sort_helper(pair.second);
        }
    }

    return status::ok;
}

status graph::exec()
{
    for (int id : order_)
    {
        node_ptr node = nodes_[id];
        node->exec();
    }

    return status::ok;
}

status graph::finalize()
{
    check_err(topo_sort() == status::ok, "topo sort return fail");

    return status::ok;
}

}  // namespace opr
