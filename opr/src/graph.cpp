#include <fstream>

#include "graph.h"
#include "logger.h"

namespace opr {

graph::graph(int id, const tensor_shape& shape) : node(id, shape) {}

status graph::add_node(node_ptr node) {
    if (nodes_.find(node->id) == nodes_.end()) {
        nodes_[node->id] = node;
    }
    return status::ok;
}

status graph::add_depe(int from, int to) {
    nodes_[from]->dependencies.push_back(nodes_[to]);
    nodes_[from]->input.push_back(nodes_[to]->output);
    return status::ok;
}

void graph::topo_sort_helper(node_ptr node) {
    node->visited = true;
    for (auto&& dep : node->dependencies) {
        if (!dep->visited) {
            topo_sort_helper(dep);
        }
    }

    order_.push_back(node->id);
}

status graph::topo_sort() {
    for (auto& pair : nodes_) {
        if (!pair.second->visited) {
            topo_sort_helper(pair.second);
        }
    }

    return status::ok;
}

status graph::exec() {
    for (int id : order_) {
        node_ptr node = nodes_[id];
        node->exec();
    }

    return status::ok;
}

status graph::gen_dot(const std::string& filename) {
    std::ofstream file(filename.c_str());

    file << "digraph G {\n";

    for (const auto& pair : nodes_) {
        auto& node = pair.second;
        if (node->dependencies.empty()) {  // Ensure all nodes appear in the graph
            file << "    " << pair.first << ";\n";
        }
        for (auto& dep : node->dependencies) {
            file << "    " << pair.first << " -> " << dep->id << ";\n";
        }
    }

    file << "}\n";
    file.close();

    return status::ok;
}

status graph::finalize() {
    check_err(topo_sort() == status::ok, "topo sort return fail");
    check_err(!order_.empty(), "the graph is empty, what u're trying to do?");

    // NOTE: set output of the graph to be the output of the final node
    output = nodes_[order_[order_.size() - 1]]->output;

    return status::ok;
}

}  // namespace opr
