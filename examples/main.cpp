#include "backends/cpu/add_node.h"
#include "backends/cpu/const_node.h"
#include "graph.h"
#include "logger.h"

int main()
{
    opr::graph g;
    opr::node_ptr add_node    = std::make_shared<opr::add_node>(0);
    opr::node_ptr const_node0 = std::make_shared<opr::const_node>(1, 3);
    opr::node_ptr const_node1 = std::make_shared<opr::const_node>(2, 6);
    g.add_node(add_node);
    g.add_node(const_node0);
    g.add_node(const_node1);
    g.add_dep(1, 0);
    g.add_dep(2, 0);

    auto ret = g.exec();
    check_err(ret == opr::status::ok, "graph execution failed");

    return 0;
}
