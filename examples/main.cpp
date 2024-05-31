#include "add_node.h"
#include "backends/cpu/op_factory_cpu.h"
#include "const_node.h"
#include "graph.h"
#include "logger.h"

int main()
{
    opr::op_factory_cpu factory;
    auto add_op     = factory.create_add_op();
    opr::status ret = add_op->exec();
    log_inf("ret value: {}", ret);

    opr::graph g;
    opr::node_ptr add_node    = std::make_shared<opr::add_node>(0);
    opr::node_ptr const_node0 = std::make_shared<opr::const_node>(1, 3);
    opr::node_ptr const_node1 = std::make_shared<opr::const_node>(2, 6);
    g.add_node(add_node);
    g.add_node(const_node0);
    g.add_node(const_node1);
    g.add_dep(1, 0);
    g.add_dep(2, 0);

    ret = g.exec();

    return 0;
}
