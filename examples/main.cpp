#include "backends/cpu/add_node.h"
#include "backends/cpu/const_node.h"
#include "backends/cpu/subtract_node.h"
#include "graph.h"
#include "logger.h"

int main()
{
    // create computation graph
    auto g = std::make_shared<opr::graph>();

    // create nodes to use in the graph, each node represent a computation step
    opr::tensor_shape shape   = {9, 2, 3};
    opr::node_ptr sub_node    = std::make_shared<opr::subtract_node>(3, shape);
    opr::node_ptr add_node0   = std::make_shared<opr::add_node>(0, shape);
    opr::node_ptr add_node1   = std::make_shared<opr::add_node>(5, shape);
    opr::node_ptr const_node0 = std::make_shared<opr::const_node>(1, shape, 3);
    opr::node_ptr const_node1 = std::make_shared<opr::const_node>(2, shape, 6);
    opr::node_ptr const_node2 = std::make_shared<opr::const_node>(4, shape, 11);

    // add nodes to graph
    g->add_node(sub_node);
    g->add_node(add_node0);
    g->add_node(add_node1);
    g->add_node(const_node0);
    g->add_node(const_node1);
    g->add_node(const_node2);

    // setup graph dependencies
    g->add_depe(0, 1);  // add {id=1} as 1st dependencies of {id=0}
    g->add_depe(0, 2);  // add {id=2} as 2nd dependencies of {id=0}
    g->add_depe(3, 4);  // add {id=4} as 1st dependencies of {id=3}
    g->add_depe(3, 2);  // add {id=2} as 2nd dependencies of {id=3}
    g->add_depe(5, 0);  // add {id=0} as 1st dependencies of {id=5}
    g->add_depe(5, 3);  // add {id=3} as 2nd dependencies of {id=5}

    check_err(g->finalize() == opr::status::ok, "graph finalize step failed");

    auto ret = g->exec();
    check_err(ret == opr::status::ok, "graph execution failed");

    return 0;
}
