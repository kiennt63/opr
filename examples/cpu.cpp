#include "backends/cpu/const_node.h"
#include "backends/cpu/vec_add_node.h"
#include "backends/cpu/vec_sub_node.h"
#include "graph.h"
#include "logger.h"

int main() {
    opr::tensor_shape shape = {1, 1920, 1080};

    // create computation graph
    auto g0 = std::make_shared<opr::graph>(99, shape);

    std::vector<float32_t> test(shape.elems(), 11);

    // create nodes to use in the graph, each node represent a computation step
    opr::node_ptr sub_node    = std::make_shared<opr::cpu::vec_sub_node<float32_t>>(3, shape);
    opr::node_ptr add_node0   = std::make_shared<opr::cpu::vec_add_node<float32_t>>(0, shape);
    opr::node_ptr add_node1   = std::make_shared<opr::cpu::vec_add_node<float32_t>>(5, shape);
    opr::node_ptr const_node0 = std::make_shared<opr::cpu::const_node<float32_t>>(1, shape, 3);
    opr::node_ptr const_node1 = std::make_shared<opr::cpu::const_node<float32_t>>(2, shape, 6);
    opr::node_ptr const_node2 =
        std::make_shared<opr::cpu::const_node<float32_t>>(4, shape, test.data());

    // add nodes to graph
    g0->add_node(sub_node);
    g0->add_node(add_node0);
    g0->add_node(add_node1);
    g0->add_node(const_node0);
    g0->add_node(const_node1);
    g0->add_node(const_node2);

    // setup graph dependencies
    g0->add_depe(0, 1);  // add {id=1} as 1st dependencies of {id=0}
    g0->add_depe(0, 2);  // add {id=2} as 2nd dependencies of {id=0}
    g0->add_depe(3, 4);  // add {id=4} as 1st dependencies of {id=3}
    g0->add_depe(3, 2);  // add {id=2} as 2nd dependencies of {id=3}
    g0->add_depe(5, 0);  // add {id=0} as 1st dependencies of {id=5}
    g0->add_depe(5, 3);  // add {id=3} as 2nd dependencies of {id=5}

    check_err(g0->finalize() == opr::status::ok, "graph finalize step failed");

    auto g1 = std::make_shared<opr::graph>(100, shape);

    opr::node_ptr add_node2   = std::make_shared<opr::cpu::vec_add_node<float32_t>>(6, shape);
    opr::node_ptr const_node3 = std::make_shared<opr::cpu::const_node<float32_t>>(7, shape, 11);

    g1->add_node(add_node2);
    g1->add_node(const_node3);
    g1->add_node(g0);

    g1->add_depe(6, 7);
    g1->add_depe(6, 99);

    check_err(g1->finalize() == opr::status::ok, "graph finalize step failed");
    auto ret = g1->exec();
    check_err(ret == opr::status::ok, "graph execution failed");

    // g0->gen_dot("output/graph.dot");

    return 0;
}
