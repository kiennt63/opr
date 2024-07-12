#include "backends/cuda/add_node_cuda.h"
#include "backends/cuda/const_node_cuda.h"

#include "graph.h"
#include "logger.h"

int main()
{
    opr::tensor_shape shape = {1, 1920, 1080};

    // create computation graph
    auto g0 = std::make_shared<opr::graph>(99, shape);

    // create nodes to use in the graph, each node represent a computation step
    opr::node_ptr add_node0   = std::make_shared<opr::add_node_cuda>(0, shape);
    opr::node_ptr const_node0 = std::make_shared<opr::const_node_cuda>(1, shape, 3);
    opr::node_ptr const_node1 = std::make_shared<opr::const_node_cuda>(2, shape, 6);

    // add nodes to graph
    g0->add_node(add_node0);
    g0->add_node(const_node0);
    g0->add_node(const_node1);

    // setup graph dependencies
    g0->add_depe(0, 1);  // add {id=1} as 1st dependencies of {id=0}
    g0->add_depe(0, 2);  // add {id=2} as 2nd dependencies of {id=0}

    check_err(g0->finalize() == opr::status::ok, "graph finalize step failed");

    g0->exec();

    // NOTE: read graph output back to cpu to print out
    auto& cuda_out = std::get<opr::cuda_buffer>(*g0->output);
    float* out     = new float[shape.elems()];
    cudaMemcpy(out, cuda_out.data, cuda_out.size_bytes(), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    get_last_cuda_errors();
    log_inf("cuda out value: {} {} {}", out[0], out[1], out[2]);
    free(out);

    return 0;
}
