#include "backends/cuda/const_node.h"
#include "backends/cuda/vec_add_node.h"
#include "backends/cuda/vec_dot_node.h"
#include "backends/cuda/vec_sub_node.h"

#include "graph.h"
#include "logger.h"

int main() {
    opr::tensor_shape shape = {1, 1920, 1080};

    // create computation graph
    auto g0 = std::make_shared<opr::graph>(99, shape);

    // create nodes to use in the graph, each node represent a computation step
    opr::node_ptr vec_add_node = std::make_shared<opr::cuda::vec_add_node<float32_t>>(0, shape);
    opr::node_ptr const_node0  = std::make_shared<opr::cuda::const_node<float32_t>>(1, shape, 1);
    opr::node_ptr const_node1  = std::make_shared<opr::cuda::const_node<float32_t>>(2, shape, 2);
    opr::node_ptr vec_sub_node = std::make_shared<opr::cuda::vec_sub_node<float32_t>>(3, shape);
    opr::node_ptr vec_dot_node = std::make_shared<opr::cuda::vec_dot_node<float32_t>>(4, shape);

    // add nodes to graph
    g0->add_node(vec_add_node);
    g0->add_node(const_node0);
    g0->add_node(const_node1);
    g0->add_node(vec_sub_node);
    g0->add_node(vec_dot_node);

    // setup graph dependencies
    g0->add_depe(0, 1);  // add {id=1} as 1st dependencies of {id=0}
    g0->add_depe(0, 2);  // add {id=2} as 2nd dependencies of {id=0}

    g0->add_depe(3, 2);
    g0->add_depe(3, 1);

    g0->add_depe(4, 0);
    g0->add_depe(4, 3);

    check_err(g0->finalize() == opr::status::ok, "graph finalize step failed");

    g0->exec();

    // NOTE: read graph output back to cpu to print out
    auto& cuda_out = std::get<opr::cuda_buffer>(*g0->output);
    float* out     = new float[cuda_out.shape().elems()];
    cudaMemcpy(out, cuda_out.data, cuda_out.size_bytes(), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    get_last_cuda_errors();
    log_inf("cuda out value: {}", out[0]);
    free(out);

    return 0;
}
