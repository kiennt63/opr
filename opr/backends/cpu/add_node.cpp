#include "add_node.h"
#include "logger.h"

namespace opr {

add_node::add_node(int id) : node(id)
{
    cpu_buffer buffer({1}, sizeof(int32_t));
    output = std::make_shared<tensor>(std::move(buffer));
}

add_node::~add_node() {}

status add_node::exec()
{
    check_err(input.size() == 2, "[add_node]: inputs size must be 2");
    auto first  = std::get<cpu_buffer>(*input[0]).get<int32_t>();
    auto second = std::get<cpu_buffer>(*input[1]).get<int32_t>();
    auto out    = std::get<cpu_buffer>(*output).get<int32_t>();

    out[0] = first[0] + second[0];
    log_inf("[add_node]: output: {}", out[0]);
    return status::ok;
}

}  // namespace opr
