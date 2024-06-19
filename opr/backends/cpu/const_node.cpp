#include "const_node.h"
#include "logger.h"

namespace opr {

const_node::const_node(int id, int value) : node(id)
{
    cpu_buffer buffer({1}, sizeof(int32_t));

    auto data = buffer.get<int32_t>();
    data[0]   = value;
    output    = std::make_shared<tensor>(std::move(buffer));
}

const_node::~const_node() {}

// do absly nothing
status const_node::exec()
{
    auto data = std::get<cpu_buffer>(*output).get<int32_t>();
    log_inf("[const_node] output: {}", data[0]);
    return status::ok;
}

}  // namespace opr
