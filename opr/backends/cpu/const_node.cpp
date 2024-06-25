#include "const_node.h"
#include "logger.h"

namespace opr {

const_node::const_node(int id, const tensor_shape& shape, int value) : node(id, shape)
{
    cpu_buffer buffer(shape, sizeof(int32_t));

    auto data = buffer.get<int32_t>();
    for (size_t i = 0; i < shape.elems(); i++)
    {
        data[i] = value;
    }
    output = std::make_shared<tensor>(std::move(buffer));
}

const_node::~const_node() {}

// do absly nothing
status const_node::exec()
{
    log_inf("[const_node] output: {}", std::get<cpu_buffer>(*output));
    return status::ok;
}

}  // namespace opr
