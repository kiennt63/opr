#include "add_node.h"
#include "logger.h"

namespace opr {

add_node::add_node(int id) : node(id)
{
    output = new tensor;
}

add_node::~add_node()
{
    delete output;
}

status add_node::exec()
{
    check_err(input.size() == 2, "[add_node]: inputs size must be 2");
    *output = *input[0] + *input[1];
    return status::ok;
}

}  // namespace opr
