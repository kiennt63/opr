#include "const_node.h"
#include "logger.h"

namespace opr {

const_node::const_node(int id, int value) : node(id)
{
    output  = new int;
    *output = value;
}

// do absly nothing
status const_node::exec()
{
    return status::ok;
}

}  // namespace opr
