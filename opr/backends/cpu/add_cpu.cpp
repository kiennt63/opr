#include "add_cpu.h"

opr::status opr::cpu::add_op::exec()
{
    log_inf("cpu::add_op::exec()");
    return status::ok;
}
