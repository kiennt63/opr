#include "backends/cpu/op_factory_cpu.h"
#include "logger.h"
#include "op.h"

int main()
{
    opr::op_factory_cpu factory;
    auto add_op     = factory.create_add_op();
    opr::status ret = add_op->exec();
    log_inf("ret value: {}", ret);
    return 0;
}
