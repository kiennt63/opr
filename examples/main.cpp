#include "logger.h"
#include "op.h"

int main()
{
    opr::op test;
    auto ret = test.exec();
    log_inf("ret value: {}", ret);
    return 0;
}
