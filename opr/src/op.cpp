#include "op.h"
#include "logger.h"

opr::op::op() {}
opr::op::~op() {}

opr::status opr::op::exec()
{
    if (prev_)
    {
        (void)prev_;
    }

    if (next_)
    {
        log_inf("working");
        return next_->exec();
    }

    return opr::status::ok;
}
