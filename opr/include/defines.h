/* Insert copyright */
#ifndef OPR_DEFINES_H_
#define OPR_DEFINES_H_

#include "logger.h"

#define print_func() log_inf("{}()", __func__)

namespace opr {

enum status
{
    ok = 0,
    err
};

}  // namespace opr

#endif  // OPR_DEFINES_H_
