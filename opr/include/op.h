/* Insert copyright */
#ifndef OPR_OP_H_
#define OPR_OP_H_

#include "buffer.h"
#include "defines.h"

namespace opr {

class op
{
public:
    buffer input;
    buffer output;
    status exec();

private:
    op* next_;
    op* prev_;
};

}  // namespace opr

#endif  // OPR_OP_H_
