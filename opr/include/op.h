/* Insert copyright */
#ifndef OPR_OP_H_
#define OPR_OP_H_

#include "buffer.h"
#include "defines.h"

namespace opr {

class op
{
public:
    op();
    ~op();
    buffer input;
    buffer output;
    status exec();

private:
    op* next_ = nullptr;
    op* prev_ = nullptr;
};

}  // namespace opr

#endif  // OPR_OP_H_
