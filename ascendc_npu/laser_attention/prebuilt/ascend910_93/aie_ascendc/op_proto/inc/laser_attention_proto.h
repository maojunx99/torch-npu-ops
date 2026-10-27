#ifndef LASER_ATTENTION_PROTO_H_
#define LASER_ATTENTION_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(LaserAttention)
    .INPUT(query, ge::TensorType::ALL())
    .INPUT(key, ge::TensorType::ALL())
    .INPUT(value, ge::TensorType::ALL())
    .OPTIONAL_INPUT(atten_mask, ge::TensorType::ALL())
    .OPTIONAL_INPUT(alibi_mask, ge::TensorType::ALL())
    .OPTIONAL_INPUT(drop_mask, ge::TensorType::ALL())
    .OUTPUT(softmax_log_max_sum, ge::TensorType::ALL())
    .OUTPUT(attention_out, ge::TensorType::ALL())
    .REQUIRED_ATTR(scale_value, Float)
    .REQUIRED_ATTR(head_num, Int)
    .REQUIRED_ATTR(input_layout, String)
    .ATTR(keep_prob, Float, 1)
    .ATTR(pre_tokens, Int, 2147483647)
    .ATTR(next_tokens, Int, 1)
    .ATTR(is_highPrecision, Bool, true)
    .OP_END_FACTORY_REG(LaserAttention);

}

#endif
