#pragma once

#include <torch/extension.h>
#include "macros.h"

namespace semitorch
{
    ST_API std::vector<torch::Tensor> maxplus_forward(const torch::Tensor x, const torch::Tensor w);
    ST_API std::vector<torch::Tensor> maxplus_backward(const torch::Tensor grad_y, const torch::Tensor hits);
}