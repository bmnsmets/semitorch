#include <cuda.h>

#include "maxplus_cuda.cuh"

namespace semitorch
{
    std::vector<torch::Tensor> maxplus_forward_cuda(const torch::Tensor x, const torch::Tensor w)
    {
        return {};
    }

    std::vector<torch::Tensor> maxplus_backward_cuda(const torch::Tensor grad_y, const torch::Tensor hits)
    {
        return {};
    }
} // namespace semitorch