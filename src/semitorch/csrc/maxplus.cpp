#include "maxplus.h"
#include "dbg.h"

#include <torch/extension.h>

namespace semitorch
{

    std::vector<torch::Tensor> maxplus_forward(const torch::Tensor x, const torch::Tensor w)
    {
        auto x_arg = torch::TensorArg(x, "x", 1);
        auto w_arg = torch::TensorArg(w, "w", 1);
        torch::CheckedFrom c{"maxplus_forward"};
        torch::checkAllDefined(c, {x_arg, w_arg});
        torch::checkAllSameType(c, {x_arg, w_arg});
        if (x.device().type() == torch::kCPU)
        {
        }
        else if (x.device().type() == torch::kCUDA)
        {
            torch::checkAllSameGPU(c, {x_arg, w_arg});
        }

        return {};
    }

    std::vector<torch::Tensor> maxplus_backward(const torch::Tensor grad_y, const torch::Tensor hits)
    {
        return {};
    }

}