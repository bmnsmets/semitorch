#include "maxplus.h"
#include "dbg.h"

#include <torch/extension.h>

#ifdef WITH_CUDA
#include "maxplus_cuda.cuh"
#endif

namespace semitorch
{

    std::vector<at::Tensor> maxplus_forward(const at::Tensor x, const at::Tensor w)
    {
        auto x_arg = at::TensorArg(x, "x", 1);
        auto w_arg = at::TensorArg(w, "w", 2);
        at::CheckedFrom c{"maxplus_forward"};
        at::checkAllDefined(c, {x_arg, w_arg});
        at::checkAllSameType(c, {x_arg, w_arg});

        if (x.device().type() == torch::kCPU)
        {
            return maxplus_forward_cpu(x.contiguous(), w.contiguous());
        }
#ifdef WITH_CUDA
        else if (x.device().type() == torch::kCUDA)
        {
            at::checkAllSameGPU(c, {x_arg, w_arg});
            return maxplus_forward_cuda(x.contiguous(), w.contiguous());
        }
#endif

        C10_THROW_ERROR(NotImplementedError, "maxplus_forward not implemented for device");
    }

    std::vector<torch::Tensor> maxplus_backward(const torch::Tensor grad_y, const torch::Tensor hits)
    {
        auto grad_y_arg = at::TensorArg(grad_y, "grad_y", 1);
        auto hits_arg = at::TensorArg(hits, "hits", 2);
        at::CheckedFrom c{"maxplus_backward"};
        at::checkAllDefined(c, {grad_y_arg, hits_arg});

        if (grad_y.device().type() == torch::kCPU)
        {
            return maxplus_backward_cpu(grad_y.contiguous(), hits.contiguous());
        }
#ifdef WITH_CUDA
        else if (grad_y.device().type() == torch::kCUDA)
        {
            at::checkAllSameGPU(c, {grad_y_arg, hits_arg});
            return maxplus_backward_cuda(grad_y.contiguous(), hits.contiguous());
        }
#endif

        C10_THROW_ERROR(NotImplementedError, "maxplus_backward not implemented for device");
    }

    std::vector<at::Tensor> maxplus_forward_cpu(const at::Tensor x, const at::Tensor w)
    {
        return {};
    }

    std::vector<at::Tensor> maxplus_backward_cpu(const at::Tensor grad_y, const at::Tensor hits)
    {
        return {};
    }
} // namespace semitorch