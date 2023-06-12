#include <torch/extension.h>
#include <vector>
#include <cstdint>
#include "macros.h"

#ifdef WITH_CUDA
#include <cuda.h>
#endif

#include "maxplus.h"

namespace semitorch {

    ST_API int64_t cuda_version() {
        #ifdef WITH_CUDA
        return CUDA_VERSION;
        #else
        return -1;
        #endif
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("cuda_version", &cuda_version, "CUDA version");
        m.def("maxplus_forward", &maxplus_forward);
        m.def("maxplus_backward", &maxplus_backward);
    } 
}