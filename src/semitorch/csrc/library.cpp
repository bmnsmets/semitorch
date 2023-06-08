#include <torch/extension.h>
#include <vector>
#include <cstdint>
#include "macros.h"

#ifdef WITH_CUDA
#include <cuda.h>
#endif

namespace semitorch {

    ST_API int64_t cuda_version() {
        #ifdef WITH_CUDA
        return 1; //CUDA_VERSION;
        #else
        return -1;
        #endif
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("cuda_version", &cuda_version, "CUDA version");
    } 

}