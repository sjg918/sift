
//original code:
//https://github.com/Celebrandil/CudaSift/

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <torch/extension.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)

int ScaleInvariantFeatureTransformCudaLauncher(float* img1, float* img2, float* matched_pts, int w_, int h_, int crop_kernel_size);

at::Tensor sift(const at::Tensor img1, const at::Tensor img2, at::Tensor numpts, const int w, const int h, const int crop_kernel_size) {
    float* img1_ = img1.contiguous().data_ptr<float>(); // must x128
    float* img2_ = img2.contiguous().data_ptr<float>();

    auto matched_pts = at::zeros({ 4096 * 4 }, torch::dtype(torch::kFloat32));
    float* matched_pts_ = matched_pts.data_ptr<float>();

    int numpts_result = 0;
    numpts_result = ScaleInvariantFeatureTransformCudaLauncher(img1_, img2_, matched_pts_, w, h, crop_kernel_size);
    numpts[0] = numpts_result;

    return matched_pts;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sift", &sift,
        "scale invariant feature transform on gpu");
}
