
#from setuptools import setup
#from torch.utils.cpp_extension import BuildExtension, CUDAExtension

#setup(
#    name='semi_global_matching',
#    ext_modules=[
#        CUDAExtension('semi_global_matching_cuda', [
#            'sgm_src/semi_global_matching.cpp',
#            'sgm_src/semi_global_matching_cuda.cu',
#        ])
#    ],
#    cmdclass={'build_ext': BuildExtension})


from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='scale_invariant_feature_transform',
    ext_modules=[
        CUDAExtension('sift_on_gpu', [
            'sift_src/scale_invariant_feature_transform_cuda.cu',
            'sift_src/scale_invariant_feature_transform.cpp'
        ])
    ],
    cmdclass={'build_ext': BuildExtension})