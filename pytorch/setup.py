from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="band_attention",
    include_dirs=["include"],
    ext_modules=[
        CUDAExtension(
            "band_attention",
            ["pytorch/band_attention_ops.cpp", "kernel/band_attention_kernel.cu"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)