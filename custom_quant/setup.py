from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name='custom_quant',
    ext_modules=[
        cpp_extension.CppExtension(
            'custom_quant',
            ['quantization.cpp'],
            extra_compile_args=['-O3', '-std=c++17']
        )
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    }
)
