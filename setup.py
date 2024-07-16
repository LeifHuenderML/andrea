from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import os
from pathlib import Path

class GetPybindInclude(object):
    def __init__(self, user=False):
        self.user = user
    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

def get_cuda_path():
    return os.environ.get('CUDA_PATH', '/usr/local/cuda')

cuda_path = get_cuda_path()

ext_modules = [
    Extension(
        'andrea._andrea',
        ['python/src/bindings.cpp', 
         'src/tensor.cpp', 
         'src/cuda.cu',
        ],
        include_dirs=[
            GetPybindInclude(),
            GetPybindInclude(user=True),
            'include',
            os.path.join(cuda_path, 'include')
        ],
        library_dirs=[os.path.join(cuda_path, 'lib64')],
        libraries=['cudart'],
        language='c++',
        extra_compile_args={
            'cxx': ['-std=c++17', '-fopenmp', '-fPIC', '-g', '-O0'],  # Added -g and -O0
            'nvcc': ['-O0', '--compiler-options', '-fPIC', '-std=c++14', '-g', '-G']  # Added -g and -G, changed -O2 to -O0
        },
        extra_link_args=['-fopenmp', '-g'],  # Added -g
    ),
]

def customize_compiler_for_nvcc(self):
    self.src_extensions.append('.cu')
    default_compiler_so = self.compiler_so
    original_compile = self._compile

    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            self.set_executable('compiler_so', 'nvcc')
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['cxx']
        original_compile(obj, src, ext, cc_args, postargs, pp_opts)
        self.compiler_so = default_compiler_so

    self._compile = _compile

class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)

setup(
    name='andrea',
    version='0.0.1',
    author='Leif Huender',
    author_email='leifhuenderai@gmail.com',
    description='Tensor and autograd library.',
    long_description='',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.4'],
    setup_requires=['pybind11>=2.4'],
    cmdclass={'build_ext': custom_build_ext},
    zip_safe=False,
    packages=['andrea'],
    package_dir={'': 'python'}
)
