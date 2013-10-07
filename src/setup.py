from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

setup(
    cmdclass = {'build_ext': build_ext}, include_dirs = [numpy.get_include()],
    ext_modules=[Extension("latent_model_cython", ["latent_model_cython.pyx"]),
                 Extension("sgd_mf_machine_cython", ["sgd_mf_machine_cython.pyx"]),
                 Extension("sgd_cython", ["sgd_cython.pyx"])]
)
