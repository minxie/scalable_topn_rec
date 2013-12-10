from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

setup(
    cmdclass = {'build_ext': build_ext}, include_dirs = [numpy.get_include()],
    ext_modules=[Extension("topn_buffer", ["topn_buffer.pyx"]),
                 Extension("latent_model_cython", ["latent_model_cython.pyx"]),
                 Extension("sgd_mf_machine_cython", ["sgd_mf_machine_cython.pyx"], language="c++"),
                 Extension("sgd_mf_machine_cython_dynamic", ["sgd_mf_machine_cython_dynamic.pyx"], language="c++"),
                 Extension("sgd_mf_machine_cython_static_kd", ["sgd_mf_machine_cython_static_kd.pyx"], language="c++"),
                 Extension("sgd_mf_machine_cython_dynamic_kd", ["sgd_mf_machine_cython_dynamic_kd.pyx"], language="c++"),
                 Extension("sgd_mf_machine_cython_static_mtree", ["sgd_mf_machine_cython_static_mtree.pyx"], language="c++"),
                 Extension("sgd_mf_machine_cython_dynamic_mtree", ["sgd_mf_machine_cython_dynamic_mtree.pyx"], language="c++"),
                 Extension("sgd_mf_machine_cython_static_onion", ["sgd_mf_machine_cython_static_onion.pyx"], language="c++"),
                 Extension("sgd_mf_machine_cython_dynamic_onion", ["sgd_mf_machine_cython_dynamic_onion.pyx"], language="c++"),
                 Extension("sgd_cython", ["sgd_cython.pyx"]),
                 Extension("sgd_cython_dynamic", ["sgd_cython_dynamic.pyx"]),
                 Extension("sgd_cython_static_kd", ["sgd_cython_static_kd.pyx"]),
                 Extension("sgd_cython_dynamic_kd", ["sgd_cython_dynamic_kd.pyx"]),
                 Extension("sgd_cython_static_mtree", ["sgd_cython_static_mtree.pyx"]),
                 Extension("sgd_cython_dynamic_mtree", ["sgd_cython_dynamic_mtree.pyx"]),
                 Extension("sgd_cython_static_onion", ["sgd_cython_static_onion.pyx"]),
                 Extension("sgd_cython_dynamic_onion", ["sgd_cython_dynamic_onion.pyx"])]
)
