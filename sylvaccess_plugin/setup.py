from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "sylvaccess_plugin.sylvaccess_cython3",
        sources=["sylvaccess_cython3.pyx"],
        include_dirs=[np.get_include()],  # Include NumPy include directory
        extra_compile_args=["/O2", "/W3", "/GL"],
    )
]

setup(
    ext_modules=cythonize(ext_modules),
    include_dirs=[np.get_include()],  # Include NumPy include directory
)
