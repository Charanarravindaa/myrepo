"""Builds the optional Cython acceleration extension for the PPM hot path.

Falls back to pure Python if Cython isn't available; the extension is
optional and the library is fully functional without it (just ~10-50x
slower on PPM-heavy pipelines).
"""

from setuptools import setup

try:
    from Cython.Build import cythonize

    extensions = cythonize(
        ["vrle/_ppm_native.pyx"],
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    )
except ImportError:
    extensions = []

setup(ext_modules=extensions)
