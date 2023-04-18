`mkl_umath for SVE`
====

This repository provides Python module `mkl_umath for SVE` to speed up
transcendental-function calculation of NumPy on CPUs supporting the
AArch64 SVE instruction set.
`mkl_umath for SVE` is based on [mkl_umath](https://github.com/IntelPython/mkl_umath).

## Supported environment

- Linux running on AArch64 CPU which supports SVE instructions.

## Requirement
- Linux
- cmake
- Cython
- GCC (10.2.x or later is mandatory, because ACLE (Arm C Language Extension) support is needed.)
- ninja-build
- NumPy (1.24.x or later is recommended.)
- Python3 (3.9.x or later is recommended.)

## Build and Install
### Build SLEEF

Because [SLEEF (SIMD Library for for Evaluating Elementary Functions)](https://sleef.org)
is used as the vectorized implementation of transcendental functions,
SLEEF must be built beforehand.

```sh
git clone https://github.com/shibatch/sleef sleef
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=directory_path_where_you_want_to_install_sleef -DENFORCE_SVE=TRUE -DBUILD_SHARED_LIBS=FALSE -G "Ninja" ../sleef
ninja
ninja test (This step can be skipped.)
ninja install
```

### Build mkl_umath for SVE
```sh
git clone git@github.com:fujitsu/mkl_umath.git
cd mkl_umath
CC=gcc CXX=g++ SLEEF_PATH=directory_path_where_you_installed_sleef python setup.py build install
```

## Usage
```python
import numpy as np
a = np.double(np.random.random_sample(1024))
np.sin(a) # NumPy's transcendental function

import mkl_umath as um
um._patch.use_in_numpy()
np.sin(a) # mkl_umath for SVE
```

## LICENCE

`mkl_umath for SVE` is licensed under
[BSD 3-Clause "New" or "Revised" License](LICENSE).
Refer to the ["LICENSE"](LICENSE) file for the full license text and copyright notice.

This distribution includes and/or uses third party software governed by separate license terms.

- BSD 3-Clause "New" or "Revised" License
  - [NumPy](https://github.com/numpy/numpy)
  - [mkl_umath](https://github.com/IntelPython/mkl_umath)
- Boost Software License 1.0
  - [SLEEF](https://github.com/shibatch/sleef)

