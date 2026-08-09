#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/media/angualberto/423ff97c-bf80-4bc9-9621-cf3ec081fd25/iaimgem/.venv/bin/python}"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export FC="${FC:-gfortran}"
export F90="${F90:-gfortran}"

"$PYTHON_BIN" -m numpy.f2py -c -m alizams_color_omp alizams_color_kernel_omp.f90 \
  --f90flags='-O3 -march=native -fopenmp -ffast-math' -lgomp

echo "Backend Fortran OpenMP compilado: alizams_color_omp"
