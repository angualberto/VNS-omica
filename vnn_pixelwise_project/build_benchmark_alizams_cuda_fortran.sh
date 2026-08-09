#!/usr/bin/env bash
set -euo pipefail

nvfortran -cuda -gpu=cc86,cuda13.1 -O3 benchmark_alizams_cuda_fortran.cuf -o benchmark_alizams_cuda_fortran
echo "CUDA Fortran benchmark compiled: benchmark_alizams_cuda_fortran"
