#!/bin/bash
# Build the campaign driver. MAGMA 2.9.0 built from source under campaign/build_magma (GPU_TARGET=Volta, OpenBLAS from linuxbrew).
set -e
SRC=/home/worker/lab/campaign/src
MAGMA=/home/worker/lab/campaign/build_magma/magma-2.9.0
OUT=/home/worker/lab/campaign/build
FLAGS="-O3 -std=c++17 -arch=sm_70 -lineinfo -DHAVE_MAGMA -I$MAGMA/include -I/usr/local/cuda/include"
LIBS="-L$MAGMA/lib -lmagma -lcusolver -lcublas -lcublasLt -lcusparse -lcudart -lcrypto -L/home/linuxbrew/.linuxbrew/lib -lopenblas -Xlinker -rpath=$MAGMA/lib -Xlinker -rpath=/home/linuxbrew/.linuxbrew/lib -Xcompiler -fopenmp -Xlinker --disable-new-dtags"
echo "nvcc $FLAGS -o $OUT/mpsolve $SRC/mpsolve.cu $LIBS" | tee $OUT/build_command.txt
nvcc $FLAGS -o $OUT/mpsolve $SRC/mpsolve.cu $LIBS 2>&1 | tee $OUT/build_log.txt
sha256sum $OUT/mpsolve | tee $OUT/mpsolve.sha256
nvcc --version | tail -2 > $OUT/nvcc_version.txt
