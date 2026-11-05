# Project-local MAGMA 2.9.0 build for the mixed-precision campaign (sm_70 only).
BACKEND     = cuda
OPENBLASDIR = /home/linuxbrew/.linuxbrew
CUDADIR     = /usr/local/cuda
CC          = gcc
CXX         = g++
FORT        = /home/linuxbrew/.linuxbrew/bin/gfortran
NVCC        = nvcc
DEVCC       = $(NVCC)
ARCH      = ar
ARCHFLAGS = cr
RANLIB    = ranlib
GPU_TARGET = Volta
FPIC      = -fPIC
CFLAGS      = -O3 $(FPIC) -DNDEBUG -DADD_ -Wall -fopenmp -std=c99
CXXFLAGS    = -O3 $(FPIC) -DNDEBUG -DADD_ -Wall -fopenmp -std=c++11
FFLAGS      = -O3 $(FPIC) -DNDEBUG -DADD_ -Wall -Wno-unused-dummy-argument
F90FLAGS    = -O3 $(FPIC) -DNDEBUG -DADD_ -Wall -Wno-unused-dummy-argument -x f95-cpp-input
LDFLAGS     =     $(FPIC)                       -fopenmp
DEVCCFLAGS  = -O3         -DNDEBUG -DADD_
LIB       += -lopenblas
LIBDIR    += -L$(OPENBLASDIR)/lib
INC       +=
-include make.check-openblas
-include make.check-cuda
DEVCCFLAGS += -Xcompiler "$(FPIC)" -std=c++11
INC    += -I$(CUDADIR)/include
LIBDIR += -L$(CUDADIR)/lib64
LIB    += -lcublas -lcusparse -lcudart -lcudadevrt
