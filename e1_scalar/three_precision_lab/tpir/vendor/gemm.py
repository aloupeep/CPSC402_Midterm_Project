"""FP16 INPUT / FP32 ACCUMULATION AND OUTPUT GEMM.

CUDA uses cublasGemmEx directly, via ctypes: no compiler, CuPy, or MAGMA
installation is needed. CPU uses rounded-input FP32 GEMM as a numerical
control, NEVER a claim of FP16 hardware execution. No silent CUDA fallback.
This wrapper has an independent cuBLAS handle and follows PyTorch's current
CUDA stream. Single-process/single-thread experiment use only.
"""
from __future__ import annotations
import ctypes as C
import ctypes.util
import glob
import os
from pathlib import Path
import sys
import torch


class UnsupportedBackend(RuntimeError): pass


class CublasHalfGemm:
    def __init__(self, device):
        self.device = torch.device(device)
        if self.device.type != "cuda": raise UnsupportedBackend("cuBLAS needs CUDA")
        self.handle = C.c_void_p()
        candidates = []
        if os.environ.get("MPLAB_CUBLAS_LIBRARY"):
            candidates.append(os.environ["MPLAB_CUBLAS_LIBRARY"])
        # Prefer the already-loaded library to mixing CUDA toolkit versions.
        try:
            for line in Path("/proc/self/maps").read_text().splitlines():
                p = line.split()[-1]
                if "/libcublas.so" in p and "Lt" not in p: candidates.append(p)
        except OSError: pass
        roots = list(sys.path) + [str(Path(torch.__file__).resolve().parent)]
        for root in roots:
            candidates += glob.glob(str(Path(root) / "nvidia/cublas/lib/libcublas.so*"))
            candidates += glob.glob(str(Path(root) / "lib/libcublas.so*"))
        found = ctypes.util.find_library("cublas")
        candidates += [found] if found else []
        candidates += ["libcublas.so.12", "libcublas.so.11", "libcublas.so"]
        errors = []
        for p in dict.fromkeys(candidates):
            try:
                self.lib = C.CDLL(p)
                self.path = p
                break
            except OSError as e: errors.append(str(e))
        else:
            raise UnsupportedBackend("Cannot load cuBLAS. Set MPLAB_CUBLAS_LIBRARY to the library shipped with your working PyTorch/CUDA installation. " + str(errors[-2:]))
        L = self.lib
        L.cublasCreate_v2.argtypes = [C.POINTER(C.c_void_p)]
        L.cublasDestroy_v2.argtypes = [C.c_void_p]
        L.cublasSetStream_v2.argtypes = [C.c_void_p, C.c_void_p]
        L.cublasSetPointerMode_v2.argtypes = [C.c_void_p, C.c_int]
        L.cublasSetMathMode.argtypes = [C.c_void_p, C.c_int]
        L.cublasGetVersion_v2.argtypes = [C.c_void_p, C.POINTER(C.c_int)]
        L.cublasGemmEx.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int,
                                  C.c_void_p, C.c_void_p, C.c_int, C.c_int,
                                  C.c_void_p, C.c_int, C.c_int, C.c_void_p,
                                  C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int]
        for name in ["cublasCreate_v2", "cublasDestroy_v2", "cublasSetStream_v2", "cublasSetPointerMode_v2", "cublasSetMathMode", "cublasGetVersion_v2", "cublasGemmEx"]:
            getattr(L, name).restype = C.c_int
        with torch.cuda.device(self.device):
            self.check(L.cublasCreate_v2(C.byref(self.handle)), "create")
            self.check(L.cublasSetPointerMode_v2(self.handle, 0), "host pointer mode")
            # CUBLAS_DEFAULT_MATH | DISALLOW_REDUCED_PRECISION_REDUCTION (16).
            self.check(L.cublasSetMathMode(self.handle, 16), "FP32 reduction math mode")
            v = C.c_int()
            self.check(L.cublasGetVersion_v2(self.handle, C.byref(v)), "version")
            self.version = v.value
        if self.version < 11000:
            raise UnsupportedBackend("This wrapper requires cuBLAS >= 11 (modern compute-type ABI).")
    @staticmethod
    def check(status, where):
        if status: raise RuntimeError(f"cuBLAS error {status} in {where}; no CPU fallback occurred")
    def __call__(self, a, b):
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
            raise ValueError("GEMM requires compatible 2D matrices")
        if a.device != b.device or a.device.type != "cuda":
            raise ValueError("GEMM inputs must share a CUDA device")
        if a.dtype != torch.float16 or b.dtype != torch.float16:
            raise ValueError("Expected FP16 operands")
        a, b = a.contiguous(), b.contiguous()
        m, k = a.shape; n = b.shape[1]
        out = torch.empty((m, n), dtype=torch.float32, device=a.device)
        alpha, beta = C.c_float(1.0), C.c_float(0.0)
        with torch.cuda.device(a.device):
            stream = torch.cuda.current_stream(a.device)
            self.check(self.lib.cublasSetStream_v2(self.handle, C.c_void_p(stream.cuda_stream)), "set stream")
            # Column-major BLAS computes C^T = B^T A^T over row-major storage.
            # CUDA_R_16F=2, CUDA_R_32F=0, CUBLAS_COMPUTE_32F=68,
            # CUBLAS_GEMM_DEFAULT_TENSOR_OP=99. Actual dispatch must be profiled.
            self.check(self.lib.cublasGemmEx(self.handle, 0, 0, n, m, k,
                C.byref(alpha), C.c_void_p(b.data_ptr()), 2, max(1,n),
                C.c_void_p(a.data_ptr()), 2, max(1,k), C.byref(beta),
                C.c_void_p(out.data_ptr()), 0, max(1,n), 68, 99), "FP16-input/FP32-output GEMM")
            # Safe allocator ownership even on a user-selected nondefault stream.
            a.record_stream(stream); b.record_stream(stream); out.record_stream(stream)
        return out
    def close(self):
        if self.handle.value:
            with torch.cuda.device(self.device):
                self.lib.cublasDestroy_v2(self.handle)
            self.handle = C.c_void_p()


_ENGINES = {}

def get_engine(device):
    d = torch.device(device)
    if d.type != "cuda": return None
    key = str(d)
    if key not in _ENGINES: _ENGINES[key] = CublasHalfGemm(d)
    return _ENGINES[key]


def half_product(a, b, scaling="none"):
    """Round FP32 operands to FP16, multiply, accumulate/output in FP32.

    'power2' scales each whole operand into [1,2) maximum magnitude before
    conversion, then rescales the result. This is NOT entrywise clipping,
    not a guarantee against underflow, and not Higham's exact squeeze scheme.
    """
    if scaling not in ("none", "power2"): raise ValueError(scaling)
    sa = torch.ones((), dtype=torch.float32, device=a.device)
    sb = sa.clone()
    if scaling == "power2":
        sa = torch.exp2(torch.floor(torch.log2(a.abs().amax().clamp_min(torch.finfo(torch.float32).tiny))))
        sb = torch.exp2(torch.floor(torch.log2(b.abs().amax().clamp_min(torch.finfo(torch.float32).tiny))))
        a, b = a / sa, b / sb
    ah, bh = a.to(torch.float16), b.to(torch.float16)
    if a.device.type == "cuda":
        out = get_engine(a.device)(ah, bh)
    elif a.device.type == "cpu":
        out = ah.float() @ bh.float()
    else:
        raise UnsupportedBackend("This release supports CPU and CUDA, not MPS half GEMM")
    if scaling == "power2": out = (out * sa) * sb
    return out
