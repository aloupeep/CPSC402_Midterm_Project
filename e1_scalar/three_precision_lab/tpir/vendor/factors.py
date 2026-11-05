"""LU preconditioners and correctly pivoted, right-looking blocked LU."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import scipy.linalg as sla
import torch
from .gemm import half_product


class NumericalFailure(RuntimeError): pass


def require_finite(x, label):
    if not bool(torch.isfinite(x).all().item()):
        raise NumericalFailure(f"nonfinite_{label}")


@dataclass
class Factor:
    lu: torch.Tensor
    pivots: torch.Tensor
    kind: str
    normalize_rhs: bool = True
    metadata: dict = field(default_factory=dict)

    def solve(self, rhs):
        """Approximate inverse application. Normalize each RHS BEFORE downcasting.

        This protects small residuals from cast underflow. Scaling is undone in
        FP64. It does not prevent overflow inside a very ill-conditioned solve.
        """
        vector = rhs.ndim == 1
        rhs = rhs[:,None] if vector else rhs
        if self.normalize_rhs:
            scale = rhs.abs().amax(dim=0, keepdim=True)
            scale = torch.where(scale==0, torch.ones_like(scale), scale)
        else:
            scale = torch.ones((1,rhs.shape[1]),dtype=rhs.dtype,device=rhs.device)
        low_rhs = (rhs/scale).to(device=self.lu.device,dtype=self.lu.dtype)
        z = torch.linalg.lu_solve(self.lu, self.pivots, low_rhs)
        z = z.to(device=rhs.device,dtype=rhs.dtype) * scale
        require_finite(z, "correction")
        return z[:,0] if vector else z


@dataclass
class QRFactor:
    q: torch.Tensor
    r: torch.Tensor
    kind: str = "qr64"
    metadata: dict = field(default_factory=lambda: {"factor_arithmetic":"FP64 unpivoted Householder QR", "purpose":"independent pivot-growth control, not low-precision LU"})
    def solve(self,rhs):
        vector=rhs.ndim==1
        b=rhs[:,None] if vector else rhs
        y=torch.linalg.solve_triangular(self.r,self.q.T@b,upper=True)
        require_finite(y,"QR_solution")
        return y[:,0] if vector else y


def make_factor(a, kind="fp32", block_size=64, operand_scaling="none", normalize_rhs=True, diagnostic=False):
    if a.ndim!=2 or a.shape[0]!=a.shape[1]: raise ValueError("Expected square A")
    if block_size < 1: raise ValueError("block_size must be positive")
    if kind=="qr64":
        q,r=torch.linalg.qr(a.to(torch.float64))
        require_finite(r,"QR_factor")
        return QRFactor(q,r)
    dtype = torch.float64 if kind=="fp64" else torch.float32
    if kind == "input16":
        # Input-only quantization control. Not an FP16 arithmetic factorization.
        w = a.to(torch.float16).to(torch.float32)
    else: w = a.to(dtype).clone()
    require_finite(w,"factor_input")
    if kind in ("fp64","fp32","input16"):
        lu, piv, info = torch.linalg.lu_factor_ex(w, check_errors=False)
        code = int(info.item())
        if code != 0: raise NumericalFailure(f"LU_info_{code}")
        require_finite(lu,"LU")
        return Factor(lu,piv,kind,normalize_rhs,{"factor_arithmetic": "FP64" if kind=="fp64" else "FP32", "input_quantized_fp16": kind=="input16"})
    if kind not in ("blocked32","blocked16"):
        raise ValueError(f"Unknown factor type: {kind}")
    n=len(a)
    piv = torch.arange(1,n+1,dtype=torch.int32,device=a.device)
    stats = {"factor_arithmetic":"FP32 panels, triangular solves, factors, and Schur storage",
             "trailing_update":"FP32 GEMM" if kind=="blocked32" else "FP16 operands / FP32 accumulator and output",
             "execution":"CUDA cuBLAS" if a.device.type=="cuda" else "CPU rounded-input FP32 reference model",
             "block_size":block_size,"operand_scaling":operand_scaling,
             "tensor_core_dispatch":"requested/eligible, NOT verified by dtype; inspect profiler" if kind=="blocked16" and a.device.type=="cuda" else "not claimed"}
    raw_overflows=0; raw_zeroed=0; raw_entries=0; schur_max=0.0
    update_count=0; update_flops=0
    for k in range(0,n,block_size):
        end=min(n,k+block_size); width=end-k
        if diagnostic:
            schur_max=max(schur_max,float(w[k:,k:].abs().amax().item()))
        # Partial pivoting across ALL remaining rows, not only the square diagonal block.
        panel, local_piv, info = torch.linalg.lu_factor_ex(w[k:,k:end].clone(),check_errors=False)
        code=int(info.item())
        if code!=0: raise NumericalFailure(f"panel_LU_info_{code}_at_{k}")
        require_finite(panel,"panel")
        # One small CPU pivot read per panel. This overhead is INCLUDED in timings.
        p = np.arange(n-k)
        for j,pj in enumerate(local_piv.cpu().tolist()): p[j],p[pj-1]=p[pj-1],p[j]
        p = torch.as_tensor(p, dtype=torch.int64, device=a.device)
        # Swap previous L columns too, a frequently missed blocked-LU correctness step.
        w[k:,:] = w[k:,:].index_select(0,p)
        w[k:,k:end] = panel
        piv[k:end] = local_piv+k
        if end<n:
            update_count+=1; update_flops+=2*(n-end)*(n-end)*width
            u12=torch.linalg.solve_triangular(w[k:end,k:end],w[k:end,end:].clone(),upper=False,unitriangular=True)
            w[k:end,end:]=u12
            l21=w[end:,k:end]
            if kind=="blocked32": prod=l21 @ u12
            else:
                if diagnostic:
                    for z in (l21,u12):
                        h=z.to(torch.float16)
                        raw_overflows+=int((~torch.isfinite(h)).sum().item())
                        raw_zeroed+=int(((z!=0)&(h==0)).sum().item())
                        raw_entries+=z.numel()
                prod=half_product(l21,u12,operand_scaling)
            w[end:,end:] -= prod
    require_finite(w,"blocked_LU")
    stats.update(num_trailing_updates=update_count, trailing_update_flops=update_flops,
                 num_half_gemm_updates=update_count if kind=="blocked16" else 0,
                 nominal_lu_flops=2*n**3/3,
                 half_update_flop_fraction=(update_flops/(2*n**3/3)) if kind=="blocked16" else 0.0)
    if diagnostic:
        stats.update(raw_update_operand_overflows=raw_overflows,raw_update_operand_zeroed=raw_zeroed,
                     raw_update_operand_entries=raw_entries, sampled_schur_max=schur_max)
    return Factor(w,piv,kind,normalize_rhs,stats)


def factor_diagnostics(factor, a, exact_limit=256):
    """Untimed diagnostics of the FROZEN factors, not the rounded solve operator.

    E=I-U^{-1}L^{-1}PA assumes exact triangular solves with stored factors.
    rho(E) is a diagnostic, not a finite-precision convergence certificate.
    """
    if factor.kind=="qr64":
        aa=a.detach().cpu().numpy()
        q=factor.q.detach().cpu().numpy(); r=factor.r.detach().cpu().numpy()
        out=dict(factor.metadata)
        if len(aa)<=exact_limit:
            out["factor_relative_residual_inf"]=float(sla.norm(aa-q@r,np.inf)/sla.norm(aa,np.inf))
        return out
    lu=factor.lu.detach().cpu().numpy().astype(np.float64)
    piv=factor.pivots.detach().cpu().numpy().astype(int)-1
    aa=a.detach().cpu().numpy().astype(np.float64); n=len(aa)
    p=np.arange(n)
    for i,j in enumerate(piv): p[i],p[j]=p[j],p[i]
    l=np.tril(lu,-1)+np.eye(n); u=np.triu(lu)
    out=dict(factor.metadata)
    out.update(u_growth_proxy=float(np.max(np.abs(u))/max(np.max(np.abs(aa)),np.finfo(float).tiny)),
               min_abs_u_diagonal=float(np.min(np.abs(np.diag(u)))))
    if n<=exact_limit:
        try:
            out["factor_relative_residual_inf"]=float(sla.norm(aa[p,:]-l@u,np.inf)/sla.norm(aa,np.inf))
            za=sla.solve_triangular(u,sla.solve_triangular(l,aa[p,:],lower=True,unit_diagonal=True),lower=False)
            e=np.eye(n)-za
            out["frozen_error_operator_rho"]=float(np.max(np.abs(sla.eigvals(e))))
            out["frozen_error_operator_inf"]=float(sla.norm(e,np.inf))
            # Random directional defect also evaluated through actual rounded solves.
            rng=np.random.default_rng(12345); v=rng.standard_normal((n,3))
            av=torch.as_tensor(aa@v,dtype=torch.float64,device=a.device)
            z=factor.solve(av).cpu().numpy()
            out["rounded_apply_defect_probe"]=float(np.max(np.linalg.norm(v-z,axis=0)/np.linalg.norm(v,axis=0)))
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
            out["diagnostic_error"]=str(exc)
    return out
