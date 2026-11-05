// mp_kernels.cuh — CUDA kernels: deterministic generators, conversions, pivoting, residual/metric evaluation.
#pragma once
#include "mp_common.cuh"

// ---------- generic elementwise
template<typename S,typename T> __global__ void k_convert(const S* a,T* b,size_t n){ size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i<n) b[i]=(T)a[i]; }
__global__ void k_f32_to_f16_count(const float* a,__half* b,size_t n,float scale,unsigned long long* cnt){
  // counts: cnt[0]=overflow (|x*scale|>65504), cnt[1]=subnormal-or-underflow (0<|x*scale|<2^-14)
  __shared__ unsigned long long c0,c1; if(threadIdx.x==0){c0=0;c1=0;} __syncthreads();
  size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; unsigned o=0,u=0;
  if(i<n){ float v=a[i]*scale; float av=fabsf(v); if(av>65504.f) o=1; else if(av>0.f && av<6.103515625e-05f) u=1; b[i]=__float2half(v); }
  if(o) atomicAdd(&c0,1ULL); if(u) atomicAdd(&c1,1ULL); __syncthreads(); if(threadIdx.x==0){ if(c0) atomicAdd(cnt,c0); if(c1) atomicAdd(cnt+1,c1);} }
// 2D conversion of an (rows x cols) FP32 submatrix (ld lds) into packed FP16 (ld ldd), counting overflow/subnormal events per block
__global__ void k_f32_to_f16_2d(const float* src,size_t lds,__half* dst,size_t ldd,int rows,int cols,unsigned long long* cnt){
  __shared__ unsigned long long c0,c1; if(threadIdx.x==0){c0=0;c1=0;} __syncthreads();
  int i=blockIdx.x*blockDim.x+threadIdx.x; int j=blockIdx.y; unsigned o=0,u=0;
  if(i<rows&&j<cols){ float v=src[i+(size_t)j*lds]; float av=fabsf(v); if(av>65504.f) o=1; else if(av>0.f && av<6.103515625e-05f) u=1; dst[i+(size_t)j*ldd]=__float2half(v); }
  if(o) atomicAdd(&c0,1ULL); if(u) atomicAdd(&c1,1ULL); __syncthreads(); if(threadIdx.x==0){ if(c0) atomicAdd(cnt,c0); if(c1) atomicAdd(cnt+1,c1);} }
template<typename T> __global__ void k_scale(T* a,size_t n,T s){ size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i<n) a[i]*=s; }
template<typename T> __global__ void k_fill(T* a,size_t n,T v){ size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i<n) a[i]=v; }
// w -= (*h) * v   (device scalar; used in modified Gram-Schmidt)
__global__ void k_axpy_dev(double* w,const double* v,const double* h,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) w[i]=fma(-*h,v[i],w[i]); }
template<typename T> __global__ void k_add(T* x,const double* d,size_t n,double scale){ size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i<n) x[i]=(T)((double)x[i]+scale*(double)d[i]); }
// row/column diagonal scaling: A[i,j] *= r[i]*c[j]
template<typename T> __global__ void k_rowcol_scale(T* A,int n,size_t lda,const double* r,const double* c){ int i=blockIdx.x*blockDim.x+threadIdx.x; int j=blockIdx.y; if(i<n) A[i+(size_t)j*lda]=(T)((double)A[i+(size_t)j*lda]*r[i]*c[j]); }
template<typename T> __global__ void k_vec_scale_rows(T* X,int n,size_t ldx,int nrhs,const double* r,int inverse){ int i=blockIdx.x*blockDim.x+threadIdx.x; int j=blockIdx.y; if(i<n&&j<nrhs){ double s=inverse? 1.0/r[i] : r[i]; X[i+(size_t)j*ldx]=(T)((double)X[i+(size_t)j*ldx]*s);} }

// ---------- LAPACK-style row interchanges (sequential swaps, applied per column by one thread)
template<typename T> __global__ void k_laswp(T* A,size_t lda,int k,int kb,const int64_t* ipiv_local,int col_begin,int col_end){
  int j=col_begin+blockIdx.x*blockDim.x+threadIdx.x; if(j>=col_end) return; T* col=A+(size_t)j*lda;
  for(int i=0;i<kb;i++){ int64_t ip=k+ipiv_local[k+i]-1; if(ip!=k+i){ T t=col[k+i]; col[k+i]=col[ip]; col[ip]=t; } } }
template<typename T> __global__ void k_laswp_global(T* A,size_t lda,int k,int kb,const int* ipiv_global1,int col_begin,int col_end){
  int j=col_begin+blockIdx.x*blockDim.x+threadIdx.x; if(j>=col_end) return; T* col=A+(size_t)j*lda;
  for(int i=0;i<kb;i++){ int ip=ipiv_global1[k+i]-1; if(ip!=k+i){ T t=col[k+i]; col[k+i]=col[ip]; col[ip]=t; } } }
__global__ void k_i32_to_i64(const int* a,int64_t* b,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) b[i]=(int64_t)a[i]; }
__global__ void k_ipiv_localize_to_global(int64_t* ipiv,int k,int kb){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<kb) ipiv[k+i]=k+ipiv[k+i]; }

// ---------- deterministic prescribed-spectrum generator: pairwise Givens stages on columns
// pairs: (2p,2p+1) entries are column indices; cs: (c,s) per pair. Operation order fixed: new1 = fma(c,a, s*b), new2 = fma(c,b, -(s*a)).
__global__ void k_rot_cols(double* A,int n,size_t lda,const int* pairs,const double* cs,int npairs){
  int p=blockIdx.y; if(p>=npairs) return; int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
  int c1=pairs[2*p],c2=pairs[2*p+1]; double c=cs[2*p],s=cs[2*p+1];
  double a=A[i+(size_t)c1*lda], b=A[i+(size_t)c2*lda];
  A[i+(size_t)c1*lda]=__fma_rn(c,a,__dmul_rn(s,b)); A[i+(size_t)c2*lda]=__fma_rn(c,b,-__dmul_rn(s,a)); }
__global__ void k_transpose_inplace(double* A,int n,size_t lda){
  __shared__ double t1[32][33],t2[32][33]; int bi=blockIdx.y,bj=blockIdx.x; if(bj<bi) return; // handle tiles with bj>=bi
  int r0=bi*32,c0=bj*32; int tx=threadIdx.x,ty=threadIdx.y; // block 32x8
  for(int k=ty;k<32;k+=8){ int r=r0+k,c=c0+tx; if(r<n&&c<n) t1[k][tx]=A[r+(size_t)c*lda]; int r2=c0+k,cc=r0+tx; if(r2<n&&cc<n) t2[k][tx]=A[r2+(size_t)cc*lda]; }
  __syncthreads();
  for(int k=ty;k<32;k+=8){ int r=r0+k,c=c0+tx; if(r<n&&c<n) A[r+(size_t)c*lda]=t2[tx][k]; if(bi!=bj){ int r2=c0+k,cc=r0+tx; if(r2<n&&cc<n) A[r2+(size_t)cc*lda]=t1[tx][k]; } } }

// ---------- RBF kernel matrix: exp(-|xi-xj|^2/(2 ell^2)) + nugget*delta_ij, points in [0,1]^d
__global__ void k_rbf(double* A,int n,size_t lda,const double* pts,int dim,double ell,double nugget){
  int i=blockIdx.x*blockDim.x+threadIdx.x; int j=blockIdx.y; if(i>=n) return; double d2=0; for(int k=0;k<dim;k++){ double t=pts[i+(size_t)k*n]-pts[j+(size_t)k*n]; d2=fma(t,t,d2);} double v=exp(-d2/(2.0*ell*ell)); if(i==j) v+=nugget; A[i+(size_t)j*lda]=v; }

// ---------- dyadic exact-reference family: C_ij integer in [-m,m] from hash(seed,i,j); A = 2^-s C; x* = ±1 from hash(seed^1,j); b = 2^-s (C x*) by int64 accumulation.
__device__ __host__ inline int dyadic_entry(uint64_t seed,int i,int j,int m){ uint64_t h=hash3(seed,(uint64_t)i,(uint64_t)j); return (int)(h%(uint64_t)(2*m+1))-m; }
__device__ __host__ inline int dyadic_sign(uint64_t seed,int j){ return (hash3(seed^0xA5A5A5A5ULL,(uint64_t)j,7ULL)&1ULL)? 1:-1; }
template<typename T> __global__ void k_dyadic_A(T* A,int n,size_t lda,uint64_t seed,int m,double scale){ int i=blockIdx.x*blockDim.x+threadIdx.x; int j=blockIdx.y; if(i<n) A[i+(size_t)j*lda]=(T)(dyadic_entry(seed,i,j,m)*scale); }
__global__ void k_dyadic_b(double* b,int n,uint64_t seed,int m,double scale,int nrhs,uint64_t rhsseed){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
  for(int c=0;c<nrhs;c++){ long long s=0; for(int j=0;j<n;j++) s+=(long long)dyadic_entry(seed,i,j,m)*dyadic_sign(rhsseed+c,j); b[i+(size_t)c*n]=(double)s*scale; } }
__global__ void k_dyadic_x(double* x,int n,int nrhs,uint64_t rhsseed){ int j=blockIdx.x*blockDim.x+threadIdx.x; if(j>=n) return; for(int c=0;c<nrhs;c++) x[j+(size_t)c*n]=(double)dyadic_sign(rhsseed+c,j); }

// ---------- residual + componentwise denominators. R = B - A X (or R = -A X when B==nullptr); T = |A||X| + |B| (per row, per rhs).
// A: TA (float or double) col-major with lda; X,B: double col-major (ldx=ldb=n). Accumulation in double, sequential over j (deterministic).
// Each thread owns one row; processes up to 8 RHS at once. Shared-memory tile of X rows for broadcast reads.
template<typename TA,int NR> __global__ void k_resid(const TA* __restrict__ A,int n,size_t lda,const double* __restrict__ X,const double* __restrict__ B,double* __restrict__ R,double* __restrict__ Tden,int c0){
  __shared__ double xs[256*NR]; int i=blockIdx.x*blockDim.x+threadIdx.x; double s[NR],t[NR];
  #pragma unroll
  for(int c=0;c<NR;c++){ s[c]=0.0; t[c]=0.0; }
  for(int j0=0;j0<n;j0+=256){ int jl=min(256,n-j0); __syncthreads();
    for(int q=threadIdx.x;q<jl*NR;q+=blockDim.x){ int c=q/jl,jj=q-c*jl; xs[c*256+jj]=X[(size_t)(c0+c)*n+j0+jj]; }
    __syncthreads(); if(i<n){ const TA* ap=A+i+(size_t)j0*lda; for(int jj=0;jj<jl;jj++){ double a=(double)ap[(size_t)jj*lda]; double aa=fabs(a);
      #pragma unroll
      for(int c=0;c<NR;c++){ double x=xs[c*256+jj]; s[c]=fma(a,x,s[c]); t[c]=fma(aa,fabs(x),t[c]); } } } }
  if(i<n){
    #pragma unroll
    for(int c=0;c<NR;c++){ double bv=B? B[i+(size_t)(c0+c)*n]:0.0; R[i+(size_t)(c0+c)*n]=bv-s[c]; Tden[i+(size_t)(c0+c)*n]=t[c]+fabs(bv); } } }
// single-precision residual control: all arithmetic in float (R = B - A X with float accumulation)
__global__ void k_resid_f32(const float* A,int n,size_t lda,const float* X,const float* B,float* R,int nrhs){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
  for(int c=0;c<nrhs;c++){ float s=0.f; const float* xp=X+(size_t)c*n; for(int j=0;j<n;j++) s=fmaf(A[i+(size_t)j*lda],xp[j],s); R[i+(size_t)c*n]=B[i+(size_t)c*n]-s; } }
// double-double (compensated) residual: returns hi and lo parts; used for references and audits only.
template<typename TA> __global__ void k_resid_dd(const TA* A,int n,size_t lda,const double* X,const double* B,double* Rhi,double* Rlo,int nrhs){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
  for(int c=0;c<nrhs;c++){ double sh=B[i+(size_t)c*n], sl=0.0; const double* xp=X+(size_t)c*n;
    for(int j=0;j<n;j++){ double a=(double)A[i+(size_t)j*lda], x=xp[j]; double p=-a*x; double pe=fma(-a,x,-p); // two-prod
      double s=sh+p; double bb=s-sh; double e=(sh-(s-bb))+(p-bb); sh=s; sl+=e+pe; }
    double s=sh+sl; sl=sl-(s-sh); sh=s; Rhi[i+(size_t)c*n]=sh; Rlo[i+(size_t)c*n]=sl; } }
// row sums of |A| (double) for the infinity norm
template<typename TA> __global__ void k_rowabs_sum(const TA* A,int n,size_t lda,double* out){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return; double s=0; for(int j=0;j<n;j++) s+=fabs((double)A[i+(size_t)j*lda]); out[i]=s; }
template<typename T> __global__ void k_absmax_block(const T* a,size_t n,double* out){ __shared__ double sh[256]; size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; double m=0; for(size_t k=i;k<n;k+=(size_t)gridDim.x*blockDim.x){ double v=fabs((double)a[k]); if(v>m) m=v; } sh[threadIdx.x]=m; __syncthreads(); for(int s=128;s>0;s>>=1){ if(threadIdx.x<s) sh[threadIdx.x]=fmax(sh[threadIdx.x],sh[threadIdx.x+s]); __syncthreads(); } if(threadIdx.x==0) out[blockIdx.x]=sh[0]; }
// per-column statistics: out[c*4+0]=max|R|, [1]=max|X|, [2]=max_i |R_i|/T_i (0/0->0, x/0->inf), [3]=sum R^2 ; out2[c]=worst row index
__global__ void k_colstats(const double* R,const double* X,const double* T,int n,int nrhs,double* out,int* worst){ __shared__ double s0[256],s1[256],s2[256],s3[256]; __shared__ int si[256]; int c=blockIdx.x; if(c>=nrhs) return; double m0=0,m1=0,m2=0,m3=0; int wi=-1;
  for(int i=threadIdx.x;i<n;i+=blockDim.x){ double r=fabs(R[i+(size_t)c*n]); double x=fabs(X[i+(size_t)c*n]); double t=T[i+(size_t)c*n]; double q; if(r==0.0) q=0.0; else if(t==0.0) q=INFINITY; else q=r/t; if(!(r<=m0)) m0=r; if(!(x<=m1)) m1=x; if(!(q<=m2)){ m2=q; wi=i;} m3=fma(r,r,m3); }
  s0[threadIdx.x]=m0; s1[threadIdx.x]=m1; s2[threadIdx.x]=m2; s3[threadIdx.x]=m3; si[threadIdx.x]=wi; __syncthreads();
  for(int s=128;s>0;s>>=1){ if(threadIdx.x<s){ s0[threadIdx.x]=fmax(s0[threadIdx.x],s0[threadIdx.x+s]); s1[threadIdx.x]=fmax(s1[threadIdx.x],s1[threadIdx.x+s]); if(!(s2[threadIdx.x]>=s2[threadIdx.x+s])){ s2[threadIdx.x]=s2[threadIdx.x+s]; si[threadIdx.x]=si[threadIdx.x+s]; } s3[threadIdx.x]+=s3[threadIdx.x+s]; } __syncthreads(); }
  if(threadIdx.x==0){ out[c*4]=s0[0]; out[c*4+1]=s1[0]; out[c*4+2]=s2[0]; out[c*4+3]=s3[0]; worst[c]=si[0]; } }
// forward-error statistics vs a reference: out[c*6]: max|x-xr|, max|xr|, sum (x-xr)^2, sum xr^2, max_{xr!=0} |x-xr|/|xr|, count(xr==0) ; out[c*6+5] also max abs err over zero comps stored in out2
__global__ void k_fwdstats(const double* X,const double* XR,int n,int nrhs,double* out){ __shared__ double s[6][256]; int c=blockIdx.x; if(c>=nrhs) return; double m0=0,m1=0,m2=0,m3=0,m4=0,m5=0,zc=0;
  for(int i=threadIdx.x;i<n;i+=blockDim.x){ double x=X[i+(size_t)c*n], xr=XR[i+(size_t)c*n]; double e=fabs(x-xr); if(!(e<=m0)) m0=e; double a=fabs(xr); if(a>m1) m1=a; m2=fma(e,e,m2); m3=fma(xr,xr,m3); if(xr!=0.0){ double q=e/a; if(!(q<=m4)) m4=q; } else { zc+=1.0; if(!(e<=m5)) m5=e; } }
  s[0][threadIdx.x]=m0; s[1][threadIdx.x]=m1; s[2][threadIdx.x]=m2; s[3][threadIdx.x]=m3; s[4][threadIdx.x]=m4; s[5][threadIdx.x]=m5; __shared__ double sz[256]; sz[threadIdx.x]=zc; __syncthreads();
  for(int st=128;st>0;st>>=1){ if(threadIdx.x<st){ s[0][threadIdx.x]=fmax(s[0][threadIdx.x],s[0][threadIdx.x+st]); s[1][threadIdx.x]=fmax(s[1][threadIdx.x],s[1][threadIdx.x+st]); s[2][threadIdx.x]+=s[2][threadIdx.x+st]; s[3][threadIdx.x]+=s[3][threadIdx.x+st]; s[4][threadIdx.x]=fmax(s[4][threadIdx.x],s[4][threadIdx.x+st]); s[5][threadIdx.x]=fmax(s[5][threadIdx.x],s[5][threadIdx.x+st]); sz[threadIdx.x]+=sz[threadIdx.x+st]; } __syncthreads(); }
  if(threadIdx.x==0){ out[c*7]=s[0][0]; out[c*7+1]=s[1][0]; out[c*7+2]=s[2][0]; out[c*7+3]=s[3][0]; out[c*7+4]=s[4][0]; out[c*7+5]=s[5][0]; out[c*7+6]=sz[0]; } }
// nonfinite count
template<typename T> __global__ void k_count_nonfinite(const T* a,size_t n,unsigned long long* cnt){ size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i<n && !isfinite((double)a[i])) atomicAdd(cnt,1ULL); }
// power-of-two row/column equilibration helpers: out[i] = 2^-floor(log2(max_j |A_ij|)) ; column version with strided loop
template<typename TA> __global__ void k_row_absmax(const TA* A,int n,size_t lda,double* out){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return; double m=0; for(int j=0;j<n;j++){ double v=fabs((double)A[i+(size_t)j*lda]); if(v>m) m=v;} out[i]=m; }
template<typename TA> __global__ void k_col_absmax(const TA* A,int n,size_t lda,double* out){ __shared__ double sh[256]; int j=blockIdx.x; double m=0; for(int i=threadIdx.x;i<n;i+=blockDim.x){ double v=fabs((double)A[i+(size_t)j*lda]); if(v>m) m=v;} sh[threadIdx.x]=m; __syncthreads(); for(int s=128;s>0;s>>=1){ if(threadIdx.x<s) sh[threadIdx.x]=fmax(sh[threadIdx.x],sh[threadIdx.x+s]); __syncthreads(); } if(threadIdx.x==0) out[j]=sh[0]; }
__global__ void k_pow2_recip(double* v,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return; double m=v[i]; if(m<=0.0||!isfinite(m)){ v[i]=1.0; return;} int e; frexp(m,&e); v[i]=ldexp(1.0,-(e-1)); }

// ---------- host-side launch helpers
static inline unsigned grid1(size_t n,int bs=256){ size_t g=(n+bs-1)/bs; return (unsigned)std::min<size_t>(g, 2147483647u); }
template<typename S,typename T> void dev_convert(const S* a,T* b,size_t n){ k_convert<S,T><<<grid1(n),256>>>(a,b,n); }
template<typename TA> void dev_resid(const TA* A,int n,size_t lda,const double* X,const double* B,double* R,double* T,int nrhs){
  int c=0; while(c<nrhs){ int left=nrhs-c; if(left>=8){ k_resid<TA,8><<<grid1(n),256>>>(A,n,lda,X,B,R,T,c); c+=8; } else if(left>=4){ k_resid<TA,4><<<grid1(n),256>>>(A,n,lda,X,B,R,T,c); c+=4; } else if(left>=2){ k_resid<TA,2><<<grid1(n),256>>>(A,n,lda,X,B,R,T,c); c+=2; } else { k_resid<TA,1><<<grid1(n),256>>>(A,n,lda,X,B,R,T,c); c+=1; } } }
