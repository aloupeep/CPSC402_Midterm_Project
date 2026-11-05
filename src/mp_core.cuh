// mp_core.cuh — problems, factorizations (cuSOLVER native + custom blocked FP16/FP32 trailing update),
// stationary / FGMRES / left-GMRES iterative refinement, recovery policies, cuSOLVER IRS wrapper, metrics, references.
#pragma once
#include "mp_kernels.cuh"
#ifdef HAVE_MAGMA
#include "magma_v2.h"
#endif

struct Ctx { cublasHandle_t cb=nullptr; cusolverDnHandle_t cs=nullptr; cusolverDnParams_t params=nullptr;
  void init(){ CUBLAS_CK(cublasCreate(&cb)); CUBLAS_CK(cublasSetMathMode(cb,CUBLAS_DEFAULT_MATH)); CUSOLVER_CK(cusolverDnCreate(&cs)); CUSOLVER_CK(cusolverDnCreateParams(&params)); }
};

// ============================================================ problem definition and deterministic generation
struct CaseSpec { int n=1024,nrhs=1; std::string family="prescribed",spectrum="geometric",dtype="fp64",rhs="random"; double kappa=100; uint64_t seed=9200;
  double ell=0.25,nugget=1e-3; int dim=3; int intmax=255,dyadic_exp=8; int grcar_k=3; int global_exp=0; int rowcol_span=0; uint64_t rowcol_seed=0; std::string case_id; };

struct Problem { CaseSpec spec; int n=0,nrhs=1; bool fp32=false; size_t lda=0;
  DBuf<float> A32; DBuf<double> A64; DBuf<double> B; DBuf<float> B32; double anorm_inf=0,amax=0; std::vector<double> bnorm_inf,bnorm_2,bnorm_inf_log;
  std::string a_sha,b_sha; std::vector<std::string> tile_sha; int ntiles=16; double gen_seconds=0,hash_seconds=0;
  DBuf<double> Xref; DBuf<double> Xexact; bool has_ref=false,has_exact=false; std::string ref_grade="unavailable",ref_method; double ref_resid_dd_rel=-1,ref_resid_dd_abs=-1; int ref_iters=0; double ref_seconds=0; std::string ref_note; std::vector<double> ref_vs_exact_finf;
  double kappa2_est=-1; double sigma_max_est=-1,sigma_min_est=-1; std::string kappa_method; double growth_ref=-1;
  double scale_assert_max_exp=0; bool scale_representable=true; double stored_min_abs=0,stored_max_abs=0;
  void* hA=nullptr; double* hB64=nullptr; float* hB32=nullptr; // pinned host copies (host_e2e mode)
  size_t a_bytes() const { return (size_t)n*n*(fp32?4:8); }
  const void* a_ptr() const { return fp32? (const void*)A32.p : (const void*)A64.p; }
  ~Problem(){ if(hA) cudaFreeHost(hA); if(hB64) cudaFreeHost(hB64); if(hB32) cudaFreeHost(hB32); }
};

static double spectrum_value(const std::string& sp,int i,int n,double kappa){
  if(sp=="geometric") return (n==1)?1.0: std::pow(kappa,-(double)i/(double)(n-1));
  if(sp=="clustered"){ int k=std::max(2,n/64); return (i>=n-k)? 1.0/kappa : 1.0; }
  if(sp=="one_small") return (i==n-1)? 1.0/kappa : 1.0;
  if(sp=="identity") return 1.0;
  throw MPError("unknown spectrum "+sp); }

// apply S random pair-rotation stages to the columns of A (n x n double, device). seed stream distinct per side.
static void apply_rotation_stages(double* A,int n,size_t lda,uint64_t seed,int side,int stages){
  Xoshiro rng(seed*1000003ULL + 17ULL*side + 0xC0FFEEULL);
  std::vector<int> perm(n),pairs; std::vector<double> cs; int np=n/2; pairs.resize(2*np); cs.resize(2*np);
  DBuf<int> dpairs(2*np,"pairs"); DBuf<double> dcs(2*np,"cs");
  for(int s=0;s<stages;s++){
    for(int i=0;i<n;i++) perm[i]=i; for(int i=n-1;i>0;i--){ int j=(int)rng.below(i+1); std::swap(perm[i],perm[j]); }
    for(int p=0;p<np;p++){ pairs[2*p]=perm[2*p]; pairs[2*p+1]=perm[2*p+1]; double th=rng.uniform()*6.283185307179586; cs[2*p]=std::cos(th); cs[2*p+1]=std::sin(th); }
    CUDA_CK(cudaMemcpy(dpairs.p,pairs.data(),sizeof(int)*2*np,cudaMemcpyHostToDevice)); CUDA_CK(cudaMemcpy(dcs.p,cs.data(),sizeof(double)*2*np,cudaMemcpyHostToDevice));
    dim3 grid(grid1(n),np); k_rot_cols<<<grid,256>>>(A,n,lda,dpairs.p,dcs.p,np); CUDA_CK(cudaGetLastError()); }
  CUDA_CK(cudaDeviceSynchronize()); }

static void transpose_inplace(double* A,int n,size_t lda){ dim3 grid((n+31)/32,(n+31)/32); dim3 blk(32,8); k_transpose_inplace<<<grid,blk>>>(A,n,lda); CUDA_CK(cudaGetLastError()); CUDA_CK(cudaDeviceSynchronize()); }

// Generate the stored problem on the device. Always generates in FP64 then rounds to the stored dtype (storage rounding is part of the recipe).
static void generate_problem(Ctx& ctx,Problem& P,const CaseSpec& spec,bool want_host_copy){
  P.spec=spec; P.n=spec.n; P.nrhs=spec.nrhs; P.fp32=(spec.dtype=="fp32"); P.lda=P.n; int n=P.n; size_t nn=(size_t)n*n; double t0=wall_now();
  P.A64.alloc(nn,"A64_gen");
  if(spec.family=="prescribed"){
    k_fill<double><<<grid1(nn),256>>>(P.A64.p,nn,0.0); std::vector<double> sig(n); for(int i=0;i<n;i++) sig[i]=spectrum_value(spec.spectrum,i,n,spec.kappa);
    CUDA_CK(cudaMemcpy2D(P.A64.p,(n+1)*sizeof(double),sig.data(),sizeof(double),sizeof(double),n,cudaMemcpyHostToDevice)); // diagonal
    int stages=(int)std::ceil(std::log2((double)n))+4;
    apply_rotation_stages(P.A64.p,n,P.lda,spec.seed,1,stages); transpose_inplace(P.A64.p,n,P.lda); apply_rotation_stages(P.A64.p,n,P.lda,spec.seed,2,stages);
  } else if(spec.family=="rbf"){
    Xoshiro rng(spec.seed*7919ULL+0xB1ULL); std::vector<double> pts((size_t)n*spec.dim); for(auto& v:pts) v=rng.uniform(); DBuf<double> dp(pts.size(),"pts"); CUDA_CK(cudaMemcpy(dp.p,pts.data(),pts.size()*8,cudaMemcpyHostToDevice));
    dim3 grid(grid1(n),n); k_rbf<<<grid,256>>>(P.A64.p,n,P.lda,dp.p,spec.dim,spec.ell,spec.nugget); CUDA_CK(cudaGetLastError()); CUDA_CK(cudaDeviceSynchronize());
  } else if(spec.family=="dyadic"){
    double scale=std::ldexp(1.0,-spec.dyadic_exp); dim3 grid(grid1(n),n); k_dyadic_A<double><<<grid,256>>>(P.A64.p,n,P.lda,spec.seed,spec.intmax,scale); CUDA_CK(cudaGetLastError()); CUDA_CK(cudaDeviceSynchronize());
  } else if(spec.family=="grcar"||spec.family=="wilkinson"){
    std::vector<double> h(nn,0.0);
    if(spec.family=="grcar"){ for(int j=0;j<n;j++) for(int i=0;i<n;i++){ if(i==j+1) h[i+(size_t)j*n]=-1.0; else if(j>=i && j<=i+spec.grcar_k) h[i+(size_t)j*n]=1.0; } }
    else { for(int j=0;j<n;j++) for(int i=0;i<n;i++){ if(i==j) h[i+(size_t)j*n]=1.0; else if(i>j) h[i+(size_t)j*n]=-1.0; if(j==n-1) h[i+(size_t)j*n]=1.0; } }
    CUDA_CK(cudaMemcpy(P.A64.p,h.data(),nn*8,cudaMemcpyHostToDevice));
  } else throw MPError("unknown family "+spec.family);
  // range transformations (exact power-of-two scalings)
  if(spec.rowcol_span>0){ Xoshiro rng(spec.rowcol_seed? spec.rowcol_seed : spec.seed*31ULL+5ULL); std::vector<double> r(n),c(n); for(int i=0;i<n;i++) r[i]=std::ldexp(1.0,(int)rng.below(spec.rowcol_span+1)-spec.rowcol_span/2); for(int i=0;i<n;i++) c[i]=std::ldexp(1.0,(int)rng.below(spec.rowcol_span+1)-spec.rowcol_span/2);
    DBuf<double> dr(n,"dr"),dc(n,"dc"); CUDA_CK(cudaMemcpy(dr.p,r.data(),n*8,cudaMemcpyHostToDevice)); CUDA_CK(cudaMemcpy(dc.p,c.data(),n*8,cudaMemcpyHostToDevice)); dim3 grid(grid1(n),n); k_rowcol_scale<double><<<grid,256>>>(P.A64.p,n,P.lda,dr.p,dc.p); CUDA_CK(cudaDeviceSynchronize()); }
  if(spec.global_exp!=0){ k_scale<double><<<grid1(nn),256>>>(P.A64.p,nn,std::ldexp(1.0,spec.global_exp)); CUDA_CK(cudaDeviceSynchronize()); }
  // RHS (FP64 master copy; exactly representable in fp32 when stored dtype is fp32 since we round)
  P.B.alloc((size_t)n*P.nrhs,"B");
  if(spec.rhs=="random"){ Xoshiro rng(spec.seed*104729ULL+0xBEEFULL+(uint64_t)P.nrhs); std::vector<double> hb((size_t)n*P.nrhs); for(size_t i=0;i<hb.size();i++) hb[i]=rng.normal(); if(spec.global_exp!=0) for(auto& v:hb) v=std::ldexp(v,spec.global_exp); CUDA_CK(cudaMemcpy(P.B.p,hb.data(),hb.size()*8,cudaMemcpyHostToDevice)); }
  else if(spec.rhs=="random_scaled"){ // columns with different scales 2^(4c)
    Xoshiro rng(spec.seed*104729ULL+0xBEEFULL+(uint64_t)P.nrhs); std::vector<double> hb((size_t)n*P.nrhs); for(int c=0;c<P.nrhs;c++) for(int i=0;i<n;i++) hb[i+(size_t)c*n]=std::ldexp(rng.normal(),4*c-2*P.nrhs+spec.global_exp); CUDA_CK(cudaMemcpy(P.B.p,hb.data(),hb.size()*8,cudaMemcpyHostToDevice)); }
  else if(spec.rhs=="exact_sign"){ if(spec.family!="dyadic") throw MPError("exact_sign rhs requires dyadic family"); if(spec.rowcol_span||spec.global_exp) throw MPError("exact_sign with scaling not supported");
    double scale=std::ldexp(1.0,-spec.dyadic_exp); k_dyadic_b<<<grid1(n),256>>>(P.B.p,n,spec.seed,spec.intmax,scale,P.nrhs,spec.seed*3ULL+11ULL); P.Xexact.alloc((size_t)n*P.nrhs,"Xexact"); k_dyadic_x<<<grid1(n),256>>>(P.Xexact.p,n,P.nrhs,spec.seed*3ULL+11ULL); CUDA_CK(cudaDeviceSynchronize()); P.has_exact=true;
    if((double)n*spec.intmax > (P.fp32? 16777216.0 : 9007199254740992.0)) throw MPError("dyadic row-sum bound violates exact representability"); }
  else if(spec.rhs=="unit_column"){ // b_c = A[:, j_c], exact sparse solution e_{j_c}
    Xoshiro rng(spec.seed*17ULL+3ULL); P.Xexact.alloc((size_t)n*P.nrhs,"Xexact"); k_fill<double><<<grid1((size_t)n*P.nrhs),256>>>(P.Xexact.p,(size_t)n*P.nrhs,0.0);
    for(int c=0;c<P.nrhs;c++){ int j=(int)rng.below(n); CUDA_CK(cudaMemcpy(P.B.p+(size_t)c*n,P.A64.p+(size_t)j*P.lda,n*8,cudaMemcpyDeviceToDevice)); double one=1.0; CUDA_CK(cudaMemcpy(P.Xexact.p+(size_t)c*n+j,&one,8,cudaMemcpyHostToDevice)); } P.has_exact=true; }
  else throw MPError("unknown rhs mode "+spec.rhs);
  // storage rounding
  if(P.fp32){ P.A32.alloc(nn,"A32"); dev_convert<double,float>(P.A64.p,P.A32.p,nn); P.B32.alloc((size_t)n*P.nrhs,"B32"); dev_convert<double,float>(P.B.p,P.B32.p,(size_t)n*P.nrhs); dev_convert<float,double>(P.B32.p,P.B.p,(size_t)n*P.nrhs); CUDA_CK(cudaDeviceSynchronize()); P.A64.release();
    if(spec.rhs=="unit_column"){ // b must be the stored fp32 column: recopy from A32 (already exact since B came from A64 rounding == A32 rounding)
    } }
  CUDA_CK(cudaDeviceSynchronize()); P.gen_seconds=wall_now()-t0;
  // diagnostics: ||A||_inf, max|A|, min nonzero |A|, nonfinite check
  { DBuf<double> rs(n,"rowsum"); if(P.fp32) k_rowabs_sum<float><<<grid1(n),256>>>(P.A32.p,n,P.lda,rs.p); else k_rowabs_sum<double><<<grid1(n),256>>>(P.A64.p,n,P.lda,rs.p); int idx=0; CUBLAS_CK(cublasIdamax(ctx.cb,n,rs.p,1,&idx)); CUDA_CK(cudaMemcpy(&P.anorm_inf,rs.p+idx-1,8,cudaMemcpyDeviceToHost));
    DBuf<double> bm(1024,"bm"); if(P.fp32) k_absmax_block<float><<<1024,256>>>(P.A32.p,nn,bm.p); else k_absmax_block<double><<<1024,256>>>(P.A64.p,nn,bm.p); std::vector<double> h(1024); CUDA_CK(cudaMemcpy(h.data(),bm.p,8192,cudaMemcpyDeviceToHost)); P.amax=0; for(double v:h) P.amax=std::max(P.amax,v); P.stored_max_abs=P.amax;
    DBuf<unsigned long long> cnt(1,"nf"); CUDA_CK(cudaMemset(cnt.p,0,8)); if(P.fp32) k_count_nonfinite<float><<<grid1(nn),256>>>(P.A32.p,nn,cnt.p); else k_count_nonfinite<double><<<grid1(nn),256>>>(P.A64.p,nn,cnt.p); unsigned long long c=0; CUDA_CK(cudaMemcpy(&c,cnt.p,8,cudaMemcpyDeviceToHost)); P.scale_representable=(c==0 && std::isfinite(P.anorm_inf) && P.amax>0); }
  // RHS norms
  { std::vector<double> hb((size_t)n*P.nrhs); CUDA_CK(cudaMemcpy(hb.data(),P.B.p,hb.size()*8,cudaMemcpyDeviceToHost)); P.bnorm_inf.assign(P.nrhs,0); P.bnorm_2.assign(P.nrhs,0); for(int c=0;c<P.nrhs;c++){ double m=0,s=0; for(int i=0;i<n;i++){ double v=std::fabs(hb[i+(size_t)c*n]); m=std::max(m,v); s+=v*v; } P.bnorm_inf[c]=m; P.bnorm_2[c]=std::sqrt(s);} }
  // hashes of stored bytes (column-major, stored dtype) + column tiles
  { double th=wall_now(); size_t bytes=P.a_bytes(); std::vector<char> h; if(want_host_copy){ CUDA_CK(cudaMallocHost(&P.hA,bytes)); CUDA_CK(cudaMemcpy(P.hA,P.a_ptr(),bytes,cudaMemcpyDeviceToHost)); } else { h.resize(bytes); CUDA_CK(cudaMemcpy(h.data(),P.a_ptr(),bytes,cudaMemcpyDeviceToHost)); }
    const char* base=want_host_copy? (const char*)P.hA : h.data(); P.a_sha=sha256_bytes(base,bytes); P.tile_sha.clear(); size_t colbytes=(size_t)n*(P.fp32?4:8); for(int t=0;t<P.ntiles;t++){ size_t c0=(size_t)n*t/P.ntiles,c1=(size_t)n*(t+1)/P.ntiles; P.tile_sha.push_back(sha256_bytes(base+c0*colbytes,(c1-c0)*colbytes)); }
    size_t bb=(size_t)n*P.nrhs*(P.fp32?4:8); std::vector<char> hbb(bb); CUDA_CK(cudaMemcpy(hbb.data(),P.fp32?(const void*)P.B32.p:(const void*)P.B.p,bb,cudaMemcpyDeviceToHost)); P.b_sha=sha256_bytes(hbb.data(),bb);
    if(want_host_copy){ if(P.fp32){ CUDA_CK(cudaMallocHost((void**)&P.hB32,bb)); memcpy(P.hB32,hbb.data(),bb);} else { CUDA_CK(cudaMallocHost((void**)&P.hB64,bb)); memcpy(P.hB64,hbb.data(),bb);} }
    P.hash_seconds=wall_now()-th; }
}

// ============================================================ factorizations
struct Workspace { size_t lwork_dev=0,lwork_host=0; DBuf<char> dwork; std::vector<char> hwork; DBuf<int> dinfo; DBuf<__half> hL,hU; int nb=256; DBuf<int> dipiv32,dipivinfo; std::string panel="cusolver";
#ifdef HAVE_MAGMA
  magma_queue_t mq=nullptr; magma_event_t mev[2];
#endif
  DBuf<float> tmp32; DBuf<double> tmp64; DBuf<double> R,T,Xd,D; DBuf<double> stats; DBuf<int> worst; DBuf<unsigned long long> cnt; DBuf<double> bm; int n=0,nrhs=0;
  void ensure(Ctx& ctx,int n_,int nrhs_,int nb_){ if(n==n_&&nrhs==nrhs_&&nb==nb_) return; n=n_; nrhs=nrhs_; nb=nb_; size_t a=0,b=0,h=0;
    CUSOLVER_CK(cusolverDnXgetrf_bufferSize(ctx.cs,ctx.params,n,n,CUDA_R_32F,nullptr,n,CUDA_R_32F,&a,&h)); lwork_dev=std::max(lwork_dev,a); lwork_host=std::max(lwork_host,h);
    CUSOLVER_CK(cusolverDnXgetrf_bufferSize(ctx.cs,ctx.params,n,n,CUDA_R_64F,nullptr,n,CUDA_R_64F,&a,&h)); lwork_dev=std::max(lwork_dev,a); lwork_host=std::max(lwork_host,h);
    CUSOLVER_CK(cusolverDnXgetrf_bufferSize(ctx.cs,ctx.params,n,nb,CUDA_R_32F,nullptr,n,CUDA_R_32F,&a,&h)); lwork_dev=std::max(lwork_dev,a); lwork_host=std::max(lwork_host,h);
    dwork.alloc(lwork_dev+256,"cusolver_work"); hwork.resize(lwork_host+256); dinfo.alloc((size_t)n/32+16,"dinfo"); hL.alloc((size_t)n*nb,"hL"); hU.alloc((size_t)n*nb,"hU"); dipiv32.alloc(n+64,"dipiv32"); dipivinfo.alloc(n+64,"dipivinfo");
#ifdef HAVE_MAGMA
    if(!mq){ magma_queue_create_from_cuda(0,(cudaStream_t)0,ctx.cb,nullptr,&mq); magma_event_create(&mev[0]); magma_event_create(&mev[1]); }
#endif
    tmp32.alloc((size_t)n*std::max(nrhs,1),"tmp32"); tmp64.alloc((size_t)n*std::max(nrhs,1),"tmp64"); R.alloc((size_t)n*nrhs,"R"); T.alloc((size_t)n*nrhs,"T"); Xd.alloc((size_t)n*nrhs,"Xd"); D.alloc((size_t)n*nrhs,"D"); stats.alloc(8*nrhs+64,"stats"); worst.alloc(nrhs+1,"worst"); cnt.alloc(4,"cnt"); bm.alloc(1024,"bm"); } };

struct Factors { int n=0; int prec=32; std::string kind; DBuf<float> LU32; DBuf<double> LU64; DBuf<int64_t> ipiv; DBuf<double> LU64_promoted; bool promoted=false;
  int scale_mode=0; double gscale=1.0; int gscale_exp=0; DBuf<double> dr,dc; int info=0; double growth_proxy=-1; unsigned long long conv_overflow=0,conv_subnormal=0; double t_factor=0,t_prepare=0,t_kernel=0; int panels=0; int nb=0; bool valid=false; };

static double dev_absmax(Workspace& W,const float* a,size_t n){ k_absmax_block<float><<<1024,256>>>(a,n,W.bm.p); std::vector<double> h(1024); CUDA_CK(cudaMemcpy(h.data(),W.bm.p,8192,cudaMemcpyDeviceToHost)); double m=0; for(double v:h) m=std::max(m,v); return m; }
static double dev_absmax(Workspace& W,const double* a,size_t n){ k_absmax_block<double><<<1024,256>>>(a,n,W.bm.p); std::vector<double> h(1024); CUDA_CK(cudaMemcpy(h.data(),W.bm.p,8192,cudaMemcpyDeviceToHost)); double m=0; for(double v:h) m=std::max(m,v); return m; }

// custom right-looking blocked LU with partial pivoting; FP32 storage; panel = cuSOLVER Xgetrf on the n_rem x nb panel;
// trailing update: FP16 operands with FP32 accumulation via cublasGemmEx (fp16_trailing=true) or cublasSgemm (false).
static void custom_blocked_lu(Ctx& ctx,Workspace& W,float* LU,int n,int nb,bool fp16_trailing,int64_t* ipiv,Factors& F){
  size_t lda=n; float one=1.f,mone=-1.f; CUDA_CK(cudaMemset(W.cnt.p,0,16)); F.panels=0;
  for(int k=0;k<n;k+=nb){ int kb=std::min(nb,n-k); int m=n-k; float* panel=LU+k+(size_t)k*lda; int pidx=k/nb; F.panels++;
    bool mpanel=false;
#ifdef HAVE_MAGMA
    if(W.panel=="magma"){ mpanel=true; magma_sgetrf_recpanel_native(m,kb,32,panel,lda,W.dipiv32.p+k,W.dipivinfo.p,W.dinfo.p+pidx,k,W.mev,W.mq,W.mq); k_i32_to_i64<<<grid1(kb),256>>>(W.dipiv32.p+k,ipiv+k,kb); }
#endif
    if(!mpanel) CUSOLVER_CK(cusolverDnXgetrf(ctx.cs,ctx.params,m,kb,CUDA_R_32F,panel,lda,ipiv+k,CUDA_R_32F,W.dwork.p,W.lwork_dev,W.hwork.data(),W.lwork_host,W.dinfo.p+pidx));
    if(mpanel){ if(k>0) k_laswp_global<float><<<grid1(k),256>>>(LU,lda,k,kb,W.dipiv32.p,0,k); if(k+kb<n) k_laswp_global<float><<<grid1(n-k-kb),256>>>(LU,lda,k,kb,W.dipiv32.p,k+kb,n); }
    else { if(k>0) k_laswp<float><<<grid1(k),256>>>(LU,lda,k,kb,ipiv,0,k); if(k+kb<n) k_laswp<float><<<grid1(n-k-kb),256>>>(LU,lda,k,kb,ipiv,k+kb,n); }
    if(k+kb<n){
      int m2=n-k-kb; float* U12=LU+k+(size_t)(k+kb)*lda; float* L21=LU+(k+kb)+(size_t)k*lda; float* A22=LU+(k+kb)+(size_t)(k+kb)*lda;
      CUBLAS_CK(cublasStrsm(ctx.cb,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,CUBLAS_DIAG_UNIT,kb,m2,&one,panel,lda,U12,lda));
      if(fp16_trailing){ // conversions into packed half buffers: L21 (m2 x kb, ld m2) and U12 (kb x m2, ld kb); gridDim.y <= 65535 so chunk columns
        { dim3 g(grid1(m2),kb); k_f32_to_f16_2d<<<g,256>>>(L21,lda,W.hL.p,m2,m2,kb,W.cnt.p); }
        for(int c=0;c<m2;c+=65535){ int cc=std::min(65535,m2-c); dim3 g(grid1(kb),cc); k_f32_to_f16_2d<<<g,256>>>(U12+(size_t)c*lda,lda,W.hU.p+(size_t)c*kb,kb,kb,cc,W.cnt.p); }
        CUBLAS_CK(cublasGemmEx(ctx.cb,CUBLAS_OP_N,CUBLAS_OP_N,m2,m2,kb,&mone,W.hL.p,CUDA_R_16F,m2,W.hU.p,CUDA_R_16F,kb,&one,A22,CUDA_R_32F,lda,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      } else { CUBLAS_CK(cublasSgemm(ctx.cb,CUBLAS_OP_N,CUBLAS_OP_N,m2,m2,kb,&mone,L21,lda,U12,lda,&one,A22,lda)); } }
    if(!mpanel) k_ipiv_localize_to_global<<<grid1(kb),256>>>(ipiv,k,kb); }
  CUDA_CK(cudaDeviceSynchronize()); CUDA_CK(cudaGetLastError());
  std::vector<int> infos(F.panels); CUDA_CK(cudaMemcpy(infos.data(),W.dinfo.p,4*F.panels,cudaMemcpyDeviceToHost)); F.info=0; for(int p=0;p<F.panels;p++) if(infos[p]>0){ F.info=p*nb+infos[p]; break; }
  unsigned long long c[2]; CUDA_CK(cudaMemcpy(c,W.cnt.p,16,cudaMemcpyDeviceToHost)); F.conv_overflow=c[0]; F.conv_subnormal=c[1]; }

#ifdef HAVE_MAGMA
static void magma_factor_into(Ctx& ctx,Workspace& W,const std::string& kind,Factors& F,int n){ std::vector<magma_int_t> ipiv(n); magma_int_t info=0;
  if(kind=="mg_ht") magma_htgetrf_gpu(n,n,F.LU32.p,n,ipiv.data(),&info); else if(kind=="mg_s") magma_sgetrf_native(n,n,F.LU32.p,n,ipiv.data(),&info); else if(kind=="mg_d") magma_dgetrf_native(n,n,F.LU64.p,n,ipiv.data(),&info); else throw MPError("bad magma factor kind");
  CUDA_CK(cudaDeviceSynchronize()); std::vector<int64_t> ip64(n); for(int i=0;i<n;i++) ip64[i]=ipiv[i]; CUDA_CK(cudaMemcpy(F.ipiv.p,ip64.data(),n*8,cudaMemcpyHostToDevice)); F.info=(int)info; F.panels=0; }
#endif
// factor_kind: cs32 | cs64 | het16 | cus32 | mg_ht | mg_s | mg_d ; scale_mode: 0 none, 1 global pow2 (max|A|->[0.5,1)), 2 two-sided pow2 row/col then global
static void factorize(Ctx& ctx,Workspace& W,const Problem& P,const std::string& kind,int nb,int scale_mode,bool promote_copy,Factors& F){
  int n=P.n; size_t nn=(size_t)n*n; double t0=wall_now(); F.n=n; F.kind=kind; F.scale_mode=scale_mode; F.nb=nb; F.valid=false; F.promoted=false; F.growth_proxy=-1; F.conv_overflow=F.conv_subnormal=0;
  F.prec=(kind=="cs64"||kind=="mg_d")?64:32; if(F.ipiv.n<(size_t)n) F.ipiv.alloc(n,"ipiv");
  if(F.prec==32){ if(F.LU32.n<nn) F.LU32.alloc(nn,"LU32"); if(P.fp32) CUDA_CK(cudaMemcpyAsync(F.LU32.p,P.A32.p,nn*4,cudaMemcpyDeviceToDevice)); else dev_convert<double,float>(P.A64.p,F.LU32.p,nn); }
  else { if(F.LU64.n<nn) F.LU64.alloc(nn,"LU64"); if(P.fp32) dev_convert<float,double>(P.A32.p,F.LU64.p,nn); else CUDA_CK(cudaMemcpyAsync(F.LU64.p,P.A64.p,nn*8,cudaMemcpyDeviceToDevice)); }
  F.gscale=1.0; F.gscale_exp=0;
  if(scale_mode==2){ if(F.dr.n<(size_t)n){ F.dr.alloc(n,"dr"); F.dc.alloc(n,"dc"); }
    if(F.prec==32){ k_row_absmax<float><<<grid1(n),256>>>(F.LU32.p,n,n,F.dr.p); k_pow2_recip<<<grid1(n),256>>>(F.dr.p,n); k_fill<double><<<grid1(n),256>>>(F.dc.p,n,1.0); dim3 g(grid1(n),n); k_rowcol_scale<float><<<g,256>>>(F.LU32.p,n,n,F.dr.p,F.dc.p); k_col_absmax<float><<<n,256>>>(F.LU32.p,n,n,F.dc.p); k_pow2_recip<<<grid1(n),256>>>(F.dc.p,n); k_fill<double><<<grid1(n),256>>>(W.tmp64.p,n,1.0); k_rowcol_scale<float><<<g,256>>>(F.LU32.p,n,n,W.tmp64.p,F.dc.p); }
    else { k_row_absmax<double><<<grid1(n),256>>>(F.LU64.p,n,n,F.dr.p); k_pow2_recip<<<grid1(n),256>>>(F.dr.p,n); k_fill<double><<<grid1(n),256>>>(F.dc.p,n,1.0); dim3 g(grid1(n),n); k_rowcol_scale<double><<<g,256>>>(F.LU64.p,n,n,F.dr.p,F.dc.p); k_col_absmax<double><<<n,256>>>(F.LU64.p,n,n,F.dc.p); k_pow2_recip<<<grid1(n),256>>>(F.dc.p,n); k_fill<double><<<grid1(n),256>>>(W.tmp64.p,n,1.0); k_rowcol_scale<double><<<g,256>>>(F.LU64.p,n,n,W.tmp64.p,F.dc.p); } }
  if(scale_mode>=1){ double m=(F.prec==32)? dev_absmax(W,F.LU32.p,nn) : dev_absmax(W,F.LU64.p,nn); if(m>0&&std::isfinite(m)){ int e; std::frexp(m,&e); F.gscale_exp=-e; F.gscale=std::ldexp(1.0,-e); if(F.prec==32) k_scale<float><<<grid1(nn),256>>>(F.LU32.p,nn,(float)F.gscale); else k_scale<double><<<grid1(nn),256>>>(F.LU64.p,nn,F.gscale); } }
  CUDA_CK(cudaDeviceSynchronize()); double amax_f=(F.prec==32)? dev_absmax(W,F.LU32.p,nn) : dev_absmax(W,F.LU64.p,nn); F.t_prepare=wall_now()-t0; double t1=wall_now();
  if(kind=="cs32"){ CUSOLVER_CK(cusolverDnXgetrf(ctx.cs,ctx.params,n,n,CUDA_R_32F,F.LU32.p,n,F.ipiv.p,CUDA_R_32F,W.dwork.p,W.lwork_dev,W.hwork.data(),W.lwork_host,W.dinfo.p)); CUDA_CK(cudaDeviceSynchronize()); CUDA_CK(cudaMemcpy(&F.info,W.dinfo.p,4,cudaMemcpyDeviceToHost)); F.panels=0; }
  else if(kind=="cs64"){ CUSOLVER_CK(cusolverDnXgetrf(ctx.cs,ctx.params,n,n,CUDA_R_64F,F.LU64.p,n,F.ipiv.p,CUDA_R_64F,W.dwork.p,W.lwork_dev,W.hwork.data(),W.lwork_host,W.dinfo.p)); CUDA_CK(cudaDeviceSynchronize()); CUDA_CK(cudaMemcpy(&F.info,W.dinfo.p,4,cudaMemcpyDeviceToHost)); F.panels=0; }
  else if(kind=="het16"){ custom_blocked_lu(ctx,W,F.LU32.p,n,nb,true,F.ipiv.p,F); }
  else if(kind=="cus32"){ custom_blocked_lu(ctx,W,F.LU32.p,n,nb,false,F.ipiv.p,F); }
#ifdef HAVE_MAGMA
  else if(kind=="mg_ht"||kind=="mg_s"||kind=="mg_d"){ magma_factor_into(ctx,W,kind,F,n); }
#endif
  else throw MPError("unknown factor kind "+kind);
  F.t_kernel=wall_now()-t1; double gm=(F.prec==32)? dev_absmax(W,F.LU32.p,nn) : dev_absmax(W,F.LU64.p,nn); F.growth_proxy=(amax_f>0)? gm/amax_f : -1;
  if(promote_copy && F.prec==32){ if(F.LU64_promoted.n<nn) F.LU64_promoted.alloc(nn,"LU64_promoted"); dev_convert<float,double>(F.LU32.p,F.LU64_promoted.p,nn); CUDA_CK(cudaDeviceSynchronize()); F.promoted=true; }
  F.t_factor=wall_now()-t0; F.valid=true; }

// d = A^{-1} r via factors (r,d: FP64 n x nrhs). apply64: use promoted FP64 copy of FP32 factors (application-precision experiment).
static void apply_inverse(Ctx& ctx,Workspace& W,const Factors& F,const double* r,double* d,int nrhs,bool apply64,int* trs_count){
  int n=F.n; size_t m=(size_t)n*nrhs; if(d!=r) CUDA_CK(cudaMemcpyAsync(d,r,m*8,cudaMemcpyDeviceToDevice));
  if(F.scale_mode==2){ dim3 g(grid1(n),nrhs); k_vec_scale_rows<double><<<g,256>>>(d,n,n,nrhs,F.dr.p,0); }
  if(F.prec==64){ CUSOLVER_CK(cusolverDnXgetrs(ctx.cs,ctx.params,CUBLAS_OP_N,n,nrhs,CUDA_R_64F,F.LU64.p,n,F.ipiv.p,CUDA_R_64F,d,n,W.dinfo.p)); }
  else if(apply64){ if(!F.promoted) throw MPError("promoted factors requested but not available"); CUSOLVER_CK(cusolverDnXgetrs(ctx.cs,ctx.params,CUBLAS_OP_N,n,nrhs,CUDA_R_64F,F.LU64_promoted.p,n,F.ipiv.p,CUDA_R_64F,d,n,W.dinfo.p)); }
  else { dev_convert<double,float>(d,W.tmp32.p,m); CUSOLVER_CK(cusolverDnXgetrs(ctx.cs,ctx.params,CUBLAS_OP_N,n,nrhs,CUDA_R_32F,F.LU32.p,n,F.ipiv.p,CUDA_R_32F,W.tmp32.p,n,W.dinfo.p)); dev_convert<float,double>(W.tmp32.p,d,m); }
  if(F.scale_mode==2){ dim3 g(grid1(n),nrhs); k_vec_scale_rows<double><<<g,256>>>(d,n,n,nrhs,F.dc.p,0); }
  if(F.gscale!=1.0) k_scale<double><<<grid1(m),256>>>(d,m,F.gscale);
  if(trs_count) (*trs_count)++; }

// ============================================================ metrics
struct Metrics { std::vector<double> rinf,r2,eta,omega,relres_inf,relres_2,xinf; std::vector<int> worst; bool finite=true; double eta_max=0,omega_max=0; std::vector<double> finf,f2,fcomp,fzero_maxabs; std::vector<int64_t> fzero_count; bool has_fwd=false; std::vector<double> einf,e2,ecomp,ezero_maxabs; std::vector<int64_t> ezero_count; bool has_exact=false; };
template<typename TA> static void eval_metrics_T(Ctx& ctx,Workspace& W,const Problem& P,const TA* A,const double* X,Metrics& M){
  int n=P.n,nrhs=P.nrhs; dev_resid<TA>(A,n,P.lda,X,P.B.p,W.R.p,W.T.p,nrhs); k_colstats<<<nrhs,256>>>(W.R.p,X,W.T.p,n,nrhs,W.stats.p,W.worst.p); CUDA_CK(cudaGetLastError());
  std::vector<double> st(4*nrhs); std::vector<int> wi(nrhs); CUDA_CK(cudaMemcpy(st.data(),W.stats.p,32*nrhs,cudaMemcpyDeviceToHost)); CUDA_CK(cudaMemcpy(wi.data(),W.worst.p,4*nrhs,cudaMemcpyDeviceToHost));
  M.rinf.assign(nrhs,0); M.r2.assign(nrhs,0); M.eta.assign(nrhs,0); M.omega.assign(nrhs,0); M.relres_inf.assign(nrhs,0); M.relres_2.assign(nrhs,0); M.xinf.assign(nrhs,0); M.worst.assign(nrhs,-1); M.finite=true; M.eta_max=0; M.omega_max=0;
  for(int c=0;c<nrhs;c++){ double ri=st[4*c],xi=st[4*c+1],om=st[4*c+2],r2=std::sqrt(st[4*c+3]); M.rinf[c]=ri; M.r2[c]=r2; M.xinf[c]=xi; M.worst[c]=wi[c]; double den=P.anorm_inf*xi+P.bnorm_inf[c];
    M.eta[c]=(ri==0)?0.0:((den==0)?INFINITY:ri/den); M.omega[c]=om; M.relres_inf[c]=(ri==0)?0.0:((P.bnorm_inf[c]==0)?INFINITY:ri/P.bnorm_inf[c]); M.relres_2[c]=(r2==0)?0.0:((P.bnorm_2[c]==0)?INFINITY:r2/P.bnorm_2[c]);
    if(!std::isfinite(ri)||!std::isfinite(xi)||!std::isfinite(om)) M.finite=false; M.eta_max=std::max(M.eta_max,M.eta[c]); M.omega_max=std::max(M.omega_max,M.omega[c]); }
  auto fwd=[&](const double* XR,std::vector<double>& finf,std::vector<double>& f2,std::vector<double>& fcomp,std::vector<double>& fz,std::vector<int64_t>& fzc){ k_fwdstats<<<nrhs,256>>>(X,XR,n,nrhs,W.stats.p); std::vector<double> s(7*nrhs); CUDA_CK(cudaMemcpy(s.data(),W.stats.p,56*nrhs,cudaMemcpyDeviceToHost)); finf.assign(nrhs,0); f2.assign(nrhs,0); fcomp.assign(nrhs,0); fz.assign(nrhs,0); fzc.assign(nrhs,0);
    for(int c=0;c<nrhs;c++){ double m0=s[7*c],m1=s[7*c+1],e2=std::sqrt(s[7*c+2]),x2=std::sqrt(s[7*c+3]); finf[c]=(m1>0)?m0/m1:(m0==0?0:INFINITY); f2[c]=(x2>0)?e2/x2:(e2==0?0:INFINITY); fcomp[c]=s[7*c+4]; fz[c]=s[7*c+5]; fzc[c]=(int64_t)s[7*c+6]; } };
  if(P.has_ref){ fwd(P.Xref.p,M.finf,M.f2,M.fcomp,M.fzero_maxabs,M.fzero_count); M.has_fwd=true; }
  if(P.has_exact){ fwd(P.Xexact.p,M.einf,M.e2,M.ecomp,M.ezero_maxabs,M.ezero_count); M.has_exact=true; } }
static void eval_metrics(Ctx& ctx,Workspace& W,const Problem& P,const double* X,Metrics& M){ if(P.fp32) eval_metrics_T<float>(ctx,W,P,P.A32.p,X,M); else eval_metrics_T<double>(ctx,W,P,P.A64.p,X,M); }
// double-double residual max-norm per RHS (audit / reference use)
static void resid_dd_norms(Workspace& W,const Problem& P,const double* X,std::vector<double>& rinf_dd){ int n=P.n,nrhs=P.nrhs; if(P.fp32) k_resid_dd<float><<<grid1(n),256>>>(P.A32.p,n,P.lda,X,P.B.p,W.R.p,W.T.p,nrhs); else k_resid_dd<double><<<grid1(n),256>>>(P.A64.p,n,P.lda,X,P.B.p,W.R.p,W.T.p,nrhs); CUDA_CK(cudaGetLastError());
  std::vector<double> hi((size_t)n*nrhs),lo((size_t)n*nrhs); CUDA_CK(cudaMemcpy(hi.data(),W.R.p,hi.size()*8,cudaMemcpyDeviceToHost)); CUDA_CK(cudaMemcpy(lo.data(),W.T.p,lo.size()*8,cudaMemcpyDeviceToHost)); rinf_dd.assign(nrhs,0); for(int c=0;c<nrhs;c++){ double m=0; for(int i=0;i<n;i++) m=std::max(m,std::fabs(hi[i+(size_t)c*n]+lo[i+(size_t)c*n])); rinf_dd[c]=m; } }

// ============================================================ refinement engine
struct MethodSpec { std::string id; std::string kind="ir"; // direct | ir | irs | magma
  std::string factor="cs32"; int working=64; int resid=64; bool apply64=false; int scale_mode=0; int nb=256;
  bool gmres=false; bool gmres_left=false; int gmres_m=40; double gmres_tol=1e-4; int gmres_budget=480; int gmres_cycles=4;
  int max_outer=30; std::string policy="none"; int budget=0; double cost_ratio_next=3.0; // policy: none|escalate|budget|cost_aware
  cusolverPrecType_t irs_main=CUSOLVER_R_64F, irs_low=CUSOLVER_R_16F; cusolverIRSRefinement_t irs_ref=CUSOLVER_IRS_REFINE_CLASSICAL; bool irs_fallback=true; bool irs_settol=false; double irs_tol=0; int irs_maxit=50; int irs_maxit_inner=50;
  std::string magma_mode; };
struct HistRec { int outer; int stage; double t; double eta_max,omega_max,rinf_max; int inner_in_step; std::string note; };
struct InnerRec { int outer; int cycle; int j; double rec_rel; double true_rel; std::string note; int rhs; };
struct TrialOut { std::string terminal="unknown",final_stage; bool fallback=false; int refactorizations=0; std::vector<std::string> stages; std::vector<double> stage_times;
  int outer=0,inner_total=0,matvecs=0,precond_applies=0,trs=0,restarts=0,true_resid_checks=0; int inner_max_rhs=0; std::vector<int> inner_per_rhs;
  double t_total=0,t_factor_total=0,t_initial=0,t_refine=0,t_eval=0,t_convert=0; std::vector<HistRec> hist; std::vector<InnerRec> ihist; int irs_niters=-9999,irs_outer=-9999,irs_info=-9999; size_t irs_work=0; int magma_iter=-9999,magma_info=-9999; double magma_facto_time=-1;
  Factors* F=nullptr; double growth=-1; unsigned long long conv_overflow=0,conv_subnormal=0; int factor_info=0; std::string error; bool ok=true; double final_eta=0,final_omega=0; };

struct Engine { Ctx& ctx; Workspace& W; const Problem& P; Factors fact; bool reuse=false; DBuf<double> V,Z,x64,Hd,yd,wv,rcur,sR,sT; DBuf<float> x32; std::vector<double> H,gg,cs,sn; int n,nrhs; double tau_eta,tau_omega; bool instrument=false;
  Engine(Ctx& c,Workspace& w,const Problem& p,double te,double to):ctx(c),W(w),P(p),n(p.n),nrhs(p.nrhs),tau_eta(te),tau_omega(to){ x64.alloc((size_t)n*nrhs,"x64"); x32.alloc((size_t)n*nrhs,"x32"); }
  Factors& factors_for(const MethodSpec& M,const std::string& kind,TrialOut& T,bool timed){ Factors& F=fact; double t0=wall_now();
    if(reuse && F.valid && F.kind==kind && F.scale_mode==M.scale_mode && F.nb==M.nb && F.promoted==(M.apply64 && kind!="cs64")){ T.stages.push_back(kind+"(reused)"); T.stage_times.push_back(0.0); T.growth=F.growth_proxy; T.factor_info=F.info; return F; } factorize(ctx,W,P,kind,M.nb,M.scale_mode,M.apply64 && kind!="cs64",F); double dt=wall_now()-t0; T.t_factor_total+=dt; T.stages.push_back(kind); T.stage_times.push_back(dt); T.refactorizations++; T.growth=F.growth_proxy; T.conv_overflow=F.conv_overflow; T.conv_subnormal=F.conv_subnormal; T.factor_info=F.info; return F; }
  void sync_x64(const MethodSpec& M){ if(M.working==32) dev_convert<float,double>(x32.p,x64.p,(size_t)n*nrhs); }
  void add_update(const MethodSpec& M,const double* d){ size_t m=(size_t)n*nrhs; if(M.working==32) k_add<float><<<grid1(m),256>>>(x32.p,d,m,1.0); else k_add<double><<<grid1(m),256>>>(x64.p,d,m,1.0); }
  // residual + stopping metrics in declared precision. returns eta_max/omega_max (omega=-1 if unavailable)
  void residual_and_metrics(const MethodSpec& M,Metrics& Mt){ sync_x64(M); if(M.resid==64){ eval_metrics(ctx,W,P,x64.p,Mt); }
    else { // single-precision residual control: r in fp32 arithmetic; eta from r32; omega not evaluated in-loop
      if(!P.fp32) throw MPError("fp32 residual control requires fp32 stored data"); k_resid_f32<<<grid1(n),256>>>(P.A32.p,n,P.lda,x32.p,P.B32.p,W.tmp32.p,nrhs); dev_convert<float,double>(W.tmp32.p,W.R.p,(size_t)n*nrhs); k_fill<double><<<grid1((size_t)n*nrhs),256>>>(W.T.p,(size_t)n*nrhs,1.0);
      k_colstats<<<nrhs,256>>>(W.R.p,x64.p,W.T.p,n,nrhs,W.stats.p,W.worst.p); std::vector<double> st(4*nrhs); CUDA_CK(cudaMemcpy(st.data(),W.stats.p,32*nrhs,cudaMemcpyDeviceToHost)); Mt.eta.assign(nrhs,0); Mt.omega.assign(nrhs,-1); Mt.rinf.assign(nrhs,0); Mt.finite=true; Mt.eta_max=0; Mt.omega_max=-1;
      for(int c=0;c<nrhs;c++){ double ri=st[4*c],xi=st[4*c+1]; double den=P.anorm_inf*xi+P.bnorm_inf[c]; Mt.eta[c]=(ri==0)?0:(den==0?INFINITY:ri/den); Mt.rinf[c]=ri; if(!std::isfinite(ri)||!std::isfinite(xi)) Mt.finite=false; Mt.eta_max=std::max(Mt.eta_max,Mt.eta[c]); } } }
  // FGMRES (right, flexible) or left-preconditioned GMRES on A d = r for one RHS column. r,d device FP64 (n). Returns status string.
  std::string gmres_solve(const MethodSpec& M,const Factors& F,const double* r,double* d,int& budget_left,TrialOut& T,int outer,int rhs){
    int m=M.gmres_m; if(V.n<(size_t)n*(m+1)){ V.alloc((size_t)n*(m+1),"V"); Z.alloc((size_t)n*m,"Z"); Hd.alloc(m+2,"Hd"); yd.alloc(m+1,"yd"); wv.alloc(n,"w"); rcur.alloc(n,"rcur"); sR.alloc(n,"sR"); sT.alloc(n,"sT"); } H.assign((size_t)(m+1)*m,0.0); gg.assign(m+1,0.0); cs.assign(m,0.0); sn.assign(m,0.0);
    CUBLAS_CK(cublasSetPointerMode(ctx.cb,CUBLAS_POINTER_MODE_DEVICE)); k_fill<double><<<grid1(n),256>>>(d,n,0.0);
    // left variant works on preconditioned residual g = F^{-1} r
    const double* rr=r; if(M.gmres_left){ apply_inverse(ctx,W,F,r,rcur.p,1,M.apply64,&T.trs); T.precond_applies++; rr=rcur.p; } else CUDA_CK(cudaMemcpyAsync(rcur.p,r,n*8,cudaMemcpyDeviceToDevice));
    double beta0; CUBLAS_CK(cublasDnrm2(ctx.cb,n,rr,1,Hd.p+m+1)); CUDA_CK(cudaMemcpy(&beta0,Hd.p+m+1,8,cudaMemcpyDeviceToHost)); if(beta0==0.0||!std::isfinite(beta0)){ CUBLAS_CK(cublasSetPointerMode(ctx.cb,CUBLAS_POINTER_MODE_HOST)); return beta0==0.0?"zero_rhs":"nonfinite"; }
    std::string status="budget"; int cycles=0;
    while(cycles<M.gmres_cycles && budget_left>0){ cycles++; if(cycles>1) T.restarts++;
      double betac; CUBLAS_CK(cublasDnrm2(ctx.cb,n,rcur.p,1,Hd.p+m+1)); CUDA_CK(cudaMemcpy(&betac,Hd.p+m+1,8,cudaMemcpyDeviceToHost)); if(betac==0){ status="converged_true"; break; }
      CUDA_CK(cudaMemcpyAsync(V.p,rcur.p,n*8,cudaMemcpyDeviceToDevice)); double ib=1.0/betac; CUBLAS_CK(cublasSetPointerMode(ctx.cb,CUBLAS_POINTER_MODE_HOST)); CUBLAS_CK(cublasDscal(ctx.cb,n,&ib,V.p,1)); CUBLAS_CK(cublasSetPointerMode(ctx.cb,CUBLAS_POINTER_MODE_DEVICE));
      std::fill(gg.begin(),gg.end(),0.0); gg[0]=betac; std::fill(H.begin(),H.end(),0.0); int j=0; bool happy=false,nonfinite=false;
      for(j=0;j<m && budget_left>0;j++){ double* vj=V.p+(size_t)j*n; double* w=wv.p;
        if(M.gmres_left){ // w = F^{-1} (A v_j)
          if(P.fp32) k_resid<float,1><<<grid1(n),256>>>(P.A32.p,n,P.lda,vj,nullptr,sR.p,sT.p,0); else k_resid<double,1><<<grid1(n),256>>>(P.A64.p,n,P.lda,vj,nullptr,sR.p,sT.p,0); T.matvecs++; k_scale<double><<<grid1(n),256>>>(sR.p,n,-1.0);
          apply_inverse(ctx,W,F,sR.p,w,1,M.apply64,&T.trs); T.precond_applies++; }
        else { double* zj=Z.p+(size_t)j*n; apply_inverse(ctx,W,F,vj,zj,1,M.apply64,&T.trs); T.precond_applies++;
          if(P.fp32) k_resid<float,1><<<grid1(n),256>>>(P.A32.p,n,P.lda,zj,nullptr,sR.p,sT.p,0); else k_resid<double,1><<<grid1(n),256>>>(P.A64.p,n,P.lda,zj,nullptr,sR.p,sT.p,0); T.matvecs++; CUDA_CK(cudaMemcpyAsync(w,sR.p,n*8,cudaMemcpyDeviceToDevice)); k_scale<double><<<grid1(n),256>>>(w,n,-1.0); }
        // modified Gram-Schmidt (one pass) with device-resident coefficients
        for(int i=0;i<=j;i++){ CUBLAS_CK(cublasDdot(ctx.cb,n,V.p+(size_t)i*n,1,w,1,Hd.p+i)); k_axpy_dev<<<grid1(n),256>>>(w,V.p+(size_t)i*n,Hd.p+i,n); }
        CUBLAS_CK(cublasDnrm2(ctx.cb,n,w,1,Hd.p+j+1)); std::vector<double> hc(j+2); CUDA_CK(cudaMemcpy(hc.data(),Hd.p,8*(j+2),cudaMemcpyDeviceToHost));
        for(int i=0;i<=j+1;i++) H[i+(size_t)j*(m+1)]=hc[i]; double hn=hc[j+1]; if(!std::isfinite(hn)){ nonfinite=true; break; }
        if(hn>0){ double ih=1.0/hn; CUBLAS_CK(cublasSetPointerMode(ctx.cb,CUBLAS_POINTER_MODE_HOST)); CUBLAS_CK(cublasDscal(ctx.cb,n,&ih,w,1)); CUBLAS_CK(cublasSetPointerMode(ctx.cb,CUBLAS_POINTER_MODE_DEVICE)); CUDA_CK(cudaMemcpyAsync(V.p+(size_t)(j+1)*n,w,n*8,cudaMemcpyDeviceToDevice)); }
        // Givens
        for(int i=0;i<j;i++){ double a=H[i+(size_t)j*(m+1)],b=H[i+1+(size_t)j*(m+1)]; H[i+(size_t)j*(m+1)]=cs[i]*a+sn[i]*b; H[i+1+(size_t)j*(m+1)]=-sn[i]*a+cs[i]*b; }
        double a=H[j+(size_t)j*(m+1)],b=H[j+1+(size_t)j*(m+1)],rho=std::hypot(a,b); if(rho==0){ cs[j]=1; sn[j]=0; } else { cs[j]=a/rho; sn[j]=b/rho; } H[j+(size_t)j*(m+1)]=rho; H[j+1+(size_t)j*(m+1)]=0; double g0=gg[j]; gg[j]=cs[j]*g0; gg[j+1]=-sn[j]*g0;
        T.inner_total++; budget_left--; double rec=std::fabs(gg[j+1])/beta0; if(instrument) T.ihist.push_back({outer,cycles,j,rec,-1,"",rhs});
        if(hn==0.0){ happy=true; j++; break; } if(rec<=M.gmres_tol){ j++; break; } }
      if(nonfinite){ status="nonfinite"; break; }
      int k=j; if(k==0) break; // solve upper-triangular H y = g
      std::vector<double> y(k); for(int i=k-1;i>=0;i--){ double s=gg[i]; for(int l=i+1;l<k;l++) s-=H[i+(size_t)l*(m+1)]*y[l]; y[i]=s/H[i+(size_t)i*(m+1)]; }
      CUDA_CK(cudaMemcpy(yd.p,y.data(),8*k,cudaMemcpyHostToDevice)); double one=1.0; CUBLAS_CK(cublasSetPointerMode(ctx.cb,CUBLAS_POINTER_MODE_HOST));
      CUBLAS_CK(cublasDgemv(ctx.cb,CUBLAS_OP_N,n,k,&one,M.gmres_left? V.p : Z.p,n,yd.p,1,&one,d,1)); CUBLAS_CK(cublasSetPointerMode(ctx.cb,CUBLAS_POINTER_MODE_DEVICE));
      // true residual of the correction equation in the declared check precision (FP64): rcur = r - A d (original) ; for left variant also preconditioned
      if(P.fp32) k_resid<float,1><<<grid1(n),256>>>(P.A32.p,n,P.lda,d,r,sR.p,sT.p,0); else k_resid<double,1><<<grid1(n),256>>>(P.A64.p,n,P.lda,d,r,sR.p,sT.p,0); T.matvecs++; T.true_resid_checks++;
      double tn; CUBLAS_CK(cublasDnrm2(ctx.cb,n,sR.p,1,Hd.p+m+1)); CUDA_CK(cudaMemcpy(&tn,Hd.p+m+1,8,cudaMemcpyDeviceToHost)); double rn0; { CUBLAS_CK(cublasDnrm2(ctx.cb,n,r,1,Hd.p+m+1)); CUDA_CK(cudaMemcpy(&rn0,Hd.p+m+1,8,cudaMemcpyDeviceToHost)); }
      double true_rel=(rn0>0)? tn/rn0 : 0.0; double rec=std::fabs(gg[k])/beta0; if(instrument) T.ihist.push_back({outer,cycles,k-1,rec,true_rel,"cycle_end",rhs});
      if(M.gmres_left){ apply_inverse(ctx,W,F,sR.p,rcur.p,1,M.apply64,&T.trs); T.precond_applies++; } else CUDA_CK(cudaMemcpyAsync(rcur.p,sR.p,n*8,cudaMemcpyDeviceToDevice));
      if(!std::isfinite(tn)){ status="nonfinite"; break; }
      if(true_rel<=M.gmres_tol){ status=happy?"happy_breakdown_verified":"converged_true"; break; }
      if(happy){ status="happy_breakdown_unverified"; break; }
      if(k<m && budget_left>0 && rec<=M.gmres_tol){ status="recursive_true_gap"; /* recursive said converged but true residual did not: continue restarting */ } }
    CUBLAS_CK(cublasSetPointerMode(ctx.cb,CUBLAS_POINTER_MODE_HOST)); return status; }

  // run one refinement stage with factors F; returns terminal reason for the stage.
  std::string refine_stage(const MethodSpec& M,const Factors& F,int stage_idx,TrialOut& T,double t_start,int& budget_left,bool initial_solve,Metrics& Mt){
    size_t mm=(size_t)n*nrhs;
    if(initial_solve){ double t0=wall_now(); apply_inverse(ctx,W,F,P.B.p,W.D.p,nrhs,M.apply64,&T.trs); if(M.working==32) dev_convert<double,float>(W.D.p,x32.p,mm); else CUDA_CK(cudaMemcpyAsync(x64.p,W.D.p,mm*8,cudaMemcpyDeviceToDevice)); CUDA_CK(cudaDeviceSynchronize()); T.t_initial+=wall_now()-t0; }
    double prev_eta=INFINITY; int stall=0; std::string reason="max_outer"; int local_outer=0;
    while(true){ double te0=wall_now(); residual_and_metrics(M,Mt); CUDA_CK(cudaDeviceSynchronize()); T.t_eval+=wall_now()-te0; T.final_eta=Mt.eta_max; T.final_omega=Mt.omega_max;
      int inner_before=T.inner_total; if(instrument||true) T.hist.push_back({T.outer,stage_idx,wall_now()-t_start,Mt.eta_max,Mt.omega_max,*std::max_element(Mt.rinf.begin(),Mt.rinf.end()),0,""});
      bool conv=Mt.finite && Mt.eta_max<=tau_eta && (M.resid!=64 || Mt.omega_max<=tau_omega);
      if(conv){ reason="converged"; break; } if(!Mt.finite){ reason="nonfinite"; break; }
      if(T.outer>=M.max_outer){ reason="max_outer"; break; }
      if(Mt.eta_max>0.9*prev_eta) stall++; else stall=0; if(stall>=3){ reason="stagnation"; break; } prev_eta=Mt.eta_max;
      if(M.gmres && budget_left<=0){ reason="gmres_budget_exhausted"; break; }
      // correction
      if(M.gmres){ for(int c=0;c<nrhs;c++){ std::string st=gmres_solve(M,F,W.R.p+(size_t)c*n,W.D.p+(size_t)c*n,budget_left,T,T.outer,c); if(T.inner_per_rhs.size()<(size_t)nrhs) T.inner_per_rhs.assign(nrhs,0); T.hist.back().note+=(c?";":"")+st; if(st=="nonfinite"){ T.hist.back().note+=";inner_nonfinite"; } } }
      else { apply_inverse(ctx,W,F,W.R.p,W.D.p,nrhs,M.apply64,&T.trs); T.precond_applies++; }
      T.hist.back().inner_in_step=T.inner_total-inner_before; add_update(M,W.D.p); T.outer++; local_outer++; }
    return reason; }

  // full method: policy over stages
  void run(const MethodSpec& M,TrialOut& T,bool timed){
    double t0=wall_now(); T.outer=0; T.inner_total=0; int budget_left=M.gmres? M.gmres_budget : 0; std::vector<std::string> ladder; ladder.push_back(M.factor);
    bool target64=(tau_eta<1e-10);
    if(M.policy!="none"){ if(M.factor=="het16"||M.factor=="cus32"){ ladder.push_back("cs32"); if(target64) ladder.push_back("cs64"); } else if(M.factor=="cs32"){ if(target64) ladder.push_back("cs64"); } }
    Metrics Mt; std::string reason="unknown"; int stage=0; MethodSpec Ms=M;
    for(stage=0;stage<(int)ladder.size();stage++){ Factors& F=factors_for(M,ladder[stage],T,timed); T.final_stage=ladder[stage]; if(stage>0) T.fallback=true;
      if(F.info>0){ reason="singular_factor"; T.hist.push_back({T.outer,stage,wall_now()-t0,-1,-1,-1,0,"singular_factor_info="+std::to_string(F.info)}); if(M.policy=="none") break; else continue; }
      if(M.policy=="budget" && M.budget==0) Ms.gmres=false; // budget 0 => bypass Krylov entirely
      if(M.policy=="budget" && M.budget>0){ if(stage==0){ Ms.gmres=true; budget_left=M.budget; } else { Ms.gmres=false; } } // Krylov recovery only with the low-precision factors; escalated stages use stationary IR
      if(M.policy=="cost_aware"){ Ms.gmres=false; }
      double ts=wall_now(); reason=refine_stage(Ms,F,stage,T,t0,budget_left,(stage==0),Mt); T.t_refine+=wall_now()-ts;
      if(M.policy=="cost_aware" && reason!="converged" && reason!="nonfinite" && M.gmres_budget>0 && stage==0){ // heuristic: try Krylov recovery when its predicted cost is below the next refactorization cost
        double c_outer=T.hist.size()>=2? (T.hist.back().t-T.hist[T.hist.size()-2].t):0.0; double c_refactor_next=T.stage_times.back()*M.cost_ratio_next; double predicted_gmres=c_outer*(double)std::min(M.gmres_budget,M.gmres_m)*2.0; T.hist.back().note+=";cost_aware:c_outer="+std::to_string(c_outer)+",c_refactor_next="+std::to_string(c_refactor_next)+",pred_gmres="+std::to_string(predicted_gmres);
        if(predicted_gmres<c_refactor_next){ Ms.gmres=true; budget_left=M.gmres_budget; T.hist.back().note+=";switch_to_gmres"; T.outer=std::max(0,T.outer-0); reason=refine_stage(Ms,F,stage,T,t0,budget_left,false,Mt); Ms.gmres=false; } }
      if(reason=="converged") break; if(M.policy=="none") break; if(stage+1<(int)ladder.size()) T.hist.push_back({T.outer,stage,wall_now()-t0,Mt.eta_max,Mt.omega_max,-1,0,"escalate_after:"+reason}); }
    T.terminal=(M.kind=="direct" && reason!="nonfinite" && reason!="singular_factor")? "direct_solve" : reason; CUDA_CK(cudaDeviceSynchronize()); T.t_total=wall_now()-t0; }
};

// cuSOLVER IRS wrapper: A (main precision) copied into LU buffer each call (restoration), X output in main precision.
struct IRSRunner { Ctx& ctx; const Problem& P; DBuf<char> work; size_t lwork=0; DBuf<double> A64work,X64,B64w; DBuf<float> A32work,X32,B32w; DBuf<int> dinfo; std::string key;
  IRSRunner(Ctx& c,const Problem& p):ctx(c),P(p){ dinfo.alloc(1,"irs_info"); }
  void prepare(const MethodSpec& M){ cusolverDnIRSParams_t ip; CUSOLVER_CK(cusolverDnIRSParamsCreate(&ip)); CUSOLVER_CK(cusolverDnIRSParamsSetSolverPrecisions(ip,M.irs_main,M.irs_low)); CUSOLVER_CK(cusolverDnIRSParamsSetRefinementSolver(ip,M.irs_ref)); size_t lw=0; CUSOLVER_CK(cusolverDnIRSXgesv_bufferSize(ctx.cs,ip,P.n,P.nrhs,&lw)); cusolverDnIRSParamsDestroy(ip); if(lw>lwork){ work.alloc(lw+256,"irs_work"); lwork=lw; }
    size_t nn=(size_t)P.n*P.n; if(M.irs_main==CUSOLVER_R_64F){ if(A64work.n<nn) A64work.alloc(nn,"irs_A64"); if(X64.n<(size_t)P.n*P.nrhs){ X64.alloc((size_t)P.n*P.nrhs,"irs_X64"); B64w.alloc((size_t)P.n*P.nrhs,"irs_B64"); } } else { if(A32work.n<nn) A32work.alloc(nn,"irs_A32"); if(X32.n<(size_t)P.n*P.nrhs){ X32.alloc((size_t)P.n*P.nrhs,"irs_X32"); B32w.alloc((size_t)P.n*P.nrhs,"irs_B32"); } } }
  void run(const MethodSpec& M,TrialOut& T){ double t0=wall_now(); size_t nn=(size_t)P.n*P.n; cusolverDnIRSParams_t ip; cusolverDnIRSInfos_t inf; CUSOLVER_CK(cusolverDnIRSParamsCreate(&ip)); CUSOLVER_CK(cusolverDnIRSInfosCreate(&inf));
    CUSOLVER_CK(cusolverDnIRSParamsSetSolverPrecisions(ip,M.irs_main,M.irs_low)); CUSOLVER_CK(cusolverDnIRSParamsSetRefinementSolver(ip,M.irs_ref)); if(M.irs_fallback) CUSOLVER_CK(cusolverDnIRSParamsEnableFallback(ip)); else CUSOLVER_CK(cusolverDnIRSParamsDisableFallback(ip));
    CUSOLVER_CK(cusolverDnIRSParamsSetMaxIters(ip,M.irs_maxit)); CUSOLVER_CK(cusolverDnIRSParamsSetMaxItersInner(ip,M.irs_maxit_inner)); if(M.irs_settol){ CUSOLVER_CK(cusolverDnIRSParamsSetTol(ip,M.irs_tol)); CUSOLVER_CK(cusolverDnIRSParamsSetTolInner(ip,M.irs_tol)); }
    int niters=-9999; cusolverStatus_t st;
    if(M.irs_main==CUSOLVER_R_64F){ if(P.fp32) dev_convert<float,double>(P.A32.p,A64work.p,nn); else CUDA_CK(cudaMemcpyAsync(A64work.p,P.A64.p,nn*8,cudaMemcpyDeviceToDevice)); CUDA_CK(cudaMemcpyAsync(B64w.p,P.B.p,(size_t)P.n*P.nrhs*8,cudaMemcpyDeviceToDevice)); st=cusolverDnIRSXgesv(ctx.cs,ip,inf,P.n,P.nrhs,A64work.p,P.n,B64w.p,P.n,X64.p,P.n,work.p,lwork,&niters,dinfo.p); }
    else { if(!P.fp32) throw MPError("IRS main fp32 requires fp32 stored data"); CUDA_CK(cudaMemcpyAsync(A32work.p,P.A32.p,nn*4,cudaMemcpyDeviceToDevice)); CUDA_CK(cudaMemcpyAsync(B32w.p,P.B32.p,(size_t)P.n*P.nrhs*4,cudaMemcpyDeviceToDevice)); st=cusolverDnIRSXgesv(ctx.cs,ip,inf,P.n,P.nrhs,A32work.p,P.n,B32w.p,P.n,X32.p,P.n,work.p,lwork,&niters,dinfo.p); }
    cudaError_t ce=cudaDeviceSynchronize(); T.t_total=wall_now()-t0; int outer=-9999; cusolverDnIRSInfosGetOuterNiters(inf,&outer); int info=-9999; cudaMemcpy(&info,dinfo.p,4,cudaMemcpyDeviceToHost); T.irs_niters=niters; T.irs_outer=outer; T.irs_info=info; T.irs_work=lwork; T.outer=outer; T.inner_total=niters;
    cusolverDnIRSParamsDestroy(ip); cusolverDnIRSInfosDestroy(inf); if(st!=CUSOLVER_STATUS_SUCCESS||ce!=cudaSuccess){ T.ok=false; T.error="cusolverDnIRSXgesv status "+std::to_string((int)st)+" cuda "+std::to_string((int)ce); T.terminal="runtime_error"; cudaGetLastError(); return; }
    // niters semantics (CUDA 12.6 docs): >0 converged iterations; <0 did not converge (fallback to main-precision solve when enabled)
    T.terminal=(niters>=0)?"vendor_converged":"vendor_not_converged"; T.fallback=(niters<0); T.final_stage=(niters<0 && M.irs_fallback)?"vendor_fallback_main_precision":"vendor_mixed"; }
};
