// mpsolve.cu — campaign driver CLI: problem generation, methods, timed/diagnostic trials, references, audits.
// Build: see build/build.sh. Original work for the CPSC 542G mixed-precision LU / iterative refinement project.
#include "mp_core.cuh"
#include <sys/stat.h>
#include <unistd.h>

// ------------------------------------------------------------ args
struct Args { std::map<std::string,std::string> kv; std::string cmd;
  std::string get(const std::string& k,const std::string& d="") const { auto it=kv.find(k); return it==kv.end()? d : it->second; }
  double getd(const std::string& k,double d) const { auto it=kv.find(k); return it==kv.end()? d : std::stod(it->second); }
  int geti(const std::string& k,int d) const { auto it=kv.find(k); return it==kv.end()? d : std::stoi(it->second); }
  bool has(const std::string& k) const { return kv.count(k)>0; } };
static Args parse_args(int argc,char** argv){ Args a; if(argc>1) a.cmd=argv[1]; for(int i=2;i<argc;i++){ std::string s=argv[i]; if(s.rfind("--",0)==0){ std::string k=s.substr(2); if(i+1<argc && std::string(argv[i+1]).rfind("--",0)!=0){ a.kv[k]=argv[++i]; } else a.kv[k]="1"; } } return a; }
static std::vector<std::string> split(const std::string& s,char d){ std::vector<std::string> r; std::stringstream ss(s); std::string t; while(std::getline(ss,t,d)) if(!t.empty()) r.push_back(t); return r; }

// ------------------------------------------------------------ method registry (IDs are immutable semantics; parameters frozen via CLI/config)
static MethodSpec make_method(const std::string& id,const Args& a){
  MethodSpec M; M.id=id; M.nb=a.geti("nb",256); M.gmres_m=a.geti("gmres-m",40); M.gmres_tol=a.getd("gmres-tol",1e-4); M.gmres_budget=a.geti("gmres-budget",480); M.gmres_cycles=a.geti("gmres-cycles",4); M.max_outer=a.geti("max-outer",30); M.cost_ratio_next=a.getd("cost-ratio-next",3.0);
  auto starts=[&](const char* p){ return id.rfind(p,0)==0; };
  // ---- native direct solves
  if(id=="native_lu32"){ M.kind="direct"; M.factor="cs32"; M.working=32; M.resid=64; M.max_outer=0; }
  else if(id=="native_lu64"||id=="native_lu64_promoted32"){ M.kind="direct"; M.factor="cs64"; M.working=64; M.max_outer=0; }
  // ---- native stationary refinement (cuSOLVER factors, custom outer loop)
  else if(id=="native_ir32_w32_r64"){ M.factor="cs32"; M.working=32; M.resid=64; }
  else if(id=="native_ir32_w32_r32"){ M.factor="cs32"; M.working=32; M.resid=32; }
  else if(id=="native_ir32_w64_r64"){ M.factor="cs32"; M.working=64; M.resid=64; }
  else if(id=="native_ir64_w64_r64"){ M.factor="cs64"; M.working=64; M.resid=64; }
  else if(id=="native_ir32_w64_r64_twosided"){ M.factor="cs32"; M.working=64; M.resid=64; M.scale_mode=2; }
  else if(id=="native_ir32_w64_r64_global"){ M.factor="cs32"; M.working=64; M.resid=64; M.scale_mode=1; }
  else if(id=="native_gmresir32_w64_r64"){ M.factor="cs32"; M.working=64; M.resid=64; M.gmres=true; }
  else if(id=="native_gmresir32_w32_r64"){ M.factor="cs32"; M.working=32; M.resid=64; M.gmres=true; }
  else if(id=="native_leftgmres32_app32_w64_r64"){ M.factor="cs32"; M.working=64; M.resid=64; M.gmres=true; M.gmres_left=true; M.gmres_cycles=1; }
  else if(id=="native_leftgmres32_app64_w64_r64"){ M.factor="cs32"; M.working=64; M.resid=64; M.gmres=true; M.gmres_left=true; M.apply64=true; M.gmres_cycles=1; }
  else if(id=="native_gmresir32_app64_w64_r64"){ M.factor="cs32"; M.working=64; M.resid=64; M.gmres=true; M.apply64=true; }
  // ---- heterogeneous custom LU (FP32 panels/storage/trsm, FP16-operand tensor-core trailing update, FP32 accumulate), global pow2 scaling
  else if(id=="hetero16_lu_direct"){ M.kind="direct"; M.factor="het16"; M.working=32; M.scale_mode=1; M.max_outer=0; }
  else if(id=="hetero16_ir_w64_r64"){ M.factor="het16"; M.working=64; M.resid=64; M.scale_mode=1; }
  else if(id=="hetero16_ir_w32_r64"){ M.factor="het16"; M.working=32; M.resid=64; M.scale_mode=1; }
  else if(id=="hetero16_ir_w64_r64_noscale"){ M.factor="het16"; M.working=64; M.resid=64; M.scale_mode=0; }
  else if(id=="hetero16_ir_w64_r64_twosided"){ M.factor="het16"; M.working=64; M.resid=64; M.scale_mode=2; }
  else if(id=="hetero16_gmresir_w64_r64"){ M.factor="het16"; M.working=64; M.resid=64; M.scale_mode=1; M.gmres=true; }
  else if(id=="hetero16_gmresir_w32_r64"){ M.factor="het16"; M.working=32; M.resid=64; M.scale_mode=1; M.gmres=true; }
  else if(id=="hetero16_gmresir_app64_w64_r64"){ M.factor="het16"; M.working=64; M.resid=64; M.scale_mode=1; M.gmres=true; M.apply64=true; }
  else if(id=="hetero16_leftgmres_app64_w64_r64"){ M.factor="het16"; M.working=64; M.resid=64; M.scale_mode=1; M.gmres=true; M.gmres_left=true; M.apply64=true; M.gmres_cycles=1; }
  // ---- custom FP32 blocked control (same code path, FP32 GEMM) for the promotion gate
  else if(id=="custom32_lu_direct"){ M.kind="direct"; M.factor="cus32"; M.working=32; M.scale_mode=1; M.max_outer=0; }
  else if(id=="custom32_ir_w64_r64"){ M.factor="cus32"; M.working=64; M.resid=64; M.scale_mode=1; }
  else if(id=="custom32_ir_w32_r64"){ M.factor="cus32"; M.working=32; M.resid=64; M.scale_mode=1; }
  // ---- recovery policies (E6): fallback ladder het16 -> cs32 -> cs64 (target64) / het16 -> cs32 (target32)
  else if(id=="policy_hetero16_stationary_only"){ M.factor="het16"; M.working=64; M.resid=64; M.scale_mode=1; M.policy="none"; }
  else if(id=="policy_hetero16_gmres_only"){ M.factor="het16"; M.working=64; M.resid=64; M.scale_mode=1; M.gmres=true; M.policy="none"; }
  else if(id=="policy_hetero16_escalate_immediate"){ M.factor="het16"; M.working=64; M.resid=64; M.scale_mode=1; M.policy="budget"; M.budget=0; }
  else if(starts("policy_hetero16_budget")){ M.factor="het16"; M.working=64; M.resid=64; M.scale_mode=1; M.policy="budget"; M.budget=std::stoi(id.substr(std::string("policy_hetero16_budget").size())); M.gmres=true; M.gmres_budget=M.budget; }
  else if(id=="policy_hetero16_cost_aware"){ M.factor="het16"; M.working=64; M.resid=64; M.scale_mode=1; M.policy="cost_aware"; M.gmres_budget=a.geti("gmres-budget",120); }
  else if(id=="policy_native32_escalate_immediate"){ M.factor="cs32"; M.working=64; M.resid=64; M.policy="budget"; M.budget=0; }
  else if(starts("policy_native32_budget")){ M.factor="cs32"; M.working=64; M.resid=64; M.policy="budget"; M.budget=std::stoi(id.substr(std::string("policy_native32_budget").size())); M.gmres=true; M.gmres_budget=M.budget; }
  // ---- cuSOLVER IRS (opaque vendor mixed solver); names encode main/lowest dtype + refinement mode + fallback + tolerance source
  else if(starts("cusolver_irs_")){ M.kind="irs"; auto p=split(id,'_'); // cusolver_irs_m64_l16_classical_fb_default
    for(auto& t:p){ if(t=="m64") M.irs_main=CUSOLVER_R_64F; else if(t=="m32") M.irs_main=CUSOLVER_R_32F; else if(t=="l16") M.irs_low=CUSOLVER_R_16F; else if(t=="l32") M.irs_low=CUSOLVER_R_32F; else if(t=="l64") M.irs_low=CUSOLVER_R_64F;
      else if(t=="classical") M.irs_ref=CUSOLVER_IRS_REFINE_CLASSICAL; else if(t=="gmres") M.irs_ref=CUSOLVER_IRS_REFINE_GMRES; else if(t=="cgmres") M.irs_ref=CUSOLVER_IRS_REFINE_CLASSICAL_GMRES; else if(t=="ggmres") M.irs_ref=CUSOLVER_IRS_REFINE_GMRES_GMRES; else if(t=="none") M.irs_ref=CUSOLVER_IRS_REFINE_NONE;
      else if(t=="fb") M.irs_fallback=true; else if(t=="nofb") M.irs_fallback=false; else if(t=="default") M.irs_settol=false; else if(t=="tol") { M.irs_settol=true; M.irs_tol=a.getd("irs-tol",1e-14);} }
    M.working=(M.irs_main==CUSOLVER_R_64F)?64:32; }
#ifdef HAVE_MAGMA
  else if(starts("magma_")){ M.kind="magma"; M.magma_mode=id.substr(6); M.working=64; if(M.magma_mode=="lu32_native"){ M.working=32; }
    if(M.magma_mode=="ht_ir_w64_r64"){ M.kind="ir"; M.factor="mg_ht"; M.working=64; M.resid=64; M.scale_mode=1; }
    else if(M.magma_mode=="ht_gmresir_w64_r64"){ M.kind="ir"; M.factor="mg_ht"; M.working=64; M.resid=64; M.scale_mode=1; M.gmres=true; }
    else if(M.magma_mode=="s32native_ir_w64_r64"){ M.kind="ir"; M.factor="mg_s"; M.working=64; M.resid=64; }
    else if(M.magma_mode=="d64native_direct"){ M.kind="direct"; M.factor="mg_d"; M.working=64; M.max_outer=0; }
    else if(M.magma_mode=="s32native_direct"){ M.kind="direct"; M.factor="mg_s"; M.working=32; M.max_outer=0; } }
#endif
  else throw MPError("unknown method id "+id);
  return M; }

#ifdef HAVE_MAGMA
// MAGMA native and mixed drivers. All MAGMA calls use MAGMA's own queues; cudaDeviceSynchronize bounds the timing.
struct MagmaRunner { Ctx& ctx; const Problem& P; DBuf<double> A64w,X64,dworkd,B64w; DBuf<float> A32w,X32,dworks,B32w; DBuf<int> dipiv; std::vector<magma_int_t> ipiv; bool prepared=false;
  MagmaRunner(Ctx& c,const Problem& p):ctx(c),P(p){}
  void prepare(const MethodSpec& M){ size_t nn=(size_t)P.n*P.n; if(!prepared){ ipiv.resize(P.n); dipiv.alloc(P.n,"magma_dipiv"); prepared=true; }
    if(M.magma_mode=="lu32_native"){ if(!A32w.p){ A32w.alloc(nn,"magma_A32"); X32.alloc((size_t)P.n*P.nrhs,"magma_X32"); } }
    else { if(!A64w.p){ A64w.alloc(nn,"magma_A64"); X64.alloc((size_t)P.n*P.nrhs,"magma_X64"); B64w.alloc((size_t)P.n*P.nrhs,"magma_B64"); } if(M.magma_mode!="lu64_native"&&M.magma_mode!="lu64_hybrid" && !dworks.p){ dworkd.alloc((size_t)P.n*P.nrhs,"magma_dworkd"); dworks.alloc(nn+(size_t)P.n*P.nrhs,"magma_dworks"); } } }
  void run(const MethodSpec& M,TrialOut& T){ double t0=wall_now(); size_t nn=(size_t)P.n*P.n; magma_int_t info=0,iter=0; real_Double_t ft=0; int n=P.n,nrhs=P.nrhs;
    if(M.magma_mode=="lu32_native"){ if(!P.fp32) throw MPError("magma lu32 requires fp32 data"); CUDA_CK(cudaMemcpyAsync(A32w.p,P.A32.p,nn*4,cudaMemcpyDeviceToDevice)); CUDA_CK(cudaMemcpyAsync(X32.p,P.B32.p,(size_t)n*nrhs*4,cudaMemcpyDeviceToDevice)); magma_sgetrf_native(n,n,A32w.p,n,ipiv.data(),&info); if(info==0) magma_sgetrs_gpu(MagmaNoTrans,n,nrhs,A32w.p,n,ipiv.data(),X32.p,n,&info); }
    else { if(P.fp32) dev_convert<float,double>(P.A32.p,A64w.p,nn); else CUDA_CK(cudaMemcpyAsync(A64w.p,P.A64.p,nn*8,cudaMemcpyDeviceToDevice));
      if(M.magma_mode=="lu64_native"){ CUDA_CK(cudaMemcpyAsync(X64.p,P.B.p,(size_t)n*nrhs*8,cudaMemcpyDeviceToDevice)); magma_dgetrf_native(n,n,A64w.p,n,ipiv.data(),&info); if(info==0) magma_dgetrs_gpu(MagmaNoTrans,n,nrhs,A64w.p,n,ipiv.data(),X64.p,n,&info); }
      else if(M.magma_mode=="lu64_hybrid"){ CUDA_CK(cudaMemcpyAsync(X64.p,P.B.p,(size_t)n*nrhs*8,cudaMemcpyDeviceToDevice)); magma_dgetrf_gpu(n,n,A64w.p,n,ipiv.data(),&info); if(info==0) magma_dgetrs_gpu(MagmaNoTrans,n,nrhs,A64w.p,n,ipiv.data(),X64.p,n,&info); }
      else { magma_refinement_t facto=Magma_PREC_SHT, solver=Magma_REFINE_GMSTRS; const std::string& m=M.magma_mode; // modes: dx_<facto>_<solver>
        if(m.find("_ss_")!=std::string::npos) facto=Magma_PREC_SS; else if(m.find("_sht_")!=std::string::npos) facto=Magma_PREC_SHT; else if(m.find("_hs_")!=std::string::npos) facto=Magma_PREC_HS; else throw MPError("unknown magma facto in "+m);
        if(m.size()>=6 && m.substr(m.size()-6)=="irstrs") solver=Magma_REFINE_IRSTRS; else if(m.substr(m.size()-6)=="irdtrs") solver=Magma_REFINE_IRDTRS; else if(m.substr(m.size()-6)=="gmstrs") solver=Magma_REFINE_GMSTRS; else if(m.substr(m.size()-6)=="gmdtrs") solver=Magma_REFINE_GMDTRS; else if(m.size()>=8&&m.substr(m.size()-8)=="irgmstrs") solver=Magma_REFINE_IRGMSTRS; else throw MPError("unknown magma solver in "+m);
        CUDA_CK(cudaMemcpyAsync(B64w.p,P.B.p,(size_t)n*nrhs*8,cudaMemcpyDeviceToDevice)); magma_dxgesv_gmres_gpu(MagmaNoTrans,n,nrhs,A64w.p,n,ipiv.data(),dipiv.p,B64w.p,n,X64.p,n,dworkd.p,dworks.p,facto,solver,&iter,&info,&ft); } }
    cudaError_t ce=cudaDeviceSynchronize(); T.t_total=wall_now()-t0; T.magma_iter=iter; T.magma_info=info; T.magma_facto_time=ft; T.outer=iter; if(ce!=cudaSuccess){ T.ok=false; T.error=std::string("cuda error after magma: ")+cudaGetErrorString(ce); T.terminal="runtime_error"; cudaGetLastError(); return; }
    if(info<0){ T.ok=false; T.error="magma info "+std::to_string(info); T.terminal="runtime_error"; return; } if(info>0){ T.terminal="singular_factor"; T.final_stage="vendor"; return; }
    if(M.magma_mode=="lu64_native"||M.magma_mode=="lu64_hybrid"||M.magma_mode=="lu32_native"){ T.terminal="direct_solve"; T.final_stage="vendor_direct"; }
    else { // MAGMA iter semantics (dsgesv-like): >0 converged iterations in mixed precision; <0 fell back to double precision factorization
      T.terminal=(iter>=0)?"vendor_converged":"vendor_not_converged"; T.fallback=(iter<0); T.final_stage=(iter<0)?"vendor_fallback_fp64":"vendor_mixed"; } }
};
#endif

// ------------------------------------------------------------ reference solution and conditioning estimate (outside timing)
static void compute_reference(Ctx& ctx,Workspace& W,Problem& P,const std::string& mode,int64_t mem_budget){
  int n=P.n,nrhs=P.nrhs; size_t nn=(size_t)n*n; double t0=wall_now(); if(mode=="none"){ P.ref_grade="unavailable"; P.ref_note="reference disabled by configuration"; return; }
  Factors F; bool use64=((int64_t)(nn*8)+(int64_t)(P.fp32?0:0) < mem_budget); std::string kind=use64?"cs64":"cs32";
  try { factorize(ctx,W,P,kind,256,0,false,F); } catch(MPError& e){ P.ref_grade="unavailable"; P.ref_note=std::string("reference factorization failed: ")+e.what(); return; }
  if(F.info>0){ P.ref_grade="unavailable"; P.ref_note="reference LU singular at pivot "+std::to_string(F.info); return; }
  P.Xref.alloc((size_t)n*nrhs,"Xref"); apply_inverse(ctx,W,F,P.B.p,P.Xref.p,nrhs,false,nullptr); CUDA_CK(cudaDeviceSynchronize());
  std::vector<double> rdd; double best=INFINITY; int it=0; DBuf<double> Xbest((size_t)n*nrhs,"Xbest"); DBuf<double> Rhi((size_t)n*nrhs,"Rhi"),Rlo((size_t)n*nrhs,"Rlo");
  for(it=0;it<12;it++){ // compensated (double-double) residual refinement with the available factors; stop when the DD residual stops decreasing
    if(P.fp32) k_resid_dd<float><<<grid1(n),256>>>(P.A32.p,n,P.lda,P.Xref.p,P.B.p,Rhi.p,Rlo.p,nrhs); else k_resid_dd<double><<<grid1(n),256>>>(P.A64.p,n,P.lda,P.Xref.p,P.B.p,Rhi.p,Rlo.p,nrhs); CUDA_CK(cudaGetLastError());
    std::vector<double> hi((size_t)n*nrhs),lo((size_t)n*nrhs),xx((size_t)n*nrhs); CUDA_CK(cudaMemcpy(hi.data(),Rhi.p,hi.size()*8,cudaMemcpyDeviceToHost)); CUDA_CK(cudaMemcpy(lo.data(),Rlo.p,lo.size()*8,cudaMemcpyDeviceToHost)); CUDA_CK(cudaMemcpy(xx.data(),P.Xref.p,xx.size()*8,cudaMemcpyDeviceToHost));
    double rm=0,xm=0; for(size_t i=0;i<hi.size();i++){ rm=std::max(rm,std::fabs(hi[i]+lo[i])); xm=std::max(xm,std::fabs(xx[i])); } double rel=(P.anorm_inf*xm>0)? rm/(P.anorm_inf*xm) : rm; if(!std::isfinite(rel)){ break; }
    if(rel<best){ best=rel; P.ref_resid_dd_abs=rm; CUDA_CK(cudaMemcpy(Xbest.p,P.Xref.p,xx.size()*8,cudaMemcpyDeviceToDevice)); } else break;
    if(rel<1e-30) break; // correction with DD residual (hi+lo rounded to double) through the factors
    k_add<double><<<grid1((size_t)n*nrhs),256>>>(Rhi.p,Rlo.p,(size_t)n*nrhs,1.0); apply_inverse(ctx,W,F,Rhi.p,W.D.p,nrhs,false,nullptr); k_add<double><<<grid1((size_t)n*nrhs),256>>>(P.Xref.p,W.D.p,(size_t)n*nrhs,1.0); CUDA_CK(cudaDeviceSynchronize()); }
  CUDA_CK(cudaMemcpy(P.Xref.p,Xbest.p,(size_t)n*nrhs*8,cudaMemcpyDeviceToDevice)); P.has_ref=std::isfinite(best); P.ref_resid_dd_rel=best; P.ref_iters=it; P.ref_grade=P.has_ref?"numerical_reference":"unavailable"; P.ref_method="cusolver_"+kind+"_lu_plus_double_double_residual_refinement"; P.growth_ref=F.growth_proxy;
  if(P.has_exact){ Metrics Mt; eval_metrics(ctx,W,P,P.Xref.p,Mt); P.ref_vs_exact_finf=Mt.einf; }
  // conditioning estimate: power iteration for sigma_max(A), inverse iteration with factors for sigma_min(A) (12 steps each, FP64 ops on stored A via dgemv/sgemv)
  { DBuf<double> v(n,"v"),w(n,"w"); std::vector<double> hv(n); Xoshiro rng(12345); for(auto& x:hv) x=rng.normal(); CUDA_CK(cudaMemcpy(v.p,hv.data(),n*8,cudaMemcpyHostToDevice)); double one=1,zero=0; DBuf<float> v32(n,"v32"),w32(n,"w32"); float fone=1,fzero=0;
    auto matvec=[&](cublasOperation_t op,const double* x,double* y){ if(P.fp32){ dev_convert<double,float>(x,v32.p,n); CUBLAS_CK(cublasSgemv(ctx.cb,op,n,n,&fone,P.A32.p,n,v32.p,1,&fzero,w32.p,1)); dev_convert<float,double>(w32.p,y,n);} else CUBLAS_CK(cublasDgemv(ctx.cb,op,n,n,&one,P.A64.p,n,x,1,&zero,y,1)); };
    double smax=0; for(int k=0;k<12;k++){ double nv; CUBLAS_CK(cublasDnrm2(ctx.cb,n,v.p,1,&nv)); double s=1.0/nv; CUBLAS_CK(cublasDscal(ctx.cb,n,&s,v.p,1)); matvec(CUBLAS_OP_N,v.p,w.p); matvec(CUBLAS_OP_T,w.p,v.p); double nw; CUBLAS_CK(cublasDnrm2(ctx.cb,n,v.p,1,&nw)); smax=std::sqrt(nw); }
    double smin=0; for(auto& x:hv) x=rng.normal(); CUDA_CK(cudaMemcpy(v.p,hv.data(),n*8,cudaMemcpyHostToDevice));
    for(int k=0;k<12;k++){ double nv; CUBLAS_CK(cublasDnrm2(ctx.cb,n,v.p,1,&nv)); double s=1.0/nv; CUBLAS_CK(cublasDscal(ctx.cb,n,&s,v.p,1)); // z = A^{-1} A^{-T} v
      if(F.prec==64){ CUDA_CK(cudaMemcpy(w.p,v.p,n*8,cudaMemcpyDeviceToDevice)); CUSOLVER_CK(cusolverDnXgetrs(ctx.cs,ctx.params,CUBLAS_OP_T,n,1,CUDA_R_64F,F.LU64.p,n,F.ipiv.p,CUDA_R_64F,w.p,n,W.dinfo.p)); CUSOLVER_CK(cusolverDnXgetrs(ctx.cs,ctx.params,CUBLAS_OP_N,n,1,CUDA_R_64F,F.LU64.p,n,F.ipiv.p,CUDA_R_64F,w.p,n,W.dinfo.p)); }
      else { dev_convert<double,float>(v.p,w32.p,n); CUSOLVER_CK(cusolverDnXgetrs(ctx.cs,ctx.params,CUBLAS_OP_T,n,1,CUDA_R_32F,F.LU32.p,n,F.ipiv.p,CUDA_R_32F,w32.p,n,W.dinfo.p)); CUSOLVER_CK(cusolverDnXgetrs(ctx.cs,ctx.params,CUBLAS_OP_N,n,1,CUDA_R_32F,F.LU32.p,n,F.ipiv.p,CUDA_R_32F,w32.p,n,W.dinfo.p)); dev_convert<float,double>(w32.p,w.p,n); }
      double nw; CUBLAS_CK(cublasDnrm2(ctx.cb,n,w.p,1,&nw)); smin=1.0/std::sqrt(nw); CUDA_CK(cudaMemcpy(v.p,w.p,n*8,cudaMemcpyDeviceToDevice)); }
    P.sigma_max_est=smax; P.sigma_min_est=smin; P.kappa2_est=(smin>0)? smax/smin : INFINITY; P.kappa_method="power_iteration_AtA_12_steps_and_inverse_iteration_with_"+kind+"_factors_12_steps_2norm_estimate"; }
  P.ref_seconds=wall_now()-t0; }

// ------------------------------------------------------------ output helpers
static void write_bin(const std::string& path,const void* data,size_t bytes){ std::ofstream f(path,std::ios::binary); f.write((const char*)data,bytes); if(!f) throw MPError("cannot write "+path); }
static std::string metric_arr(const std::vector<double>& v){ return J::arr(v); }

int main(int argc,char** argv){
  Args a=parse_args(argc,argv); if(a.cmd.empty()){ fprintf(stderr,"usage: mpsolve run|regen|caps|selftest --key value ...\n"); return 2; }
  try{
    Ctx ctx; ctx.init();
#ifdef HAVE_MAGMA
    magma_init();
#endif
    if(a.cmd=="caps"){ cudaDeviceProp p; CUDA_CK(cudaGetDeviceProperties(&p,0)); size_t fr,tot; CUDA_CK(cudaMemGetInfo(&fr,&tot)); int rt=0,drv=0; cudaRuntimeGetVersion(&rt); cudaDriverGetVersion(&drv); int cM=0,cm=0,cp=0; cusolverGetProperty(MAJOR_VERSION,&cM); cusolverGetProperty(MINOR_VERSION,&cm); cusolverGetProperty(PATCH_LEVEL,&cp); int cbv=0; cublasGetVersion(ctx.cb,&cbv);
      J j; j.s("device",p.name).s("compute_capability",std::to_string(p.major)+"."+std::to_string(p.minor)).i("total_mem_bytes",(int64_t)tot).i("free_mem_bytes",(int64_t)fr).i("cuda_runtime",rt).i("cuda_driver",drv).s("cusolver_version",std::to_string(cM)+"."+std::to_string(cm)+"."+std::to_string(cp)).i("cublas_version",cbv).i("multiprocessors",p.multiProcessorCount).i("sm_clock_khz",p.clockRate).i("mem_clock_khz",p.memoryClockRate).b("ecc_enabled",p.ECCEnabled).i("l2_bytes",p.l2CacheSize);
#ifdef HAVE_MAGMA
      magma_int_t maj,min,mic; magma_version(&maj,&min,&mic); j.s("magma_version",std::to_string(maj)+"."+std::to_string(min)+"."+std::to_string(mic)); j.b("magma_linked",true);
#else
      j.b("magma_linked",false);
#endif
      printf("%s\n",j.str_obj().c_str()); return 0; }
    // ---- case spec
    CaseSpec S; S.n=a.geti("n",1024); S.nrhs=a.geti("nrhs",1); S.family=a.get("family","prescribed"); S.spectrum=a.get("spectrum","geometric"); S.dtype=a.get("dtype","fp64"); S.rhs=a.get("rhs","random"); S.kappa=a.getd("kappa",100); S.seed=(uint64_t)std::stoull(a.get("seed","9200")); S.ell=a.getd("ell",0.25); S.nugget=a.getd("nugget",1e-3); S.dim=a.geti("dim",3); S.intmax=a.geti("intmax",255); S.dyadic_exp=a.geti("dyadic-exp",8); S.grcar_k=a.geti("grcar-k",3); S.global_exp=a.geti("global-exp",0); S.rowcol_span=a.geti("rowcol-span",0); S.case_id=a.get("case-id","case");
    bool host_mode=split(a.get("modes","resident"),',').size()>1 || a.get("modes","resident")=="host"; std::vector<std::string> modes=split(a.get("modes","resident"),',');
    Problem P; generate_problem(ctx,P,S,host_mode || a.cmd=="regen");
    if(a.cmd=="regen"){ J j; j.s("case_id",S.case_id).s("a_sha256",P.a_sha).s("b_sha256",P.b_sha).kv("tile_sha256",J::arr(P.tile_sha)).d("gen_seconds",P.gen_seconds).d("hash_seconds",P.hash_seconds).i("n",P.n).i("nrhs",P.nrhs).s("dtype",S.dtype); printf("%s\n",j.str_obj().c_str()); return 0; }
    if(a.cmd=="selftest"){ // orthogonality/density check of the generator at the requested n with identity spectrum
      std::vector<double> h((size_t)P.n*P.n); CUDA_CK(cudaMemcpy(h.data(),P.A64.p,h.size()*8,cudaMemcpyDeviceToHost)); int n=P.n; double maxdev=0; size_t zeros=0; double minabs=INFINITY;
      for(int i=0;i<n;i++) for(int j=i;j<n;j++){ double s=0; for(int k=0;k<n;k++) s+=h[k+(size_t)i*n]*h[k+(size_t)j*n]; maxdev=std::max(maxdev,std::fabs(s-(i==j?1.0:0.0))); }
      for(double v:h){ if(v==0) zeros++; minabs=std::min(minabs,std::fabs(v)); } printf("{\"selftest\":\"generator\",\"n\":%d,\"max_orthogonality_deviation\":%.3e,\"zero_entries\":%zu,\"min_abs\":%.3e,\"anorm_inf\":%.6f}\n",n,maxdev,zeros,minabs,P.anorm_inf); return 0; }
    // ---- run
    std::string out=a.get("out","."); mkdir(out.c_str(),0755); std::string target=a.get("target",S.dtype); double u=(target=="fp32")? std::ldexp(1.0,-24) : std::ldexp(1.0,-53); double mult=a.getd("tau-mult",100); double tau=mult*u;
    int reps=a.geti("reps",1),warm=a.geti("warmup",1); bool instrument=a.has("instrument"); int64_t mem_budget=(int64_t)a.getd("mem-budget-bytes",0); if(mem_budget<=0){ size_t fr,tot; CUDA_CK(cudaMemGetInfo(&fr,&tot)); mem_budget=(int64_t)(0.8*fr); }
    Workspace W; int nbmax=std::max(a.geti("nb",256),a.geti("nb-max",512)); W.ensure(ctx,P.n,P.nrhs,nbmax); W.nb=a.geti("nb",256); W.panel=a.get("panel","cusolver");
    std::vector<std::string> mids=split(a.get("methods","native_lu64"),','); std::vector<MethodSpec> ms; for(auto& id:mids) ms.push_back(make_method(id,a));
    compute_reference(ctx,W,P,a.get("reference","auto"),mem_budget - (int64_t)g_tracker.current);
    // case record
    std::string case_path=out+"/case_"+S.case_id+".json"; { J j; j.s("case_id",S.case_id).i("n",P.n).i("nrhs",P.nrhs).s("family",S.family).s("spectrum",S.spectrum).s("dtype",S.dtype).s("rhs",S.rhs).d("kappa_nominal",S.kappa).i("seed",(int64_t)S.seed).d("ell",S.ell).d("nugget",S.nugget).i("dim",S.dim).i("intmax",S.intmax).i("dyadic_exp",S.dyadic_exp).i("grcar_k",S.grcar_k).i("global_exp",S.global_exp).i("rowcol_span",S.rowcol_span)
      .s("a_sha256",P.a_sha).s("b_sha256",P.b_sha).kv("tile_sha256",J::arr(P.tile_sha)).i("ntiles",P.ntiles).d("anorm_inf",P.anorm_inf).d("amax",P.amax).b("stored_finite",P.scale_representable).kv("bnorm_inf",J::arr(P.bnorm_inf)).kv("bnorm_2",J::arr(P.bnorm_2)).d("gen_seconds",P.gen_seconds).d("hash_seconds",P.hash_seconds)
      .s("reference_grade",P.ref_grade).s("reference_method",P.ref_method).d("reference_resid_dd_rel",P.ref_resid_dd_rel).d("reference_resid_dd_abs",P.ref_resid_dd_abs).i("reference_iters",P.ref_iters).d("reference_seconds",P.ref_seconds).s("reference_note",P.ref_note).b("has_exact_solution",P.has_exact).kv("reference_vs_exact_finf",J::arr(P.ref_vs_exact_finf))
      .d("kappa2_estimate",P.kappa2_est).d("sigma_max_estimate",P.sigma_max_est).d("sigma_min_estimate",P.sigma_min_est).s("kappa_method",P.kappa_method).d("growth_proxy_reference_lu",P.growth_ref).s("target",target).d("tau",tau).d("tau_mult",mult).i("mem_budget_bytes",mem_budget).s("timestamp",iso_now());
      std::ofstream f(case_path); f<<j.str_obj()<<"\n"; }
    // save B (stored dtype) and X_exact / X_ref
    { size_t bb=(size_t)P.n*P.nrhs*(P.fp32?4:8); std::vector<char> hb(bb); CUDA_CK(cudaMemcpy(hb.data(),P.fp32?(const void*)P.B32.p:(const void*)P.B.p,bb,cudaMemcpyDeviceToHost)); write_bin(out+"/B_"+S.case_id+(P.fp32?".f32":".f64"),hb.data(),bb);
      if(P.has_ref){ std::vector<double> h((size_t)P.n*P.nrhs); CUDA_CK(cudaMemcpy(h.data(),P.Xref.p,h.size()*8,cudaMemcpyDeviceToHost)); write_bin(out+"/Xref_"+S.case_id+".f64",h.data(),h.size()*8); }
      if(P.has_exact){ std::vector<double> h((size_t)P.n*P.nrhs); CUDA_CK(cudaMemcpy(h.data(),P.Xexact.p,h.size()*8,cudaMemcpyDeviceToHost)); write_bin(out+"/Xexact_"+S.case_id+".f64",h.data(),h.size()*8); }
      if(a.has("embed-a")){ size_t ab=P.a_bytes(); std::vector<char> h(ab); CUDA_CK(cudaMemcpy(h.data(),P.a_ptr(),ab,cudaMemcpyDeviceToHost)); write_bin(out+"/A_"+S.case_id+(P.fp32?".f32":".f64"),h.data(),ab); } }
    std::ofstream trials(out+"/trials_"+S.case_id+".jsonl",std::ios::app), hist(out+"/hist_"+S.case_id+".jsonl",std::ios::app);
    Engine E(ctx,W,P,tau,tau); E.instrument=instrument; E.reuse=a.has("reuse"); IRSRunner IRS(ctx,P);
#ifdef HAVE_MAGMA
    MagmaRunner MG(ctx,P);
#endif
    int order_seed=a.geti("order-seed",1); std::string phase=a.get("phase","evaluation"); std::string run_tag=a.get("tag",""); int nm=(int)ms.size();
    for(auto& mode:modes){ for(int rep=-warm;rep<reps;rep++){ bool warmup=rep<0; int rot=(order_seed+rep+warm)%std::max(1,nm);
      for(int mi=0;mi<nm;mi++){ MethodSpec& M=ms[(mi+rot)%nm]; TrialOut T; Metrics Mt; std::string xpath; std::string xsha; std::string xdtype=(M.working==32)?"float32":"float64"; double t_e2e=-1,t_dev=-1; bool skipped=false; std::string skip_reason; size_t mem_before=g_tracker.current; double tstart=wall_now();
        try{
          // admission: rough estimate of extra buffers for this method
          size_t nn=(size_t)P.n*P.n; size_t need=0; if(M.kind=="irs"||M.kind=="magma"){ need=nn*(M.working==64?8:4)+ (size_t)P.n*P.n*4; } else { need=(M.factor=="cs64"||M.factor=="mg_d")? nn*8 : nn*4; if(M.apply64) need+=nn*8; if(M.policy!="none"){ need+=nn*8; } if(M.gmres) need+=(size_t)P.n*(2*M.gmres_m+2)*8; }
          // subtract buffers that already exist and will be reused (persistent factor / vendor work buffers)
          size_t have=0; if(M.kind=="irs"){ if(M.working==64 && IRS.A64work.n>=nn) have+=nn*8; if(M.working==32 && IRS.A32work.n>=nn) have+=nn*4; if(IRS.lwork) have+=IRS.lwork; }
#ifdef HAVE_MAGMA
          else if(M.kind=="magma"){ if(MG.A64w.n>=nn) have+=nn*8; if(MG.A32w.n>=nn) have+=nn*4; if(MG.dworks.n>=nn) have+=nn*4; }
#endif
          else { if((M.factor=="cs64"||M.factor=="mg_d") && E.fact.LU64.n>=nn) have+=nn*8; if(M.factor!="cs64" && M.factor!="mg_d" && E.fact.LU32.n>=nn) have+=nn*4; if(M.apply64 && E.fact.LU64_promoted.n>=nn) have+=nn*8; }
          need = (need>have)? need-have : 0;
          size_t fr,tot; CUDA_CK(cudaMemGetInfo(&fr,&tot)); if((int64_t)need>(int64_t)fr-(int64_t)(512ull<<20)){ skipped=true; skip_reason="memory_skip: need "+std::to_string(need)+" additional bytes, free "+std::to_string(fr); }
          if(!skipped){
            if(M.kind=="irs") IRS.prepare(M);
#ifdef HAVE_MAGMA
            if(M.kind=="magma") MG.prepare(M);
#endif
            CUDA_CK(cudaDeviceSynchronize()); double t0=wall_now();
            if(mode=="host"){ CUDA_CK(cudaMemcpy((void*)P.a_ptr(),P.hA,P.a_bytes(),cudaMemcpyHostToDevice)); if(P.fp32){ CUDA_CK(cudaMemcpy(P.B32.p,P.hB32,(size_t)P.n*P.nrhs*4,cudaMemcpyHostToDevice)); dev_convert<float,double>(P.B32.p,P.B.p,(size_t)P.n*P.nrhs);} else CUDA_CK(cudaMemcpy(P.B.p,P.hB64,(size_t)P.n*P.nrhs*8,cudaMemcpyHostToDevice)); }
            CUDA_CK(cudaDeviceSynchronize()); double t1=wall_now();
            if(M.kind=="irs"){ IRS.run(M,T); }
#ifdef HAVE_MAGMA
            else if(M.kind=="magma"){ MG.run(M,T); }
#endif
            else { E.run(M,T,!warmup); }
            CUDA_CK(cudaDeviceSynchronize()); double t2=wall_now();
            // final X in working dtype -> host (part of host_e2e boundary)
            const void* xsrc=nullptr; size_t xb=(size_t)P.n*P.nrhs*(M.working==32?4:8);
            if(M.kind=="irs"){ xsrc=(M.working==64)?(const void*)IRS.X64.p:(const void*)IRS.X32.p; }
#ifdef HAVE_MAGMA
            else if(M.kind=="magma"){ xsrc=(M.working==64)?(const void*)MG.X64.p:(const void*)MG.X32.p; }
#endif
            else { xsrc=(M.working==32)?(const void*)E.x32.p:(const void*)E.x64.p; }
            std::vector<char> hx(xb); if(T.ok){ CUDA_CK(cudaMemcpy(hx.data(),xsrc,xb,cudaMemcpyDeviceToHost)); } CUDA_CK(cudaDeviceSynchronize()); double t3=wall_now();
            t_dev=t2-t1; t_e2e=(mode=="host")? (t3-t0) : -1;
            if(T.ok){ // external evaluation (excluded from timing): metrics on the original stored system in FP64
              if(M.working==32){ CUDA_CK(cudaMemcpy(W.tmp32.p,xsrc,xb,cudaMemcpyDeviceToDevice)); dev_convert<float,double>(W.tmp32.p,W.Xd.p,(size_t)P.n*P.nrhs); } else CUDA_CK(cudaMemcpy(W.Xd.p,xsrc,xb,cudaMemcpyDeviceToDevice));
              eval_metrics(ctx,W,P,W.Xd.p,Mt); xsha=sha256_bytes(hx.data(),xb); if(!warmup){ xpath="X_"+S.case_id+"_"+M.id+"_"+mode+"_r"+std::to_string(rep)+(run_tag.empty()?"":"_"+run_tag)+(M.working==32?".f32":".f64"); write_bin(out+"/"+xpath,hx.data(),xb); } } }
        } catch(MPError& e){ T.ok=false; T.error=e.what(); T.terminal="runtime_error"; cudaGetLastError(); cudaDeviceSynchronize(); }
        double tend=wall_now(); size_t fr2,tot2; cudaMemGetInfo(&fr2,&tot2);
        bool b_ok=false; { size_t bb=(size_t)P.n*P.nrhs*(P.fp32?4:8); std::vector<char> hb(bb); if(cudaMemcpy(hb.data(),P.fp32?(const void*)P.B32.p:(const void*)P.B.p,bb,cudaMemcpyDeviceToHost)==cudaSuccess) b_ok=(sha256_bytes(hb.data(),bb)==P.b_sha); }
        // ---- trial record
        J j; j.s("case_id",S.case_id).s("method_id",M.id).s("mode",mode).i("rep",rep).b("warmup",warmup).s("phase",phase).s("tag",run_tag).s("target",target).d("tau",tau).b("skipped",skipped).s("skip_reason",skip_reason).b("ok",T.ok).s("error",T.error).s("terminal_reason",skipped?"memory_skip":T.terminal).s("final_stage",T.final_stage.empty()?(M.kind=="direct"?"direct":M.factor):T.final_stage).b("used_fallback",T.fallback).b("b_integrity_ok",b_ok)
          .d("host_e2e_s",t_e2e).d("device_ready_wall_s",t_dev).d("method_internal_total_s",T.t_total).d("factor_total_s",T.t_factor_total).d("initial_solve_s",T.t_initial).d("refine_s",T.t_refine).d("eval_s",T.t_eval).kv("stage_times_s",J::arr(T.stage_times)).kv("stages",J::arr(T.stages)).d("wall_including_eval_s",tend-tstart)
          .i("outer_iters",T.outer).i("inner_iters_total",T.inner_total).i("matvecs",T.matvecs).i("precond_applies",T.precond_applies).i("triangular_solves",T.trs).i("restarts",T.restarts).i("true_residual_checks",T.true_resid_checks).i("refactorizations",T.refactorizations).i("factor_info",T.factor_info).d("growth_proxy",T.growth).i("conv_overflow",(int64_t)T.conv_overflow).i("conv_subnormal",(int64_t)T.conv_subnormal)
          .i("irs_niters",T.irs_niters).i("irs_outer_niters",T.irs_outer).i("irs_info",T.irs_info).i("irs_workspace_bytes",(int64_t)T.irs_work).i("magma_iter",T.magma_iter).i("magma_info",T.magma_info).d("magma_facto_time_s",T.magma_facto_time)
          .d("solver_final_eta_max",T.final_eta).d("solver_final_omega_max",T.final_omega).i("peak_device_bytes_tracked",(int64_t)g_tracker.peak).i("device_free_after_bytes",(int64_t)fr2).i("method_extra_bytes_tracked",(int64_t)(g_tracker.current>mem_before? g_tracker.current-mem_before:0))
          .s("x_path",xpath).s("x_sha256",xsha).s("x_dtype",xdtype).b("metrics_finite",Mt.finite).d("eta_max",Mt.eta.empty()?-1:Mt.eta_max).d("omega_max",Mt.omega.empty()?-1:Mt.omega_max)
          .kv("eta",metric_arr(Mt.eta)).kv("omega",metric_arr(Mt.omega)).kv("rinf",metric_arr(Mt.rinf)).kv("r2",metric_arr(Mt.r2)).kv("relres_inf",metric_arr(Mt.relres_inf)).kv("relres_2",metric_arr(Mt.relres_2)).kv("xinf",metric_arr(Mt.xinf)).kv("worst_row",J::arr(std::vector<int64_t>(Mt.worst.begin(),Mt.worst.end())))
          .b("has_fwd",Mt.has_fwd).kv("finf",metric_arr(Mt.finf)).kv("f2",metric_arr(Mt.f2)).kv("fcomp_nonzero",metric_arr(Mt.fcomp)).kv("fzero_maxabs",metric_arr(Mt.fzero_maxabs)).kv("fzero_count",J::arr(Mt.fzero_count)).b("has_exact",Mt.has_exact).kv("einf",metric_arr(Mt.einf)).kv("e2",metric_arr(Mt.e2)).kv("ecomp_nonzero",metric_arr(Mt.ecomp)).kv("ezero_maxabs",metric_arr(Mt.ezero_maxabs)).kv("ezero_count",J::arr(Mt.ezero_count))
          .s("timestamp",iso_now()).i("nb",M.nb).i("gmres_m",M.gmres_m).d("gmres_tol",M.gmres_tol).i("gmres_budget",M.gmres_budget).i("gmres_cycles",M.gmres_cycles).i("max_outer",M.max_outer).s("policy",M.policy).i("budget",M.budget).i("scale_mode",M.scale_mode).i("working",M.working).i("resid",M.resid).b("apply64",M.apply64).b("reuse",a.has("reuse")).s("panel",W.panel).i("stages_count",(int64_t)T.stages.size());
        trials<<j.str_obj()<<"\n"; trials.flush();
        if(!warmup){ for(auto& h:T.hist){ J jh; jh.s("case_id",S.case_id).s("method_id",M.id).s("mode",mode).i("rep",rep).s("tag",run_tag).s("kind","outer").i("outer",h.outer).i("stage",h.stage).d("t_s",h.t).d("eta_max",h.eta_max).d("omega_max",h.omega_max).d("rinf_max",h.rinf_max).i("inner_in_step",h.inner_in_step).s("note",h.note); hist<<jh.str_obj()<<"\n"; }
          for(auto& h:T.ihist){ J jh; jh.s("case_id",S.case_id).s("method_id",M.id).s("mode",mode).i("rep",rep).s("tag",run_tag).s("kind","inner").i("outer",h.outer).i("cycle",h.cycle).i("j",h.j).i("rhs",h.rhs).d("recursive_rel",h.rec_rel).d("true_rel",h.true_rel).s("note",h.note); hist<<jh.str_obj()<<"\n"; } hist.flush(); }
        fprintf(stderr,"[%s] %s %s rep=%d %s dev=%.4fs e2e=%.4fs outer=%d inner=%d eta=%.2e omega=%.2e %s\n",S.case_id.c_str(),M.id.c_str(),mode.c_str(),rep,skipped?"SKIP":T.terminal.c_str(),t_dev,t_e2e,T.outer,T.inner_total,Mt.eta.empty()?-1:Mt.eta_max,Mt.omega.empty()?-1:Mt.omega_max,T.error.c_str());
      } } }
#ifdef HAVE_MAGMA
    magma_finalize();
#endif
    return 0;
  } catch(std::exception& e){ fprintf(stderr,"FATAL: %s\n",e.what()); J j; j.s("fatal",e.what()); printf("%s\n",j.str_obj().c_str()); return 1; }
}
