// Stage A probe: native cuSOLVER LU (FP32/FP64), cuSOLVER IRS mixed solver modes, FP16-input GEMM dispatch.
// Prints JSON lines. Reports errors instead of asserting.
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <string>
#define CK(x) do{ auto e=(x); if((int)e!=0){ printf("{\"error\":\"%s\",\"code\":%d,\"line\":%d}\n",#x,(int)e,__LINE__); fflush(stdout); return 1;} }while(0)
static double now(){ return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
__global__ void d2s(const double*a,float*b,size_t n){ size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i<n) b[i]=(float)a[i]; }
__global__ void s2h(const float*a,__half*b,size_t n){ size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i<n) b[i]=__float2half(a[i]); }
// residual r = b - A x in double, A double col-major
__global__ void resid(const double*A,const double*x,const double*b,double*r,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return; double s=b[i]; for(int j=0;j<n;j++) s=fma(-A[i+(size_t)j*n],x[j],s); r[i]=s; }
__global__ void absAx(const double*A,const double*x,const double*b,double*d,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return; double s=fabs(b[i]); for(int j=0;j<n;j++) s=fma(fabs(A[i+(size_t)j*n]),fabs(x[j]),s); d[i]=s; }

int main(int argc,char**argv){
  int n = argc>1? atoi(argv[1]) : 2048; int nrhs=1;
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
  size_t fr,tot; CK(cudaMemGetInfo(&fr,&tot));
  int rt=0,drv=0; cudaRuntimeGetVersion(&rt); cudaDriverGetVersion(&drv);
  int cusM=0,cusm=0,cusp=0; cusolverGetProperty(MAJOR_VERSION,&cusM); cusolverGetProperty(MINOR_VERSION,&cusm); cusolverGetProperty(PATCH_LEVEL,&cusp);
  int cbv=0; cublasHandle_t cb; CK(cublasCreate(&cb)); cublasGetVersion(cb,&cbv);
  printf("{\"device\":\"%s\",\"cc\":\"%d.%d\",\"total_mem\":%zu,\"free_mem\":%zu,\"runtime\":%d,\"driver\":%d,\"cusolver\":\"%d.%d.%d\",\"cublas\":%d,\"sm\":%d,\"clock_khz\":%d,\"mem_clock_khz\":%d,\"ecc\":%d}\n",p.name,p.major,p.minor,tot,fr,rt,drv,cusM,cusm,cusp,cbv,p.multiProcessorCount,p.clockRate,p.memoryClockRate,p.ECCEnabled);
  // random well-conditioned matrix: A = I*n^0.5 + N(0,1)? use diag dominant-ish for a quick probe
  std::mt19937_64 rng(42); std::normal_distribution<double> nd(0.0,1.0);
  std::vector<double> hA((size_t)n*n), hb(n);
  for(size_t i=0;i<hA.size();i++) hA[i]=nd(rng);
  for(int i=0;i<n;i++) hA[i+(size_t)i*n]+= 2.0*std::sqrt((double)n); // mild diag boost so kappa moderate
  for(int i=0;i<n;i++) hb[i]=nd(rng);
  double *dA,*dLU,*db,*dx,*dr,*dd; float *sA,*sLU,*sb,*sx;
  CK(cudaMalloc(&dA,sizeof(double)*n*n)); CK(cudaMalloc(&dLU,sizeof(double)*n*n)); CK(cudaMalloc(&db,sizeof(double)*n)); CK(cudaMalloc(&dx,sizeof(double)*n)); CK(cudaMalloc(&dr,sizeof(double)*n)); CK(cudaMalloc(&dd,sizeof(double)*n));
  CK(cudaMalloc(&sA,sizeof(float)*n*n)); CK(cudaMalloc(&sLU,sizeof(float)*n*n)); CK(cudaMalloc(&sb,sizeof(float)*n)); CK(cudaMalloc(&sx,sizeof(float)*n));
  CK(cudaMemcpy(dA,hA.data(),sizeof(double)*n*n,cudaMemcpyHostToDevice)); CK(cudaMemcpy(db,hb.data(),sizeof(double)*n,cudaMemcpyHostToDevice));
  d2s<<<(n*(size_t)n+255)/256,256>>>(dA,sA,(size_t)n*n); d2s<<<(n+255)/256,256>>>(db,sb,n);
  cusolverDnHandle_t h; CK(cusolverDnCreate(&h));
  cusolverDnParams_t params; CK(cusolverDnCreateParams(&params));
  int64_t *ipiv; int *dinfo; CK(cudaMalloc(&ipiv,sizeof(int64_t)*n)); CK(cudaMalloc(&dinfo,sizeof(int)));
  // ---- FP64 Xgetrf/Xgetrs
  size_t wd=0,wh=0; CK(cusolverDnXgetrf_bufferSize(h,params,n,n,CUDA_R_64F,dLU,n,CUDA_R_64F,&wd,&wh));
  void *dwork; CK(cudaMalloc(&dwork,wd+1)); std::vector<char> hwork(wh+1);
  double aNrm=0; { std::vector<double> rs(n,0.0); for(int j=0;j<n;j++) for(int i=0;i<n;i++) rs[i]+=std::fabs(hA[i+(size_t)j*n]); for(int i=0;i<n;i++) aNrm=std::max(aNrm,rs[i]); }
  auto eval=[&](const char*tag,double t,int extra)->int{
    resid<<<(n+255)/256,256>>>(dA,dx,db,dr,n); absAx<<<(n+255)/256,256>>>(dA,dx,db,dd,n); CK(cudaDeviceSynchronize());
    std::vector<double> r(n),d(n),x(n); CK(cudaMemcpy(r.data(),dr,sizeof(double)*n,cudaMemcpyDeviceToHost)); CK(cudaMemcpy(d.data(),dd,sizeof(double)*n,cudaMemcpyDeviceToHost)); CK(cudaMemcpy(x.data(),dx,sizeof(double)*n,cudaMemcpyDeviceToHost));
    double rinf=0,xinf=0,binf=0,om=0; for(int i=0;i<n;i++){ rinf=std::max(rinf,std::fabs(r[i])); xinf=std::max(xinf,std::fabs(x[i])); binf=std::max(binf,std::fabs(hb[i])); om=std::max(om,std::fabs(r[i])/d[i]); }
    double eta=rinf/(aNrm*xinf+binf);
    printf("{\"probe\":\"%s\",\"n\":%d,\"time_s\":%.6f,\"eta_inf\":%.3e,\"omega\":%.3e,\"extra\":%d}\n",tag,n,t,eta,om,extra); fflush(stdout); return 0; };
  for(int rep=0;rep<2;rep++){
    CK(cudaMemcpy(dLU,dA,sizeof(double)*n*n,cudaMemcpyDeviceToDevice)); CK(cudaMemcpy(dx,db,sizeof(double)*n,cudaMemcpyDeviceToDevice)); CK(cudaDeviceSynchronize());
    double t0=now(); CK(cusolverDnXgetrf(h,params,n,n,CUDA_R_64F,dLU,n,ipiv,CUDA_R_64F,dwork,wd,hwork.data(),wh,dinfo));
    CK(cusolverDnXgetrs(h,params,CUBLAS_OP_N,n,nrhs,CUDA_R_64F,dLU,n,ipiv,CUDA_R_64F,dx,n,dinfo)); CK(cudaDeviceSynchronize()); double t1=now();
    int info; CK(cudaMemcpy(&info,dinfo,4,cudaMemcpyDeviceToHost)); if(eval("xgetrf64_getrs",t1-t0,info)) return 1;
  }
  // ---- FP32 Xgetrf/Xgetrs (promote x to double for metrics)
  size_t ws=0,wsh=0; CK(cusolverDnXgetrf_bufferSize(h,params,n,n,CUDA_R_32F,sLU,n,CUDA_R_32F,&ws,&wsh)); void*swork; CK(cudaMalloc(&swork,ws+1)); std::vector<char> swh(wsh+1);
  for(int rep=0;rep<2;rep++){
    CK(cudaMemcpy(sLU,sA,sizeof(float)*n*n,cudaMemcpyDeviceToDevice)); CK(cudaMemcpy(sx,sb,sizeof(float)*n,cudaMemcpyDeviceToDevice)); CK(cudaDeviceSynchronize());
    double t0=now(); CK(cusolverDnXgetrf(h,params,n,n,CUDA_R_32F,sLU,n,ipiv,CUDA_R_32F,swork,ws,swh.data(),wsh,dinfo));
    CK(cusolverDnXgetrs(h,params,CUBLAS_OP_N,n,nrhs,CUDA_R_32F,sLU,n,ipiv,CUDA_R_32F,sx,n,dinfo)); CK(cudaDeviceSynchronize()); double t1=now();
    std::vector<float> xs(n); CK(cudaMemcpy(xs.data(),sx,4*n,cudaMemcpyDeviceToHost)); std::vector<double> xd(n); for(int i=0;i<n;i++) xd[i]=xs[i]; CK(cudaMemcpy(dx,xd.data(),8*n,cudaMemcpyHostToDevice));
    int info; CK(cudaMemcpy(&info,dinfo,4,cudaMemcpyDeviceToHost)); if(eval("xgetrf32_getrs",t1-t0,info)) return 1;
  }
  // ---- legacy sgetrf (int ipiv) for comparison
  { int lw=0; CK(cusolverDnSgetrf_bufferSize(h,n,n,sLU,n,&lw)); float*w; CK(cudaMalloc(&w,sizeof(float)*(lw+1))); int*ip; CK(cudaMalloc(&ip,4*n));
    for(int rep=0;rep<2;rep++){ CK(cudaMemcpy(sLU,sA,sizeof(float)*n*n,cudaMemcpyDeviceToDevice)); CK(cudaMemcpy(sx,sb,4*n,cudaMemcpyDeviceToDevice)); CK(cudaDeviceSynchronize()); double t0=now();
      CK(cusolverDnSgetrf(h,n,n,sLU,n,w,ip,dinfo)); CK(cusolverDnSgetrs(h,CUBLAS_OP_N,n,1,sLU,n,ip,sx,n,dinfo)); CK(cudaDeviceSynchronize()); double t1=now();
      std::vector<float> xs(n); CK(cudaMemcpy(xs.data(),sx,4*n,cudaMemcpyDeviceToHost)); std::vector<double> xd(n); for(int i=0;i<n;i++) xd[i]=xs[i]; CK(cudaMemcpy(dx,xd.data(),8*n,cudaMemcpyHostToDevice)); int info; CK(cudaMemcpy(&info,dinfo,4,cudaMemcpyDeviceToHost)); if(eval("legacy_sgetrf_getrs",t1-t0,info)) return 1; }
    cudaFree(w); cudaFree(ip);
    int lwd=0; CK(cusolverDnDgetrf_bufferSize(h,n,n,dLU,n,&lwd)); double*wdd; CK(cudaMalloc(&wdd,sizeof(double)*(lwd+1))); int*ipd; CK(cudaMalloc(&ipd,4*n));
    for(int rep=0;rep<2;rep++){ CK(cudaMemcpy(dLU,dA,sizeof(double)*n*n,cudaMemcpyDeviceToDevice)); CK(cudaMemcpy(dx,db,8*n,cudaMemcpyDeviceToDevice)); CK(cudaDeviceSynchronize()); double t0=now();
      CK(cusolverDnDgetrf(h,n,n,dLU,n,wdd,ipd,dinfo)); CK(cusolverDnDgetrs(h,CUBLAS_OP_N,n,1,dLU,n,ipd,dx,n,dinfo)); CK(cudaDeviceSynchronize()); double t1=now(); int info; CK(cudaMemcpy(&info,dinfo,4,cudaMemcpyDeviceToHost)); if(eval("legacy_dgetrf_getrs",t1-t0,info)) return 1; }
    cudaFree(wdd); cudaFree(ipd);
  }
  // ---- IRS modes
  struct Mode{ const char*name; cusolverPrecType_t mainp, lowp; cusolverIRSRefinement_t ref; int fallback; };
  Mode modes[]={ {"irs_64_64_classical",CUSOLVER_R_64F,CUSOLVER_R_64F,CUSOLVER_IRS_REFINE_CLASSICAL,1},
                 {"irs_64_32_classical",CUSOLVER_R_64F,CUSOLVER_R_32F,CUSOLVER_IRS_REFINE_CLASSICAL,1},
                 {"irs_64_16_classical",CUSOLVER_R_64F,CUSOLVER_R_16F,CUSOLVER_IRS_REFINE_CLASSICAL,1},
                 {"irs_64_16_classical_gmres",CUSOLVER_R_64F,CUSOLVER_R_16F,CUSOLVER_IRS_REFINE_CLASSICAL_GMRES,1},
                 {"irs_64_16_gmres",CUSOLVER_R_64F,CUSOLVER_R_16F,CUSOLVER_IRS_REFINE_GMRES,1},
                 {"irs_64_16_gmres_gmres",CUSOLVER_R_64F,CUSOLVER_R_16F,CUSOLVER_IRS_REFINE_GMRES_GMRES,1},
                 {"irs_64_16_classical_nofallback",CUSOLVER_R_64F,CUSOLVER_R_16F,CUSOLVER_IRS_REFINE_CLASSICAL,0},
                 {"irs_64_16bf_classical",CUSOLVER_R_64F,CUSOLVER_R_16BF,CUSOLVER_IRS_REFINE_CLASSICAL,1},
                 {"irs_64_tf32_classical",CUSOLVER_R_64F,CUSOLVER_R_TF32,CUSOLVER_IRS_REFINE_CLASSICAL,1},
                 {"irs_32_16_classical",CUSOLVER_R_32F,CUSOLVER_R_16F,CUSOLVER_IRS_REFINE_CLASSICAL,1},
                 {"irs_32_32_classical",CUSOLVER_R_32F,CUSOLVER_R_32F,CUSOLVER_IRS_REFINE_CLASSICAL,1},
                 {"irs_32_16_gmres",CUSOLVER_R_32F,CUSOLVER_R_16F,CUSOLVER_IRS_REFINE_GMRES,1} };
  for(auto&m:modes){
    cusolverDnIRSParams_t ip; cusolverDnIRSInfos_t inf; CK(cusolverDnIRSParamsCreate(&ip)); CK(cusolverDnIRSInfosCreate(&inf));
    CK(cusolverDnIRSParamsSetSolverPrecisions(ip,m.mainp,m.lowp)); CK(cusolverDnIRSParamsSetRefinementSolver(ip,m.ref));
    if(m.fallback) cusolverDnIRSParamsEnableFallback(ip); else cusolverDnIRSParamsDisableFallback(ip);
    CK(cusolverDnIRSParamsSetMaxIters(ip,50)); cusolverDnIRSInfosRequestResidual(inf);
    size_t lw=0; auto st=cusolverDnIRSXgesv_bufferSize(h,ip,n,nrhs,&lw);
    if(st!=CUSOLVER_STATUS_SUCCESS){ printf("{\"probe\":\"%s\",\"n\":%d,\"status\":\"bufferSize_failed\",\"code\":%d}\n",m.name,n,(int)st); cusolverDnIRSParamsDestroy(ip); cusolverDnIRSInfosDestroy(inf); continue; }
    void*w; CK(cudaMalloc(&w,lw+1));
    bool is32 = (m.mainp==CUSOLVER_R_32F);
    for(int rep=0;rep<3;rep++){
      cusolverDnIRSInfosDestroy(inf); CK(cusolverDnIRSInfosCreate(&inf)); cusolverDnIRSInfosRequestResidual(inf);
      int niters=-999; CK(cudaDeviceSynchronize()); double t0=now();
      cusolverStatus_t s2;
      if(is32){ CK(cudaMemcpy(sLU,sA,sizeof(float)*n*n,cudaMemcpyDeviceToDevice)); s2=cusolverDnIRSXgesv(h,ip,inf,n,nrhs,sLU,n,sb,n,sx,n,w,lw,&niters,dinfo); }
      else { CK(cudaMemcpy(dLU,dA,sizeof(double)*n*n,cudaMemcpyDeviceToDevice)); s2=cusolverDnIRSXgesv(h,ip,inf,n,nrhs,dLU,n,db,n,dx,n,w,lw,&niters,dinfo); }
      cudaError_t ce=cudaDeviceSynchronize(); double t1=now();
      int outer=-999; cusolverDnIRSInfosGetOuterNiters(inf,&outer); int info=-999; cudaMemcpy(&info,dinfo,4,cudaMemcpyDeviceToHost);
      if(s2!=CUSOLVER_STATUS_SUCCESS||ce!=cudaSuccess){ printf("{\"probe\":\"%s\",\"n\":%d,\"status\":\"solve_failed\",\"code\":%d,\"cuda\":%d,\"niters\":%d,\"info\":%d,\"workspace_bytes\":%zu}\n",m.name,n,(int)s2,(int)ce,niters,info,lw); fflush(stdout); cudaGetLastError(); break; }
      if(is32){ std::vector<float> xs(n); CK(cudaMemcpy(xs.data(),sx,4*n,cudaMemcpyDeviceToHost)); std::vector<double> xd(n); for(int i=0;i<n;i++) xd[i]=xs[i]; CK(cudaMemcpy(dx,xd.data(),8*n,cudaMemcpyHostToDevice)); }
      printf("{\"probe_mode\":\"%s\",\"niters\":%d,\"outer_niters\":%d,\"info\":%d,\"workspace_bytes\":%zu,\"fallback_enabled\":%d}\n",m.name,niters,outer,info,lw,m.fallback);
      if(eval(m.name,t1-t0,niters)) return 1;
    }
    cudaFree(w); cusolverDnIRSParamsDestroy(ip); cusolverDnIRSInfosDestroy(inf);
  }
  // ---- FP16-input GEMM via cublasGemmEx with FP32 accumulation (tensor op)
  { int m2=n; __half *hA16,*hB16; float*C; CK(cudaMalloc(&hA16,2*(size_t)m2*m2)); CK(cudaMalloc(&hB16,2*(size_t)m2*m2)); CK(cudaMalloc(&C,4*(size_t)m2*m2));
    s2h<<<((size_t)m2*m2+255)/256,256>>>(sA,hA16,(size_t)m2*m2); s2h<<<((size_t)m2*m2+255)/256,256>>>(sA,hB16,(size_t)m2*m2); CK(cudaDeviceSynchronize());
    float one=1.f,zero=0.f; /* default math mode: no FP16 down-conversion for FP32 GEMM */
    for(int rep=0;rep<3;rep++){ CK(cudaDeviceSynchronize()); double t0=now(); CK(cublasGemmEx(cb,CUBLAS_OP_N,CUBLAS_OP_N,m2,m2,m2,&one,hA16,CUDA_R_16F,m2,hB16,CUDA_R_16F,m2,&zero,C,CUDA_R_32F,m2,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP)); CK(cudaDeviceSynchronize()); double t1=now();
      printf("{\"probe\":\"gemmex_fp16in_fp32acc\",\"n\":%d,\"time_s\":%.6f,\"tflops\":%.2f}\n",m2,t1-t0,2.0*m2*(double)m2*m2/(t1-t0)/1e12); }
    for(int rep=0;rep<3;rep++){ CK(cudaDeviceSynchronize()); double t0=now(); CK(cublasSgemm(cb,CUBLAS_OP_N,CUBLAS_OP_N,m2,m2,m2,&one,sA,m2,sA,m2,&zero,C,m2)); CK(cudaDeviceSynchronize()); double t1=now();
      printf("{\"probe\":\"sgemm\",\"n\":%d,\"time_s\":%.6f,\"tflops\":%.2f}\n",m2,t1-t0,2.0*m2*(double)m2*m2/(t1-t0)/1e12); }
    double done=1,dzero=0; for(int rep=0;rep<3;rep++){ CK(cudaDeviceSynchronize()); double t0=now(); CK(cublasDgemm(cb,CUBLAS_OP_N,CUBLAS_OP_N,m2,m2,m2,&done,dA,m2,dA,m2,&dzero,dLU,m2)); CK(cudaDeviceSynchronize()); double t1=now();
      printf("{\"probe\":\"dgemm\",\"n\":%d,\"time_s\":%.6f,\"tflops\":%.2f}\n",m2,t1-t0,2.0*m2*(double)m2*m2/(t1-t0)/1e12); }
    cudaFree(hA16); cudaFree(hB16); cudaFree(C); }
  printf("{\"probe\":\"done\"}\n");
  return 0;
}
