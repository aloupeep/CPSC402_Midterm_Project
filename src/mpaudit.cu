// mpaudit.cu — independent audit tool: regenerates the stored problem in a fresh process (hash verification),
// recomputes original-system metrics for saved final solutions with FP64 and double-double residuals, optional A dump.
// Shares generators/kernels with the solver sources but is a separate executable; it never runs a solver.
#include "mp_core.cuh"
#include <sys/stat.h>
#include <unistd.h>
static std::map<std::string,std::string> parse(int argc,char** argv){ std::map<std::string,std::string> kv; for(int i=1;i<argc;i++){ std::string s=argv[i]; if(s.rfind("--",0)==0){ std::string k=s.substr(2); if(i+1<argc && std::string(argv[i+1]).rfind("--",0)!=0) kv[k]=argv[++i]; else kv[k]="1"; } } return kv; }
int main(int argc,char** argv){
  auto a=parse(argc,argv); auto get=[&](const std::string& k,const std::string& d){ auto it=a.find(k); return it==a.end()? d: it->second; };
  try{
    Ctx ctx; ctx.init();
#ifdef HAVE_MAGMA
    magma_init();
#endif
    CaseSpec S; S.n=std::stoi(get("n","1024")); S.nrhs=std::stoi(get("nrhs","1")); S.family=get("family","prescribed"); S.spectrum=get("spectrum","geometric"); S.dtype=get("dtype","fp64"); S.rhs=get("rhs","random"); S.kappa=std::stod(get("kappa","100")); S.seed=std::stoull(get("seed","9200")); S.ell=std::stod(get("ell","0.25")); S.nugget=std::stod(get("nugget","1e-3")); S.dim=std::stoi(get("dim","3")); S.intmax=std::stoi(get("intmax","255")); S.dyadic_exp=std::stoi(get("dyadic-exp","8")); S.grcar_k=std::stoi(get("grcar-k","3")); S.global_exp=std::stoi(get("global-exp","0")); S.rowcol_span=std::stoi(get("rowcol-span","0")); S.case_id=get("case-id","case");
    Problem P; generate_problem(ctx,P,S,false); int n=P.n,nrhs=P.nrhs;
    Workspace W; W.ensure(ctx,n,nrhs,64);
    J jc; jc.s("case_id",S.case_id).s("regenerated_a_sha256",P.a_sha).s("regenerated_b_sha256",P.b_sha).kv("regenerated_tile_sha256",J::arr(P.tile_sha)).d("gen_seconds",P.gen_seconds).d("hash_seconds",P.hash_seconds).d("anorm_inf",P.anorm_inf).s("timestamp",iso_now()).s("audit_process_pid",std::to_string((long)getpid()));
    printf("%s\n",jc.str_obj().c_str()); fflush(stdout);
    if(a.count("dump-a")){ size_t ab=P.a_bytes(); std::vector<char> h(ab); CUDA_CK(cudaMemcpy(h.data(),P.a_ptr(),ab,cudaMemcpyDeviceToHost)); std::ofstream f(get("dump-a",""),std::ios::binary); f.write(h.data(),ab); }
    if(a.count("xref")){ std::ifstream f(get("xref",""),std::ios::binary); std::vector<double> h((size_t)n*nrhs); f.read((char*)h.data(),h.size()*8); if(f){ P.Xref.alloc(h.size(),"Xref"); CUDA_CK(cudaMemcpy(P.Xref.p,h.data(),h.size()*8,cudaMemcpyHostToDevice)); P.has_ref=true; } }
    if(a.count("xexact")){ std::ifstream f(get("xexact",""),std::ios::binary); std::vector<double> h((size_t)n*nrhs); f.read((char*)h.data(),h.size()*8); if(f){ if(!P.Xexact.p) P.Xexact.alloc(h.size(),"Xexact"); CUDA_CK(cudaMemcpy(P.Xexact.p,h.data(),h.size()*8,cudaMemcpyHostToDevice)); P.has_exact=true; } }
    // list of solutions: file with lines "<path> <float32|float64>"
    std::ifstream lst(get("x-list","")); std::string path,dt; DBuf<double> X((size_t)n*nrhs,"X"); DBuf<float> X32((size_t)n*nrhs,"X32");
    while(lst>>path>>dt){ std::ifstream f(path,std::ios::binary); if(!f){ printf("{\"x_path\":\"%s\",\"error\":\"cannot open\"}\n",path.c_str()); continue; }
      size_t cnt=(size_t)n*nrhs; std::vector<char> raw(cnt*(dt=="float32"?4:8)); f.read(raw.data(),raw.size()); if(!f){ printf("{\"x_path\":\"%s\",\"error\":\"short read\"}\n",path.c_str()); continue; }
      std::string xsha=sha256_bytes(raw.data(),raw.size());
      if(dt=="float32"){ CUDA_CK(cudaMemcpy(X32.p,raw.data(),raw.size(),cudaMemcpyHostToDevice)); dev_convert<float,double>(X32.p,X.p,cnt); } else CUDA_CK(cudaMemcpy(X.p,raw.data(),raw.size(),cudaMemcpyHostToDevice));
      Metrics M; eval_metrics(ctx,W,P,X.p,M); std::vector<double> rdd; resid_dd_norms(W,P,X.p,rdd);
      std::vector<double> eta_dd(nrhs),om_ratio(nrhs); for(int c=0;c<nrhs;c++){ double den=P.anorm_inf*M.xinf[c]+P.bnorm_inf[c]; eta_dd[c]=(rdd[c]==0)?0:(den==0?INFINITY:rdd[c]/den); }
      J j; j.s("x_path",path).s("x_sha256",xsha).s("x_dtype",dt).b("finite",M.finite).kv("eta_fp64",J::arr(M.eta)).kv("omega_fp64",J::arr(M.omega)).kv("rinf_fp64",J::arr(M.rinf)).kv("rinf_dd",J::arr(rdd)).kv("eta_dd",J::arr(eta_dd)).kv("relres_inf",J::arr(M.relres_inf)).kv("relres_2",J::arr(M.relres_2)).kv("xinf",J::arr(M.xinf))
        .b("has_ref",M.has_fwd).kv("finf_ref",J::arr(M.finf)).kv("f2_ref",J::arr(M.f2)).kv("fcomp_ref",J::arr(M.fcomp)).b("has_exact",M.has_exact).kv("finf_exact",J::arr(M.einf)).kv("f2_exact",J::arr(M.e2)).kv("fcomp_exact",J::arr(M.ecomp)).kv("ezero_count",J::arr(M.ezero_count));
      printf("%s\n",j.str_obj().c_str()); fflush(stdout); }
#ifdef HAVE_MAGMA
    magma_finalize();
#endif
    return 0;
  } catch(std::exception& e){ printf("{\"fatal\":\"%s\"}\n",J::esc(e.what()).c_str()); return 1; }
}
