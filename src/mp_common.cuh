// mp_common.cuh — shared infrastructure for the mixed-precision LU / iterative-refinement campaign driver.
// Original work for the CPSC 542G project (2026 campaign). See ATTRIBUTION.md in the result package.
#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_fp16.h>
#include <openssl/evp.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <stdexcept>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <fstream>

// ------------------------------------------------------------------ error handling
struct MPError : std::runtime_error { using std::runtime_error::runtime_error; };
#define CUDA_CK(x) do{ cudaError_t e_=(x); if(e_!=cudaSuccess){ std::ostringstream o_; o_<<"CUDA error "<<cudaGetErrorString(e_)<<" ("<<(int)e_<<") at "<<__FILE__<<":"<<__LINE__<<" in "<<#x; throw MPError(o_.str()); } }while(0)
#define CUBLAS_CK(x) do{ cublasStatus_t e_=(x); if(e_!=CUBLAS_STATUS_SUCCESS){ std::ostringstream o_; o_<<"cuBLAS error "<<(int)e_<<" at "<<__FILE__<<":"<<__LINE__<<" in "<<#x; throw MPError(o_.str()); } }while(0)
#define CUSOLVER_CK(x) do{ cusolverStatus_t e_=(x); if(e_!=CUSOLVER_STATUS_SUCCESS){ std::ostringstream o_; o_<<"cuSOLVER error "<<(int)e_<<" at "<<__FILE__<<":"<<__LINE__<<" in "<<#x; throw MPError(o_.str()); } }while(0)

static inline double wall_now(){ return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static inline std::string iso_now(){ auto t=std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()); char b[64]; std::strftime(b,sizeof b,"%Y-%m-%dT%H:%M:%S%z",std::localtime(&t)); return b; }

// ------------------------------------------------------------------ deterministic PRNG (host + device): splitmix64 / xoshiro256**
__host__ __device__ inline uint64_t splitmix64(uint64_t& s){ uint64_t z=(s+=0x9E3779B97F4A7C15ULL); z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL; z=(z^(z>>27))*0x94D049BB133111EBULL; return z^(z>>31); }
__host__ __device__ inline uint64_t hash3(uint64_t a,uint64_t b,uint64_t c){ uint64_t s=a*0x9E3779B97F4A7C15ULL ^ (b+0x7F4A7C15ULL)*0xBF58476D1CE4E5B9ULL ^ (c+0x1234567ULL)*0x94D049BB133111EBULL; splitmix64(s); return splitmix64(s); }
struct Xoshiro { uint64_t s[4];
  explicit Xoshiro(uint64_t seed){ uint64_t x=seed; for(int i=0;i<4;i++) s[i]=splitmix64(x); }
  static inline uint64_t rotl(uint64_t x,int k){ return (x<<k)|(x>>(64-k)); }
  uint64_t next(){ uint64_t r=rotl(s[1]*5,7)*9; uint64_t t=s[1]<<17; s[2]^=s[0]; s[3]^=s[1]; s[1]^=s[2]; s[0]^=s[3]; s[2]^=t; s[3]=rotl(s[3],45); return r; }
  double uniform(){ return (next()>>11)*0x1.0p-53; }           // [0,1)
  double normal(){ double u1,u2; do{ u1=uniform(); }while(u1<=0); u2=uniform(); return std::sqrt(-2.0*std::log(u1))*std::cos(6.283185307179586*u2); }
  int64_t below(int64_t n){ return (int64_t)(next()%(uint64_t)n); }
};

// ------------------------------------------------------------------ SHA-256 (OpenSSL EVP)
static inline std::string sha256_bytes(const void* data,size_t len){
  unsigned char md[32]; unsigned int mdl=0; EVP_MD_CTX* c=EVP_MD_CTX_new(); EVP_DigestInit_ex(c,EVP_sha256(),nullptr);
  const unsigned char* p=(const unsigned char*)data; size_t off=0; while(off<len){ size_t ch=std::min<size_t>(len-off,1u<<28); EVP_DigestUpdate(c,p+off,ch); off+=ch; }
  EVP_DigestFinal_ex(c,md,&mdl); EVP_MD_CTX_free(c); std::ostringstream o; for(unsigned i=0;i<mdl;i++) o<<std::hex<<std::setw(2)<<std::setfill('0')<<(int)md[i]; return o.str(); }

// ------------------------------------------------------------------ minimal JSON emitter (finite numbers only; nonfinite -> null)
struct J { std::ostringstream o; bool first=true;
  static std::string esc(const std::string& s){ std::string r; for(char c:s){ if(c=='"') r+="\\\""; else if(c=='\\') r+="\\\\"; else if(c=='\n') r+="\\n"; else r+=c; } return r; }
  static std::string num(double v){ if(!std::isfinite(v)) return "null"; char b[40]; std::snprintf(b,sizeof b,"%.17g",v); std::string s=b; if(s.find_first_of(".eEn")==std::string::npos) s+=".0"; return s; }
  static std::string num(int64_t v){ return std::to_string(v); }
  static std::string str(const std::string& s){ return "\""+esc(s)+"\""; }
  static std::string arr(const std::vector<double>& v){ std::string r="["; for(size_t i=0;i<v.size();i++){ if(i) r+=","; r+=num(v[i]); } return r+"]"; }
  static std::string arr(const std::vector<int64_t>& v){ std::string r="["; for(size_t i=0;i<v.size();i++){ if(i) r+=","; r+=num(v[i]); } return r+"]"; }
  static std::string arr(const std::vector<std::string>& v){ std::string r="["; for(size_t i=0;i<v.size();i++){ if(i) r+=","; r+=str(v[i]); } return r+"]"; }
  J& kv(const std::string& k,const std::string& rawvalue){ if(!first) o<<","; first=false; o<<"\""<<esc(k)<<"\":"<<rawvalue; return *this; }
  J& s(const std::string& k,const std::string& v){ return kv(k,str(v)); }
  J& d(const std::string& k,double v){ return kv(k,num(v)); }
  J& i(const std::string& k,int64_t v){ return kv(k,num(v)); }
  J& b(const std::string& k,bool v){ return kv(k,v?"true":"false"); }
  J& null(const std::string& k){ return kv(k,"null"); }
  std::string str_obj() const { return "{"+o.str()+"}"; }
};

// ------------------------------------------------------------------ device memory tracking
struct DevTracker { size_t current=0, peak=0; std::map<void*,size_t> sizes; } ;
static DevTracker g_tracker;
static inline void* dmalloc(size_t bytes,const char* what){ void* p=nullptr; cudaError_t e=cudaMalloc(&p,bytes?bytes:1); if(e!=cudaSuccess){ cudaGetLastError(); std::ostringstream o; o<<"cudaMalloc failed for "<<what<<" ("<<bytes<<" bytes): "<<cudaGetErrorString(e); throw MPError(o.str()); } g_tracker.sizes[p]=bytes; g_tracker.current+=bytes; g_tracker.peak=std::max(g_tracker.peak,g_tracker.current); return p; }
static inline void dfree(void* p){ if(!p) return; auto it=g_tracker.sizes.find(p); if(it!=g_tracker.sizes.end()){ g_tracker.current-=it->second; g_tracker.sizes.erase(it);} cudaFree(p); }
template<typename T> struct DBuf { T* p=nullptr; size_t n=0; DBuf(){} explicit DBuf(size_t n_,const char* what="buf"){ alloc(n_,what);} void alloc(size_t n_,const char* what="buf"){ release(); n=n_; p=(T*)dmalloc(n_*sizeof(T),what);} void release(){ if(p){ dfree(p); p=nullptr; n=0;} } ~DBuf(){ release(); } DBuf(const DBuf&)=delete; DBuf& operator=(const DBuf&)=delete; };
