/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Included by ft_gpu_cuda.cu: phase-owned fixed-tree matrix-model evaluation. */
struct FtGpuResidentTree {
  FtGpuContext *ctx;
  int nLeaves, nPos, nJobs, nRates, nCodes, modelCount;
  Buf<unsigned char> leaves, gaps;
  Buf<unsigned int> categories;
  Buf<int> children, order;
  Buf<double> vectors, sites, exponentials, model, result, siteResult;
  std::vector<int> offsets;
  std::vector<double> hostResult, hostSite;
};

namespace {
template<int NC>
__device__ const double *resident_vector(int node, int site, int nLeaves, int nPos,
 const unsigned char *leaves, const unsigned char *gaps, const double *vectors,
 const double *model, bool *gap) {
  if (node < nLeaves) {
    unsigned char ch = leaves[(size_t)node*nPos+site];
    *gap = ch == FT_GPU_NOCODE;
    return model + (*gap ? NC : ch)*NC;
  }
  size_t index = (size_t)(node-nLeaves)*nPos+site;
  *gap = gaps[index] != 0;
  return *gap ? model+NC*NC : vectors+index*NC;
}

template<int NC>
__global__ void resident_level(int nLeaves, int nPos, int nJobs, int nRates,
 const unsigned char *leaves, unsigned char *gaps, double *vectors,
 const unsigned int *categories, const int *children, const int *order, int start,
 const double *exponentials, const double *model, double *sites) {
  int job = order[start+blockIdx.x];
  int site = blockIdx.y*blockDim.x+threadIdx.x;
  if (site >= nPos) return;
  bool gapA, gapB;
  const double *a = resident_vector<NC>(children[job*2],site,nLeaves,nPos,leaves,gaps,vectors,model,&gapA);
  const double *b = resident_vector<NC>(children[job*2+1],site,nLeaves,nPos,leaves,gaps,vectors,model,&gapB);
  size_t index = (size_t)job*nPos+site;
  gaps[index] = gapA && gapB;
  if (gapA && gapB) { sites[index]=1.0; return; }
  unsigned int rate=categories[site];
  const double *ea=exponentials+((size_t)job*3*nRates+rate)*NC;
  const double *eb=ea+nRates*NC;
  const double *ep=eb+nRates*NC;
  double lk=0.0;
  for (int k=0;k<NC;k++) lk += ep[k]*a[k]*b[k];
  sites[index]=lk;
  if (job == nJobs-1) return; /* last root pair has no parent profile */
  double ma[NC],mb[NC],post[NC];
  for(int k=0;k<NC;k++) { ma[k]=a[k]*ea[k];mb[k]=b[k]*eb[k]; }
  for(int j=0;j<NC;j++) {
    double x=0.0,y=0.0;
    for(int k=0;k<NC;k++) { x+=ma[k]*model[j*NC+k]; y+=mb[k]*model[j*NC+k]; }
    double value=x*y*model[(NC+1)*NC+j];
    post[j]=NC==20 ? (value>=0?value:0) : value;
  }
  double total=0.0;
  for(int j=0;j<NC;j++) total+=post[j];
  if (!(total>1.0e-20) || !isfinite(total)) {
    sites[index]=NAN;
    for(int j=0;j<NC;j++) vectors[index*NC+j]=NAN;
    return;
  }
  double inverse=1.0/total;
  for(int j=0;j<NC;j++) post[j]*=inverse;
  constexpr int inv=(NC+1)*NC+NC;
  if constexpr(NC==20) {
    constexpr int nearP=inv+NC*NC,nearFreq=nearP+NC*NC,param=nearFreq+NC*NC;
    int ch=-1;
    if(model[param]==0)for(int j=0;j<NC;j++)if(post[j]>=model[param+1]) {ch=j;break;}
    double w=0.0;
    if(ch>=0) {
      w=(post[ch]-model[nearP+ch*NC+ch])/(1.0-model[nearP+ch*NC+ch]);
      for(int j=0;j<NC;j++)if(j!=ch && (1.0-w)*model[nearP+ch*NC+j]<post[j]*model[param+2]) {ch=-1;break;}
    }
    if(ch>=0) {
      double wInvStat=w*model[(NC+1)*NC+ch];
      for(int j=0;j<NC;j++)vectors[index*NC+j]=wInvStat*model[ch*NC+j]+(1.0-w)*model[nearFreq+ch*NC+j];
      return;
    }
  }
  for(int j=0;j<NC;j++) {
    double sum=0.0;
    for(int k=0;k<NC;k++)sum+=post[k]*model[inv+(NC==4?k*NC+j:j*NC+k)];
    vectors[index*NC+j]=sum;
  }
}

/* Keep the CPU site traversal/product order, including gap skips. Only the
   final log() and postorder node sum run on the host using its libm. */
__global__ void resident_reduce(int nJobs, int nPos, const double *sites,
 const unsigned char *gaps, double *result) {
  int job=blockIdx.x;
  if(job>=nJobs)return;
  double product=1.0,correction=0.0;
  for(int i=0;i<nPos;i++) {
    size_t index=(size_t)job*nPos+i;
    if(gaps[index])continue;
    double x=sites[index];
    if(!(x>0.0)||!isfinite(x)) { product=NAN;break; }
    product*=x;
    if(!(product>0.0)||!isfinite(product)) { product=NAN;break; }
    while(product<kLkUnderflow) { product*=kLkUnderflowInv;correction-=kLogLkUnderflow; }
    while(product>kLkUnderflowInv) { product*=kLkUnderflow;correction+=kLogLkUnderflow; }
  }
  result[2*job]=product;result[2*job+1]=correction;
}
/* Match TreeLogLk's per-site postorder scaling; the final root pair is
   multiplied after the final scaling point in the reference implementation. */
__global__ void resident_site_reduce(int nJobs,int nPos,const double *sites,
 const unsigned char *gaps,double *result) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=nPos)return;
  double product=1.0,correction=0.0;
  for(int j=0;j<nJobs;j++) {
    size_t index=(size_t)j*nPos+i;
    if(!gaps[index])product*=sites[index];
    if(!(product>0.0)||!isfinite(product)) {product=NAN;break;}
    if(j<nJobs-1)while(product<kLkUnderflow) {product*=kLkUnderflowInv;correction-=kLogLkUnderflow;}
  }
  result[2*i]=product;result[2*i+1]=correction;
}

}

extern "C" void ft_gpu_ml_tree_destroy(FtGpuResidentTree *t) {
  if(!t)return;
  cudaSetDevice(t->ctx->deviceId);
  t->leaves.release(t->ctx);t->gaps.release(t->ctx);t->categories.release(t->ctx);
  t->children.release(t->ctx);t->order.release(t->ctx);t->vectors.release(t->ctx);
  t->sites.release(t->ctx);t->exponentials.release(t->ctx);t->model.release(t->ctx);
  t->result.release(t->ctx);t->siteResult.release(t->ctx);delete t;
}

extern "C" FtGpuStatus ft_gpu_ml_tree_create(FtGpuContext *ctx,int nLeaves,int nPos,int nJobs,int nCodes,
 int nRates,const unsigned char *leafCodes,const unsigned int *ratecat,
 const int *children,FtGpuResidentTree **out) {
  if(out)*out=NULL;
  if(!ctx||!out||!leafCodes||!ratecat||!children||nLeaves<3||nPos<=0||nJobs!=nLeaves-1||nRates<=0 || (nCodes!=4&&nCodes!=20))
    return FT_GPU_ERR_INVALID;
  if(nRates>64 || nPos>65535*128 || nLeaves>std::numeric_limits<int>::max()/2)return FT_GPU_ERR_UNSUPPORTED;
  if((size_t)nJobs>SIZE_MAX/(size_t)nPos/sizeof(double)/(size_t)nCodes)return FT_GPU_ERR_OVERFLOW;
  FtGpuResidentTree *t=NULL;FtGpuStatus status=FT_GPU_OK;
  try {
    cuda_checked(ctx,cudaSetDevice(ctx->deviceId),"resident device");
    if(ctx->fatal)return FT_GPU_ERR_CUDA;
    for(size_t i=0;i<(size_t)nLeaves*nPos;i++)
      if(leafCodes[i]>=nCodes&&leafCodes[i]!=FT_GPU_NOCODE)return FT_GPU_ERR_INVALID;
    for(int i=0;i<nPos;i++)if(ratecat[i]>=(unsigned int)nRates)return FT_GPU_ERR_INVALID;
    std::vector<int> levels(nJobs),order;
    int maxLevel=0;
    for(int j=0;j<nJobs;j++) {
      int level=0;
      for(int side=0;side<2;side++) {
        int child=children[2*j+side];
        if(child<0||child>=nLeaves+j)return FT_GPU_ERR_INVALID;
        if(child>=nLeaves)level=std::max(level,levels[child-nLeaves]+1);
      }
      levels[j]=level;maxLevel=std::max(level,maxLevel);
    }
    t=new FtGpuResidentTree();t->ctx=ctx;
    t->nLeaves=nLeaves;t->nPos=nPos;t->nJobs=nJobs;t->nRates=nRates;t->nCodes=nCodes;t->modelCount=nCodes==4?40:1643;
    for(int level=0;level<=maxLevel;level++) {
      t->offsets.push_back((int)order.size());
      for(int j=0;j<nJobs;j++)if(levels[j]==level)order.push_back(j);
    }
    t->offsets.push_back(nJobs);t->hostResult.resize((size_t)nJobs*2);t->hostSite.resize((size_t)nPos*2);
#define RESERVE(field,count) if(!t->field.ensure(ctx,(count),"resident allocation")) throw BackendFailure{FT_GPU_ERR_MEMORY}
    RESERVE(leaves,(size_t)nLeaves*nPos);RESERVE(gaps,(size_t)nJobs*nPos);
    RESERVE(vectors,(size_t)nJobs*nPos*nCodes);RESERVE(sites,(size_t)nJobs*nPos);
    RESERVE(categories,(size_t)nPos);RESERVE(children,(size_t)nJobs*2);
    RESERVE(order,(size_t)nJobs);RESERVE(exponentials,(size_t)nJobs*3*nRates*nCodes);
    RESERVE(model,t->modelCount);RESERVE(result,(size_t)nJobs*2);RESERVE(siteResult,(size_t)nPos*2);
#undef RESERVE
#define COPY(field,source,count) cuda_checked(ctx,cudaMemcpy(t->field.ptr,(source),(count)*sizeof(*t->field.ptr),cudaMemcpyHostToDevice),"resident upload")
    COPY(leaves,leafCodes,(size_t)nLeaves*nPos);COPY(categories,ratecat,(size_t)nPos);
    COPY(children,children,(size_t)nJobs*2);COPY(order,order.data(),(size_t)nJobs);
#undef COPY
    ctx->stats.bytesH2D+=(size_t)nLeaves*nPos+(size_t)nPos*sizeof(unsigned int)+(size_t)nJobs*3*sizeof(int);
    *out=t;return FT_GPU_OK;
  } catch(const BackendFailure &e) { status=e.status; }
    catch(const std::bad_alloc &) { status=FT_GPU_ERR_MEMORY; }
    catch(...) { status=FT_GPU_ERR_INTERNAL; }
  cudaStreamSynchronize(ctx->stream);ft_gpu_ml_tree_destroy(t);return status;
}

extern "C" FtGpuStatus ft_gpu_ml_tree_eval(FtGpuResidentTree *t,const double *expTables,
 const double *model,double *out,double *outSite) {
  if(!t||!expTables||!model||!out)return FT_GPU_ERR_INVALID;
  FtGpuContext *ctx=t->ctx;FtGpuStatus status=FT_GPU_OK;
  int64_t start=now_us();
  try {
    cuda_checked(ctx,cudaSetDevice(ctx->deviceId),"resident device");
    if(ctx->fatal)return FT_GPU_ERR_CUDA;
    size_t nExp=(size_t)t->nJobs*3*t->nRates*t->nCodes;
    for(size_t i=0;i<nExp;i++)if(!std::isfinite(expTables[i])||expTables[i]<0)return FT_GPU_ERR_INVALID;
    for(int i=0;i<t->modelCount;i++)if(!std::isfinite(model[i]))return FT_GPU_ERR_INVALID;
    cuda_checked(ctx,cudaMemcpyAsync(t->exponentials.ptr,expTables,nExp*sizeof(double),cudaMemcpyHostToDevice,ctx->stream),"resident exponentials");
    cuda_checked(ctx,cudaMemcpyAsync(t->model.ptr,model,t->modelCount*sizeof(double),cudaMemcpyHostToDevice,ctx->stream),"resident model");
    for(size_t level=0;level+1<t->offsets.size();level++) {
      dim3 grid(t->offsets[level+1]-t->offsets[level],(t->nPos+127)/128);
      if(t->nCodes==4) {
      resident_level<4><<<grid,128,0,ctx->stream>>>(t->nLeaves,t->nPos,t->nJobs,t->nRates,
       t->leaves.ptr,t->gaps.ptr,t->vectors.ptr,t->categories.ptr,t->children.ptr,t->order.ptr,t->offsets[level],
       t->exponentials.ptr,t->model.ptr,t->sites.ptr);
      } else {
      resident_level<20><<<grid,128,0,ctx->stream>>>(t->nLeaves,t->nPos,t->nJobs,t->nRates,
       t->leaves.ptr,t->gaps.ptr,t->vectors.ptr,t->categories.ptr,t->children.ptr,t->order.ptr,t->offsets[level],
       t->exponentials.ptr,t->model.ptr,t->sites.ptr);
      }
      cuda_checked(ctx,cudaGetLastError(),"resident posterior launch");
      ctx->stats.nKernelLaunches++;
    }
    resident_reduce<<<t->nJobs,1,0,ctx->stream>>>(t->nJobs,t->nPos,t->sites.ptr,t->gaps.ptr,t->result.ptr);
    cuda_checked(ctx,cudaGetLastError(),"resident reduction launch");ctx->stats.nKernelLaunches++;
    cuda_checked(ctx,cudaMemcpyAsync(t->hostResult.data(),t->result.ptr,t->nJobs*2*sizeof(double),cudaMemcpyDeviceToHost,ctx->stream),"resident result");
    if(outSite) {
      resident_site_reduce<<<(t->nPos+127)/128,128,0,ctx->stream>>>(t->nJobs,t->nPos,t->sites.ptr,t->gaps.ptr,t->siteResult.ptr);
      cuda_checked(ctx,cudaGetLastError(),"resident site reduction");ctx->stats.nKernelLaunches++;
      cuda_checked(ctx,cudaMemcpyAsync(t->hostSite.data(),t->siteResult.ptr,(size_t)t->nPos*2*sizeof(double),cudaMemcpyDeviceToHost,ctx->stream),"resident site result");
    }
    cuda_checked(ctx,cudaStreamSynchronize(ctx->stream),"resident completion");
    double total=0.0;
    for(int j=0;j<t->nJobs;j++) {
      double product=t->hostResult[j*2],correction=t->hostResult[j*2+1];
      if(!(product>0.0)||!std::isfinite(product)||!std::isfinite(correction))return FT_GPU_ERR_INTERNAL;
      total+=correction+std::log(product);
    }
    if(!std::isfinite(total))return FT_GPU_ERR_INTERNAL;
    if(outSite)for(int i=0;i<t->nPos;i++) {
      double product=t->hostSite[2*i],correction=t->hostSite[2*i+1];
      if(!(product>0.0)||!std::isfinite(product)||!std::isfinite(correction))return FT_GPU_ERR_INTERNAL;
      double value=correction+std::log(product);
      if(!std::isfinite(value))return FT_GPU_ERR_INTERNAL;
      t->hostSite[i]=value;
    }
    ctx->stats.nResidentEvaluations++;
    ctx->stats.nResidentNodes+=t->nJobs-1;
    ctx->stats.bytesH2D+=(nExp+t->modelCount)*sizeof(double);
    ctx->stats.bytesD2H+=(size_t)t->nJobs*2*sizeof(double);
    ctx->stats.tResident+=(now_us()-start)*1e-6;
    if(outSite) { std::memcpy(outSite,t->hostSite.data(),(size_t)t->nPos*sizeof(double));
      ctx->stats.bytesD2H+=(size_t)t->nPos*2*sizeof(double); }
    *out=total;return FT_GPU_OK;
  } catch(const BackendFailure &e) { status=e.status; }
    catch(const std::bad_alloc &) { status=FT_GPU_ERR_MEMORY; }
    catch(...) { status=FT_GPU_ERR_INTERNAL; }
  cudaStreamSynchronize(ctx->stream);return status;
}
