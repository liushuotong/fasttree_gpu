/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Experimental JC posterior. CPU supplies libm-derived transition tables.
   Preserve simple codes/weights: expanding them into vectors changes rounding. */
namespace {
__global__ void jc_posterior_kernel(ProfileDev a, ProfileDev b, int n,
 const unsigned int *cat, int nr, const double *table,
 double *weights, unsigned char *codes, double *vectors) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=n)return;
  int r=cat[i],ca=a.codes[i],cb=b.codes[i];
  double wa=a.weights[i],wb=b.weights[i];
  double sa=table[r],da=table[nr+r],sb=table[2*nr+r],db=table[3*nr+r];
  const double *fa=a.index[i]>=0?a.vectors+4*(size_t)a.index[i]:NULL;
  const double *fb=b.index[i]>=0?b.vectors+4*(size_t)b.index[i]:NULL;
  codes[i]=FT_GPU_NOCODE;weights[i]=1.0;
  for(int j=0;j<4;j++)vectors[(size_t)i*4+j]=0.0;
  if(!fa&&!fb) {
    if(ca==FT_GPU_NOCODE&&cb==FT_GPU_NOCODE){weights[i]=0.0;return;}
    if(ca==FT_GPU_NOCODE){codes[i]=cb;weights[i]=wb*(sb-db);return;}
    if(cb==FT_GPU_NOCODE){codes[i]=ca;weights[i]=wa*(sa-da);return;}
    if(ca==cb){
      codes[i]=ca;
      double x=(wa*sa+(1-wa)*0.25)*(wb*sb+(1-wb)*0.25);
      double y=(wa*da+(1-wa)*0.25)*(wb*db+(1-wb)*0.25);
      double p=x/(x+3*y);
      weights[i]=(p-0.25)*4.0/3.0;
      if(weights[i]<1e-6)weights[i]=1e-6;
      return;
    }
  }
  double ma[4],mb[4];
  if(!fa){for(int j=0;j<4;j++)ma[j]=(1-wa)*0.25;if(ca!=FT_GPU_NOCODE)ma[ca]+=wa;fa=ma;}
  if(!fb){for(int j=0;j<4;j++)mb[j]=(1-wb)*0.25;if(cb!=FT_GPU_NOCODE)mb[cb]+=wb;fb=mb;}
  double sum=0.0;
  for(int j=0;j<4;j++){
    double x=(fa[j]*sa+(1.0-fa[j])*da)*(fb[j]*sb+(1.0-fb[j])*db);
    vectors[(size_t)i*4+j]=x;sum+=x;
  }
  double inv=1.0/sum;
  for(int j=0;j<4;j++)vectors[(size_t)i*4+j]*=inv;
}
}

extern "C" FtGpuStatus ft_gpu_jc_posterior(FtGpuContext *ctx,int n,int nr,
 const FtGpuProfileView *a,const FtGpuProfileView *b,const unsigned int *cat,
 const double *table,double *weights,unsigned char *codes,double *vectors) {
  if(!ctx||!a||!b||!cat||!table||!weights||!codes||!vectors||n<=0||nr<=0)
    return FT_GPU_ERR_INVALID;
  if(nr>64||n>65535*128)return FT_GPU_ERR_UNSUPPORTED;
  FtGpuStatus status=FT_GPU_OK;
  Buf<double> output,transition;
  Buf<unsigned char> chars;
  Buf<unsigned int> categories;
  std::vector<double> host;
  std::vector<unsigned char> hc;
  try {
    cuda_checked(ctx,cudaSetDevice(ctx->deviceId),"JC posterior device");
    if(ctx->fatal)return FT_GPU_ERR_CUDA;
    for(int i=0;i<n;i++)if(cat[i]>=(unsigned int)nr)return FT_GPU_ERR_INVALID;
    for(int i=0;i<4*nr;i++)if(!std::isfinite(table[i])||table[i]<0||table[i]>1)return FT_GPU_ERR_INVALID;
    if(!valid_view(ctx,*a,"JC posterior")||!valid_view(ctx,*b,"JC posterior"))return FT_GPU_ERR_INVALID;
    host.resize((size_t)n*5);hc.resize(n);
    ctx->pinned.clear();
    const Entry *ea=get_profile(ctx,*a,n,4,"JC posterior");
    if(!ea)throw BackendFailure{FT_GPU_ERR_MEMORY};
    ctx->pinned.push_back(ea->key);
    const Entry *eb=get_profile(ctx,*b,n,4,"JC posterior");
    if(!eb)throw BackendFailure{FT_GPU_ERR_MEMORY};
    ctx->pinned.push_back(eb->key);
    if(!output.ensure(ctx,(size_t)n*5,"JC posterior")||!chars.ensure(ctx,n,"JC posterior")||
       !categories.ensure(ctx,n,"JC posterior")||!transition.ensure(ctx,4*nr,"JC posterior"))
      throw BackendFailure{FT_GPU_ERR_MEMORY};
    cuda_checked(ctx,cudaMemcpyAsync(categories.ptr,cat,(size_t)n*sizeof(*cat),cudaMemcpyHostToDevice,ctx->stream),"JC categories");
    cuda_checked(ctx,cudaMemcpyAsync(transition.ptr,table,4*nr*sizeof(double),cudaMemcpyHostToDevice,ctx->stream),"JC transition");
    jc_posterior_kernel<<<(n+127)/128,128,0,ctx->stream>>>(dev_view(ea),dev_view(eb),n,categories.ptr,nr,transition.ptr,output.ptr,chars.ptr,output.ptr+n);
    cuda_checked(ctx,cudaGetLastError(),"JC posterior launch");
    ctx->stats.nKernelLaunches++;
    cuda_checked(ctx,cudaMemcpyAsync(host.data(),output.ptr,(size_t)n*5*sizeof(double),cudaMemcpyDeviceToHost,ctx->stream),"JC posterior values");
    cuda_checked(ctx,cudaMemcpyAsync(hc.data(),chars.ptr,n,cudaMemcpyDeviceToHost,ctx->stream),"JC posterior codes");
    cuda_checked(ctx,cudaStreamSynchronize(ctx->stream),"JC posterior completion");
    for(double x:host)if(!std::isfinite(x))throw BackendFailure{FT_GPU_ERR_INTERNAL};
    memcpy(weights,host.data(),(size_t)n*sizeof(double));
    memcpy(codes,hc.data(),n);
    memcpy(vectors,host.data()+n,(size_t)n*4*sizeof(double));
    ctx->stats.bytesH2D+=(size_t)n*sizeof(*cat)+4*nr*sizeof(double);
    ctx->stats.bytesD2H+=(size_t)n*(5*sizeof(double)+1);
  } catch(const BackendFailure &e){status=e.status;}
    catch(const std::bad_alloc &){status=FT_GPU_ERR_MEMORY;}
    catch(...){status=FT_GPU_ERR_INTERNAL;}
  /* Pending profile uploads may still reference host staging on failures. */
  cudaStreamSynchronize(ctx->stream);
  output.release(ctx);transition.release(ctx);chars.release(ctx);categories.release(ctx);
  ctx->pinned.clear();reset_uploads(ctx);
  return status;
}
