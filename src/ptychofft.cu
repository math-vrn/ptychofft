#include "ptychofft.cuh"
#include "kernels.cuh"
#include <stdio.h>

ptychofft::ptychofft(size_t Ntheta_, size_t Nz_, size_t N_, 
	size_t Nscanx_, size_t Nscany_, size_t detx_, size_t dety_, size_t Nprb_)
{
	N = N_;
	Ntheta = Ntheta_;
	Nz = Nz_;
	Nscanx = Nscanx_;
	Nscany = Nscany_;
	detx = detx_;
	dety = dety_;
	Nprb = Nprb_;

	cudaMalloc((void**)&f,Ntheta*Nz*N*sizeof(float2));
	cudaMalloc((void**)&g,Ntheta*Nscanx*Nscany*detx*dety*sizeof(float2));
	cudaMalloc((void**)&scanx,Ntheta*Nscanx*sizeof(int));
	cudaMalloc((void**)&scany,Ntheta*Nscany*sizeof(int));
	cudaMalloc((void**)&prb,Nprb*Nprb*sizeof(float2));
	cudaMalloc((void**)&ff,Ntheta*Nz*N*sizeof(float2));
	cudaMalloc((void**)&data,Ntheta*Nscanx*Nscany*detx*dety*sizeof(float));

	int ffts[2];
	int idist;int odist;
	int inembed[2];int onembed[2];
	ffts[0] = detx; ffts[1] = dety;
	idist = detx*dety; odist = detx*dety;
	inembed[0] = detx; inembed[1] = dety;
	onembed[0] = detx; onembed[1] = dety;
	cufftPlanMany(&plan2dfwd, 2, ffts, inembed, 1, idist, onembed, 1, odist, CUFFT_C2C, Ntheta*Nscanx*Nscany); 
	fprintf(stderr,"ptycho created %d angles\n",Ntheta);
}

ptychofft::~ptychofft()
{	
	cudaFree(f);
	cudaFree(g);
	cudaFree(scanx);
	cudaFree(scany);
	cudaFree(prb);
	cudaFree(ff);
	cudaFree(data);
	cufftDestroy(plan2dfwd);
	fprintf(stderr,"ptycho removed\n");
}

void ptychofft::setobjc(int* scanx_, int* scany_, float2* prb_)
{
	cudaMemcpy(scanx,scanx_,Ntheta*Nscanx*sizeof(int),cudaMemcpyDefault);  	
	cudaMemcpy(scany,scany_,Ntheta*Nscany*sizeof(int),cudaMemcpyDefault);  	
	cudaMemcpy(prb,prb_,Nprb*Nprb*sizeof(float2),cudaMemcpyDefault);
}

void ptychofft::fwdc(float2* g_, float2* f_)
{
	dim3 BS3d(32,32,1);
	dim3 GS3d0(ceil(Nprb*Nprb/(float)BS3d.x),ceil(Nscanx*Nscany/(float)BS3d.y),ceil(Ntheta/(float)BS3d.z));

	cudaMemcpy(f,f_,Ntheta*Nz*N*sizeof(float2),cudaMemcpyDefault);
	cudaMemset(g,0,Ntheta*Nscanx*Nscany*detx*dety*sizeof(float2));

	mul<<<GS3d0,BS3d>>>(g,f,prb,scanx,scany,Ntheta,Nz,N,Nscanx,Nscany,Nprb,detx,dety);
	cufftExecC2C(plan2dfwd, (cufftComplex*)g,(cufftComplex*)g,CUFFT_FORWARD);

	cudaMemcpy(g_,g,Ntheta*Nscanx*Nscany*detx*dety*sizeof(float2),cudaMemcpyDefault);  	
}

void ptychofft::adjc(float2* f_, float2* g_)
{
	dim3 BS3d(32,32,1);
	dim3 GS3d0(ceil(Nprb*Nprb/(float)BS3d.x),ceil(Nscanx*Nscany/(float)BS3d.y),ceil(Ntheta/(float)BS3d.z));

	cudaMemcpy(g,g_,Ntheta*Nscanx*Nscany*detx*dety*sizeof(float2),cudaMemcpyDefault);  	
	cudaMemset(f,0,Ntheta*Nz*N*sizeof(float2));

	cufftExecC2C(plan2dfwd, (cufftComplex*)g,(cufftComplex*)g,CUFFT_INVERSE);
	mula<<<GS3d0,BS3d>>>(f,g,prb,scanx,scany,Ntheta,Nz,N,Nscanx,Nscany,Nprb,detx,dety);

	cudaMemcpy(f_,f,Ntheta*Nz*N*sizeof(float2),cudaMemcpyDefault);  	
}

void ptychofft::adjfwd_prbc(float2* f_, float2* ff_)
{
	dim3 BS3d(32,32,1);
	dim3 GS3d0(ceil(Nprb*Nprb/(float)BS3d.x),ceil(Nscanx*Nscany/(float)BS3d.y),ceil(Ntheta/(float)BS3d.z));

	cudaMemset(f,0,Ntheta*Nz*N*sizeof(float2));

	cudaMemcpy(ff,ff_,Ntheta*Nz*N*sizeof(float2),cudaMemcpyDefault);
	mulamul<<<GS3d0,BS3d>>>(f,ff,prb,scanx,scany,Ntheta,Nz,N,Nscanx,Nscany,Nprb,detx,dety);

	cudaMemcpy(f_,f,Ntheta*Nz*N*sizeof(float2),cudaMemcpyDefault);  	
}

void ptychofft::update_ampc(float2* g_, float* data_)
{
	dim3 BS3d(32,32,1);
	dim3 GS3d0(ceil(detx*dety/(float)BS3d.x),ceil(Nscanx*Nscany/(float)BS3d.y),ceil(Ntheta/(float)BS3d.z));

	cudaMemcpy(g,g_,Ntheta*Nscanx*Nscany*detx*dety*sizeof(float2),cudaMemcpyDefault);
	cudaMemcpy(data,data_,Ntheta*Nscanx*Nscany*detx*dety*sizeof(float),cudaMemcpyDefault);
	updateamp<<<GS3d0,BS3d>>>(g,data,Ntheta,Nscanx*Nscany,detx*dety);
	cudaMemcpy(g_,g,Ntheta*Nscanx*Nscany*detx*dety*sizeof(float2),cudaMemcpyDefault);  	
}






void ptychofft::setobj(int* scanx_, int N30, int N31,
					int* scany_, int N40, int n41,
					float* prb_, int N50, int N51)
{
	setobjc(scanx_, scany_, (float2*)prb_);
}

void ptychofft::fwd(float* g_, int N00, int N01, int N02, int N03,
					float* f_, int N10, int N11, int N12)	
{
	fwdc((float2*)g_, (float2*)f_);
}

void ptychofft::adj(float* f_, int N10, int N11, int N12,
					float* g_, int N00, int N01, int N02, int N03)	
{
	adjc((float2*)f_, (float2*)g_);
}

void ptychofft::adjfwd_prb(float* f_, int N10, int N11, int N12, float* ff_, int N60, int N61, int N62)
{
	adjfwd_prbc((float2*)f_,(float2*)ff_);
}


void ptychofft::update_amp(float* g_, int N00, int N01, int N02, int N03,
	float* data_, int N70, int N71, int N72, int N73)
{
	update_ampc((float2*)g_,data_);
}








