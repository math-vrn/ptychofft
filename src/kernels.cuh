#define PI 3.1415926535
void __global__ mul(float2 *g, float2 *f, float2 *prb, int *scanx, int *scany, 
	int Ntheta, int Nz, int N, int Nscanx, int Nscany, int Nprb, int detx, int dety)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int tz = blockDim.z * blockIdx.z + threadIdx.z;

	if (tx>=Nprb*Nprb||ty>=Nscanx*Nscany||tz>=Ntheta) return;
	int ix = tx/Nprb;
	int iy = tx%Nprb;
	int m = ty/Nscany;
	int n = ty%Nscany;

	int stx = scanx[m+tz*Nscanx];
	int sty = scany[n+tz*Nscany];
	if(stx==-1||sty==-1) return;

	int shift = (detx-Nprb)/2*dety+(dety-Nprb)/2;
	float2 f0 = f[(sty+iy)+(stx+ix)*N+tz*Nz*N];
	float2 prb0 = prb[iy+ix*Nprb];
	float c = 1/sqrtf(detx*dety);//fft constant
	g[shift+iy+ix*dety+(n+m*Nscany)*detx*dety+tz*detx*dety*Nscanx*Nscany].x = c*prb0.x*f0.x-c*prb0.y*f0.y;
	g[shift+iy+ix*dety+(n+m*Nscany)*detx*dety+tz*detx*dety*Nscanx*Nscany].y = c*prb0.x*f0.y+c*prb0.y*f0.x;

}

void __global__ mula(float2 *f, float2 *g, float2 *prb, int *scanx, int *scany, 
	int Ntheta, int Nz, int N, int Nscanx, int Nscany, int Nprb, int detx, int dety)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int tz = blockDim.z * blockIdx.z + threadIdx.z;

	if (tx>=Nprb*Nprb||ty>=Nscanx*Nscany||tz>=Ntheta) return;
	int ix = tx/Nprb;
	int iy = tx%Nprb;
	int m = ty/Nscany;
	int n = ty%Nscany;

	int stx = scanx[m+tz*Nscanx];
	int sty = scany[n+tz*Nscany];
	if(stx==-1||sty==-1) return;

	int shift = (detx-Nprb)/2*dety+(dety-Nprb)/2;
	float2 g0 = g[shift+iy+ix*dety+(n+m*Nscany)*detx*dety+tz*detx*dety*Nscanx*Nscany];
	float2 prb0 = prb[iy+ix*Nprb];
	float c = 1/sqrtf(detx*dety);//fft constant
	atomicAdd(&f[(sty+iy)+(stx+ix)*N+tz*Nz*N].x, c*prb0.x*g0.x+c*prb0.y*g0.y);
	atomicAdd(&f[(sty+iy)+(stx+ix)*N+tz*Nz*N].y, c*prb0.x*g0.y-c*prb0.y*g0.x);
}


void __global__ mulamul(float2 *f, float2* ff, float2 *prb, int *scanx, int *scany, 
	int Ntheta, int Nz, int N, int Nscanx, int Nscany, int Nprb, int detx, int dety)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int tz = blockDim.z * blockIdx.z + threadIdx.z;

	if (tx>=Nprb*Nprb||ty>=Nscanx*Nscany||tz>=Ntheta) return;
	int ix = tx/Nprb;
	int iy = tx%Nprb;
	int m = ty/Nscany;
	int n = ty%Nscany;

	int stx = scanx[m+tz*Nscanx];
	int sty = scany[n+tz*Nscany];
	if(stx==-1||sty==-1) return;

	float2 ff0 = ff[(sty+iy)+(stx+ix)*N+tz*Nz*N];
	float prb0 = prb[iy+ix*Nprb].x*prb[iy+ix*Nprb].x+prb[iy+ix*Nprb].y*prb[iy+ix*Nprb].y;
	atomicAdd(&f[(sty+iy)+(stx+ix)*N+tz*Nz*N].x, prb0*ff0.x);
	atomicAdd(&f[(sty+iy)+(stx+ix)*N+tz*Nz*N].y, prb0*ff0.y);
}




void __global__ updateamp(float2 *g, float* data, 
	int Ntheta, int Nz, int N, int Nscanx, int Nscany, int Nprb, int detx, int dety)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int tz = blockDim.z * blockIdx.z + threadIdx.z;

	if (tx>=detx*dety||ty>=Nscanx*Nscany||tz>=Ntheta) return;

	int ind = tx+ty*detx*dety+tz*detx*dety*Nscanx*Nscany;
	float2 g0 = g[ind];
	float ga = sqrtf(data[ind]/(g0.x*g0.x+g0.y*g0.y));
	g[ind].x = g0.x*ga;
	g[ind].y = g0.y*ga;




}
