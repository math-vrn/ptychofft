#include <cufft.h>

class ptychofft
{
	size_t N;
	size_t Ntheta;
	size_t Nz;
	size_t Nscanx;
	size_t Nscany;
	size_t detx;
	size_t dety;
	size_t Nprb;
	
	float2* f;
	float2* g;
	float2* prb; 
	int* scanx; 
	int* scany; 
	float2* ff;
	float* data;

	cufftHandle plan2dfwd;
	cufftHandle plan2dadj;

public:
	ptychofft(size_t Ntheta, size_t Nz, size_t N, 
		size_t Nscanx, size_t Nscany, size_t detx, size_t dety, size_t Nprb);
	~ptychofft();
	void setobjc(int* scanx_, int* scany_, float2* prb_);
	void fwdc(float2* g_, float2* f_);
	void adjc(float2* f_, float2* g_);
	void adjfwd_prbc(float2* f_, float2* ff_);
	void update_ampc(float2* g_,float* data_);

	// python wrap
	void setobj(
			int* scanx_, int N30, int N31,
			int* scany_, int N40, int n41,
			float2* prb_, int N50, int N51);

	void fwd(float2* g_, int N00, int N01, int N02, int N03,
			float2* f_, int N10, int N11, int N12);
	
	void adj(float2* f_, int N10, int N11, int N12,
			float2* g_, int N00, int N01, int N02, int N03);

	void adjfwd_prb(float2* f_, int N10, int N11, int N12, 
			float2* ff_, int N60, int N61, int N62);

	void update_amp(float2* g_, int N00, int N01, int N02, int N03,
			float* data_, int N70, int N71, int N72, int N73);
};

