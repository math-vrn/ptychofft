/*interface*/
%module ptychofft

%{
#define SWIG_FILE_WITH_INIT
#include "ptychofft.cuh"
%}

%include "numpy.i"

%init %{
import_array();
%}
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
	void update_ampc(float2* f_, float* data_);


	// python wrap

	%apply (float *IN_ARRAY1, int DIM1) {(float* theta_, int N20)};
	%apply (int *IN_ARRAY2, int DIM1, int DIM2) {(int* scanx_, int N30, int N31)};
	%apply (int *IN_ARRAY2, int DIM1, int DIM2) {(int* scany_, int N40, int n41)};
	%apply (float *IN_ARRAY2, int DIM1, int DIM2) {(float* prb_, int N50, int N51)};
	
	void setobj(
			int* scanx_, int N30, int N31,
			int* scany_, int N40, int n41,
			float* prb_, int N50, int N51);

    %apply (float *INPLACE_ARRAY4, int DIM1, int DIM2, int DIM3, int DIM4) {(float* g_, int N00, int N01, int N02, int N03)};
	%apply (float *IN_ARRAY3, int DIM1, int DIM2, int DIM3) {(float* f_, int N10, int N11, int N12)};
	void fwd(float* g_, int N00, int N01, int N02, int N03,
			float* f_, int N10, int N11, int N12);
	%clear (float* g_, int N00, int N01, int N02, int N03);
	%clear (float* f_, int N10, int N11, int N12);

    %apply (float *IN_ARRAY4, int DIM1, int DIM2, int DIM3, int DIM4) {(float* g_, int N00, int N01, int N02, int N03)};
	%apply (float *INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) {(float* f_, int N10, int N11, int N12)};
	void adj(float* f_, int N10, int N11, int N12,
			float* g_, int N00, int N01, int N02, int N03);
	%clear (float* g_, int N00, int N01, int N02, int N03);
	%clear (float* f_, int N10, int N11, int N12);

	%apply (float *INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) {(float* f_, int N10, int N11, int N12)};
	%apply (float *IN_ARRAY3, int DIM1, int DIM2, int DIM3) {(float* ff_, int N60, int N61, int N62)};
	void adjfwd_prb(float* f_, int N10, int N11, int N12,
			float* ff_, int N60, int N61, int N62);
	%clear (float* f_, int N10, int N11, int N12);
	%clear (float* ff_, int N60, int N61, int N62);

    %apply (float *INPLACE_ARRAY4, int DIM1, int DIM2, int DIM3, int DIM4) {(float* g_, int N00, int N01, int N02, int N03)};	
	%apply (float *IN_ARRAY4, int DIM1, int DIM2, int DIM3, int DIM4) {(float* data_, int N70, int N71, int N72, int N73)};	
	void update_amp(float* g_, int N00, int N01, int N02, int N03,
			float* data_, int N70, int N71, int N72, int N73);
	%clear (float* g_, int N00, int N01, int N02, int N03);
	%clear (float* data_, int N70, int N71, int N72, int N73);



};


