//#define TOKURA_ZGEEV_BATCHED_GEHRD_TUNING
//#define TOKURA_ZGEEV_BATCHED_HSEQR_TUNING

#ifndef TOKURABLASINCLUDEGUARD_TOKURA_BLAS
#define TOKURABLASINCLUDEGUARD_TOKURA_BLAS

#include<cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuComplex.h>

#define TOKURA_ZGEEV_BATCHED_MAX_MATRIX_SIZE 64
#define TARGET_CUDA_WARP 32


#define TOKURA_ZGEEV_BATCHED_GEHRD_TUNING_BATCHCOUNT (1024*8)
#define TOKURA_ZGEEV_BATCHED_HSEQR_TUNING_BATCHCOUNT (1024*8)
#define TOKURA_ZGEEV_BATCHED_TUNING_COUNT 1
typedef enum
{
	TOKURA_BLAS_SUCCESS=0,
	TOKURA_BLAS_FAIL
}TOKURA_ZGEEV_BATCHED_STATUS;

typedef enum
{
	MatrixWise = 0,
	ElementWise,
	RowWise,


	NotSet
}TOKURA_ZGEEV_BATCHED_MATRIX_ARRANGEMENT;


typedef enum
{
	ZGEHRD_NORMAL_MWB_METHOD = 0,
	ZGEHRD_SHARED_MWB_METHOD
}TOKURA_ZGEHRD_BATCHED_WHICHFAST;



typedef enum
{
	ZHSEQR_NORMAL_MWB = 0,

}TOKURA_ZHSEQR_BATCHED_WHICHFAST;


#ifndef TOKURA_ZGEEV_BATCHED_GEHRD_TUNING
#include"tokura_zgehrd_parameters.h"
#endif

#if !defined(TOKURA_ZGEEV_BATCHED_GEHRD_TUNING) && !defined(TOKURA_ZGEEV_BATCHED_HSEQR_TUNING)
#include"tokura_zhseqr_parameters.h"
#endif

typedef struct tokurablas_t
{
	cudaStream_t stream;
	char* flags;
	cudaDeviceProp dev;

	//matrix arrangement
	TOKURA_ZGEEV_BATCHED_MATRIX_ARRANGEMENT currentmatrixarrange;
	TOKURA_ZGEEV_BATCHED_MATRIX_ARRANGEMENT nexttmatrixarrange;

	//
	TOKURA_ZGEEV_BATCHED_STATUS execstatus;



	//for zgeev
	int zgeev_n;
	int zgeev_batchcount;
	cuDoubleComplex* zgeev_srcmatrices;
	cuDoubleComplex* zgeev_wrkmatrices;
	cuDoubleComplex* zgeev_srceigenvalues;
	cuDoubleComplex* zgeev_wrkeigenvalues;
	
	//parameters for normal HDR method
	int* zgehrd_normal_MWB;

	//parameters for shared HDR method
	int* zgehrd_shared_MWB;

	//select method for HRD
	TOKURA_ZGEHRD_BATCHED_WHICHFAST* zgehrd_fastmethod;

	//select matrix arragement for HRD
	//int* zgehrd_matrixarrange;

	//parameters for hseqr method
	int* zhseqr_normal_MWB;

	TOKURA_ZHSEQR_BATCHED_WHICHFAST* zhseqr_fastmethod;


}tokurablas_t;





#ifdef WIN64
__declspec(dllexport) int tokura_zgeev_batched_gpu
(
	tokurablas_t* handle,
	int matrix_size,
	int mat_num,
	cuDoubleComplex* mymatrix_input,
	cuDoubleComplex* myeigenvalues,
	cuDoubleComplex* work,
	char* flag,
	cudaStream_t cudastream
);
__declspec(dllexport) size_t tokura_get_zgeeveigenvaluesgetworspacesize(int matrix_size, int mat_num);
__declspec(dllexport) void tokura_matrix_rearrangements_helper(tokurablas_t* handle);
__declspec(dllexport) void tokura_zgehrd_normal_MWB_helper(tokurablas_t* handle);
__declspec(dllexport) void tokura_zgehrd_shared_MWB_helper(tokurablas_t* handle);
__declspec(dllexport) void tokura_zgehrd_helper(tokurablas_t* handle);
__declspec(dllexport) void tokura_zhseqr_helper(tokurablas_t* handle);
__declspec(dllexport) void tokura_zhseqr_normal_MWB_helper(tokurablas_t* handle);
__declspec(dllexport) void tokura_eigenvaluesrearrangement_for_output(tokurablas_t* handle);
__declspec(dllexport) int tokuraCreate(tokurablas_t** handle_);
__declspec(dllexport) int tokuraDestroy(tokurablas_t* handle);
#endif
#ifdef __unix__
int tokura_zgeev_batched_gpu
(
	tokurablas_t* handle,
	int matrix_size,
	int mat_num,
	cuDoubleComplex* mymatrix_input,
	cuDoubleComplex* myeigenvalues,
	cuDoubleComplex* work,
	char* flag,
	cudaStream_t cudastream
);
size_t tokura_get_zgeeveigenvaluesgetworspacesize(int matrix_size, int mat_num);
void tokura_matrix_rearrangements_helper(tokurablas_t* handle);
void tokura_zgehrd_normal_MWB_helper(tokurablas_t* handle);
void tokura_zgehrd_shared_MWB_helper(tokurablas_t* handle);
void tokura_zgehrd_helper(tokurablas_t* handle);
void tokura_zhseqr_helper(tokurablas_t* handle);
void tokura_zhseqr_normal_MWB_helper(tokurablas_t* handle);
void tokura_eigenvaluesrearrangement_for_output(tokurablas_t* handle);
int tokuraCreate(tokurablas_t** handle_);
int tokuraDestroy(tokurablas_t* handle);
#endif
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		//if (abort) exit(code);
	}
}



#endif

