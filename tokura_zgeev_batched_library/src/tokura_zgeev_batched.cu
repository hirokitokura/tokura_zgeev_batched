#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuComplex.h>

#include <tokura_blas.h>

#ifndef TOKURA_ZGEEV_BATCHED_GEHRD_TUNING
void load_zgehrd_normal_MWB(tokurablas_t* handle)
{

	//�ʏ�HRD����
	#include"load_zgehrd_normal_MWB_helper.h"

	handle->execstatus = TOKURA_BLAS_SUCCESS;

}
void load_zgehrd_shared_MWB(tokurablas_t* handle)
{

	//shared������HRD����
	#include"load_zgehrd_shared_MWB_helper.h"

	handle->execstatus = TOKURA_BLAS_SUCCESS;

}

void load_zgehrd_fastmethod(tokurablas_t* handle)
{

	//�ʏ�HRD����
	#include"load_zgehrd_fastmethod_helper.h"

	handle->execstatus = TOKURA_BLAS_SUCCESS;

}

#endif //TOKURA_ZGEEV_BATCHED_GEHRD_TUNING

#if !defined(TOKURA_ZGEEV_BATCHED_GEHRD_TUNING) && !defined(TOKURA_ZGEEV_BATCHED_HSEQR_TUNING)
void load_zhseqr_normal_MWB(tokurablas_t* handle)
{
	//�ʏ�HRD����
#include"load_zhseqr_normal_MWB_helper.h"

	handle->execstatus = TOKURA_BLAS_SUCCESS;

}


void load_zhseqr_fastmethod(tokurablas_t* handle)
{
#include"load_zgehrd_fastmethod_helper.h"
	handle->execstatus = TOKURA_BLAS_SUCCESS;

}
#endif


int tokuraCreate(tokurablas_t** handle_)
{
	tokurablas_t* handle;

	handle = (tokurablas_t *)malloc(sizeof(tokurablas_t));
	if (handle == NULL)
	{
		return -1;
	}
	*handle_ = handle;

	handle->zgehrd_normal_MWB = (int*)malloc(sizeof(int) * (TOKURA_ZGEEV_BATCHED_MAX_MATRIX_SIZE + 1));
	if ((handle->zgehrd_normal_MWB) == NULL)
	{
		handle->execstatus = TOKURA_BLAS_FAIL;
		return 1;

	}
	handle->zgehrd_shared_MWB = (int*)malloc(sizeof(int) * (TOKURA_ZGEEV_BATCHED_MAX_MATRIX_SIZE + 1));
	if (handle->zgehrd_shared_MWB == NULL)
	{
		handle->execstatus = TOKURA_BLAS_FAIL;
		return 2;

	}
	handle->zgehrd_fastmethod = (TOKURA_ZGEHRD_BATCHED_WHICHFAST*)malloc(sizeof(TOKURA_ZGEHRD_BATCHED_WHICHFAST) * (TOKURA_ZGEEV_BATCHED_MAX_MATRIX_SIZE + 1));
	if (handle->zgehrd_fastmethod == NULL)
	{
		handle->execstatus = TOKURA_BLAS_FAIL;
		return 3;

	}	
	handle->zhseqr_normal_MWB = (int*)malloc(sizeof(int) * (TOKURA_ZGEEV_BATCHED_MAX_MATRIX_SIZE + 1));
	if (handle->zhseqr_normal_MWB == NULL)
	{
		handle->execstatus = TOKURA_BLAS_FAIL;
		return 4;

	}
	handle->zhseqr_fastmethod = (TOKURA_ZHSEQR_BATCHED_WHICHFAST*)malloc(sizeof(TOKURA_ZHSEQR_BATCHED_WHICHFAST) * (TOKURA_ZGEEV_BATCHED_MAX_MATRIX_SIZE + 1));
	if (handle->zhseqr_fastmethod == NULL)
	{
		handle->execstatus = TOKURA_BLAS_FAIL;
		return 5;

	}


#ifndef TOKURA_ZGEEV_BATCHED_GEHRD_TUNING

	load_zgehrd_normal_MWB(handle);
	if (TOKURA_BLAS_SUCCESS != handle->execstatus)
	{
		return 1;
	}

	load_zgehrd_shared_MWB(handle);
	if (TOKURA_BLAS_SUCCESS != handle->execstatus)
	{
		return 1;
	}

	load_zgehrd_fastmethod(handle);
	if (TOKURA_BLAS_SUCCESS != handle->execstatus)
	{
		return 1;
	}
	if (TOKURA_BLAS_SUCCESS != handle->execstatus)
	{
		return 1;
	}
#endif //TOKURA_ZGEEV_BATCHED_GEHRD_TUNING

#if !defined(TOKURA_ZGEEV_BATCHED_GEHRD_TUNING) && !defined(TOKURA_ZGEEV_BATCHED_HSEQR_TUNING)
	load_zhseqr_normal_MWB(handle);
	if (TOKURA_BLAS_SUCCESS != handle->execstatus)
	{
		return 1;
	}

	load_zhseqr_fastmethod(handle);
	if (TOKURA_BLAS_SUCCESS != handle->execstatus)
	{
		return 1;
	}
#endif
	cudaError_t cudaerror;
	cudaerror=cudaGetDeviceProperties(&(handle->dev), 0);

	if (cudaSuccess != cudaerror)
	{
		return 1;
	}

	return 0;

}
int tokuraDestroy(tokurablas_t* handle)
{
	if (handle->zgehrd_normal_MWB != NULL)
	{
		free(handle->zgehrd_normal_MWB);
	}
	if (handle->zgehrd_shared_MWB != NULL)
	{
		free(handle->zgehrd_shared_MWB);
	}
	if (handle->zgehrd_fastmethod != NULL)
	{
		free(handle->zgehrd_fastmethod);
	}
	if (handle->zhseqr_normal_MWB != NULL)
	{
		free(handle->zhseqr_normal_MWB);
	}
	if (handle->zhseqr_fastmethod != NULL)
	{
		free(handle->zhseqr_fastmethod);
	}

	if (handle != NULL)
	{
		free(handle);
	}

	return 0;
}

size_t tokura_get_zgeeveigenvaluesgetworspacesize(int matrix_size, int mat_num)
{
	size_t retval;


	if (!(0 < matrix_size && matrix_size <= TOKURA_ZGEEV_BATCHED_MAX_MATRIX_SIZE))
	{
		return -1;
	}
	if (!(0 < mat_num ))
	{
		return -2;
	}

	retval = sizeof(cuDoubleComplex) * matrix_size * matrix_size * mat_num
		+ sizeof(cuDoubleComplex) * matrix_size * mat_num;

	return retval;
}

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
)
{
	handle->zgeev_n = matrix_size;
	handle->zgeev_batchcount = mat_num;
	handle->zgeev_srcmatrices = mymatrix_input;
	handle->zgeev_wrkmatrices = &work[0];
	handle->zgeev_srceigenvalues = &work[handle->zgeev_n * handle->zgeev_n * handle->zgeev_batchcount];
	handle->zgeev_wrkeigenvalues = myeigenvalues;
	handle->flags = flag;

	handle->currentmatrixarrange = MatrixWise;
	handle->nexttmatrixarrange = NotSet;
	handle->execstatus = TOKURA_BLAS_SUCCESS;
	handle->stream= cudastream;

	if (handle == NULL)
	{
		return -1;
	}
	if (!(handle->zgeev_n > 0 && handle->zgeev_n <= TOKURA_ZGEEV_BATCHED_MAX_MATRIX_SIZE))
	{
		return -2;
	}
	if (handle->zgeev_srcmatrices==NULL)
	{
		return -3;
	}
	if (handle->zgeev_wrkeigenvalues == NULL)
	{
		return -4;
	}
	if (work == NULL)
	{
		return -5;
	}
	if (!(handle->zgeev_batchcount > 0))
	{
		return -6;
	}
	if (handle->flags == NULL)
	{
		return -7;
	}

	tokura_zgehrd_helper(handle);
	if (TOKURA_BLAS_SUCCESS != handle->execstatus)
	{
		return 1;
	}

	tokura_zhseqr_helper(handle);
	if (TOKURA_BLAS_SUCCESS != handle->execstatus)
	{
		return 2;
	}

	tokura_eigenvaluesrearrangement_for_output(handle);
	if (TOKURA_BLAS_SUCCESS != handle->execstatus)
	{
		return 3;
	}

	return 0;
}
