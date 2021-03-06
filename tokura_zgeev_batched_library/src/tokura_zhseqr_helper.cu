#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>



#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuComplex.h>

#include <tokura_blas.h>

void getandset_optimul_method_for_zhseqr(tokurablas_t* handle)
{
	TOKURA_ZHSEQR_BATCHED_WHICHFAST nextHSEQR_method;
	//nextHSEQR_method = handle->zhseqr_fastmethod[handle->zgeev_n];
	nextHSEQR_method = ZHSEQR_NORMAL_MWB;
	switch (nextHSEQR_method)
	{
	case ZHSEQR_NORMAL_MWB:
		handle->nexttmatrixarrange = ElementWise;
		handle->execstatus = TOKURA_BLAS_SUCCESS;
		break;

	default:
		handle->execstatus = TOKURA_BLAS_FAIL;
		break;
	}


}
void tokura_zhseqr_helper(tokurablas_t* handle)
{
	getandset_optimul_method_for_zhseqr(handle);
	if (TOKURA_BLAS_SUCCESS != handle->execstatus)
	{
		return;
	}

	tokura_matrix_rearrangements_helper(handle);
	if (TOKURA_BLAS_SUCCESS != handle->execstatus)
	{
		return;
	}

	//switch (handle->zhseqr_fastmethod[handle->zgeev_n])
	switch (ZHSEQR_NORMAL_MWB)
	{
	case ZHSEQR_NORMAL_MWB:
		tokura_zhseqr_normal_MWB_helper(handle);
		break;


	default:
		break;
	}

	if (TOKURA_BLAS_SUCCESS != handle->execstatus)
	{
		return;
	}
}
