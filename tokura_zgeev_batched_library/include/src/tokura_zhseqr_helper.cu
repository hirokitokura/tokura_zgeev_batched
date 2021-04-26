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

	//	printf("getandset_optimul_method_for_zhseqr:ZHSEQR_NORMAL_MWB: %d -> %d\n", handle->currentmatrixarrange, handle->nexttmatrixarrange);

		break;


	default:
		//printf("getandset_optimul_method_for_zhseqr: %d\n", nextHSEQR_method);
		handle->execstatus = TOKURA_BLAS_FAIL;
		break;
	}


}
void tokura_zhseqr_helper(tokurablas_t* handle)
{
//	printf("getandset_optimul_method_for_zhseqr start\n");
//	printf("IN getandset_optimul_method_for_zhseqr:ZHSEQR_NORMAL_MWB: %d -> %d\n", handle->currentmatrixarrange, handle->nexttmatrixarrange);

	getandset_optimul_method_for_zhseqr(handle);
	if (TOKURA_BLAS_SUCCESS != handle->execstatus)
	{
		return;
	}
//	printf("tokura_matrix_rearrangements_helper start\n");

	tokura_matrix_rearrangements_helper(handle);
	if (TOKURA_BLAS_SUCCESS != handle->execstatus)
	{
	//	printf("tokura_matrix_rearrangements_helper ERROR\n");

		return;
	}

	//switch (handle->zhseqr_fastmethod[handle->zgeev_n])
	switch (ZHSEQR_NORMAL_MWB)
	{
	case ZHSEQR_NORMAL_MWB:
		//printf("tokura_zhseqr_normal_MWB_helper \n");

		tokura_zhseqr_normal_MWB_helper(handle);
		break;


	default:
		//	printf("tokura_matrix_rearrangements_helper ERROR\n");

		break;
	}

	if (TOKURA_BLAS_SUCCESS != handle->execstatus)
	{
		return;
	}
}
