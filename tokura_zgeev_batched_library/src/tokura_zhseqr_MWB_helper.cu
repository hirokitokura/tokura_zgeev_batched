#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>



#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuComplex.h>

#include <tokura_blas.h>

#include"tokura_zhseqr_MWB_kernel.cuh"

void tokura_zhseqr_normal_MWB_helper(tokurablas_t* handle)
{


	dim3 thread_num(TARGET_CUDA_WARP, handle->zhseqr_normal_MWB[handle->zgeev_n]);
	dim3 block_num((handle->zgeev_batchcount + thread_num.x - 1) / thread_num.x);
	//printf("%d %d %d\n", thread_num.x, thread_num.y, thread_num.z);

	int shared_size = sizeof(int) * (handle->zgeev_n + 1) * thread_num.x
		+ sizeof(int) * thread_num.x;
	//printf("tokura_zhseqr_normal_MWB_kernel start\n");

	if (handle->stream == NULL)
	{
		tokura_zhseqr_normal_MWB_kernel
			<< < block_num, thread_num, shared_size >> >
			(
				handle->zgeev_n,
				handle->zgeev_batchcount,
				handle->zgeev_srcmatrices,
				handle->zgeev_srceigenvalues,
				handle->flags
				//	MWB_reduction_thread_num
				);
	}
	else
	{
		tokura_zhseqr_normal_MWB_kernel
			<< < block_num, thread_num, shared_size, handle->stream >> >
			(
				handle->zgeev_n,
				handle->zgeev_batchcount,
				handle->zgeev_srcmatrices,
				handle->zgeev_srceigenvalues,
				handle->flags
				//	MWB_reduction_thread_num
				);
	}
	

	cudaDeviceSynchronize();
	{
		cudaError_t err2 = cudaGetLastError();
		if (err2 != cudaSuccess) {
			gpuErrchk(err2);
		}
	}
	handle->currentmatrixarrange = handle->nexttmatrixarrange;
	handle->nexttmatrixarrange = NotSet;
	handle->execstatus = TOKURA_BLAS_SUCCESS;



}
