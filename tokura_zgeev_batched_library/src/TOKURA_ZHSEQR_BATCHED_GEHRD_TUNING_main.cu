

#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>

#include<float.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuComplex.h>

#include <tokura_blas.h>

#ifndef   TOKURA_ZGEEV_BATCHED_GEHRD_TUNING
#ifdef TOKURA_ZGEEV_BATCHED_HSEQR_TUNING
void set_mat(cuDoubleComplex* mat, const int matrix_size, const int mat_num);

void write_zhseqr_parameters(tokurablas_t* handle)
{
	FILE* fp;
	int i;
	fp = fopen("tokura_zhseqr_parameters.h", "w");
	if (fp == NULL)
	{
		printf("FILE OPEN ERROR\n");
		exit(-1);
	}
	//handle->zgehrd_normal_MWB[handle->zgeev_n]

	//normal HRD
	for (i = 1; i <= TOKURA_ZGEEV_BATCHED_MAX_MATRIX_SIZE; i++)
	{
		fprintf(fp, "#define TOKURA_ZHSEQR_NORMAL_MWB_%d %d\n", i, handle->zhseqr_normal_MWB[i]);
	}

	fprintf(fp, "\n");


	//fast method 
	for (i = 1; i <= TOKURA_ZGEEV_BATCHED_MAX_MATRIX_SIZE; i++)
	{
		fprintf(fp, "#define TOKURA_ZHSEQR_FASTMETHOD_%d %d\n", i, ZHSEQR_NORMAL_MWB);
	}

	fprintf(fp, "\n");


	fclose(fp);
}

void zhseqr_heder_gen()
{
	FILE* fp;
	char filename[1024];
	int i;


	sprintf(filename, "load_zhseqr_normal_MWB_helper.h");
	fp = fopen(filename, "w");
	if (fp == NULL)
	{
		printf("FILE OPEN ERROR\n");
		exit(-1);
	}
	for (i = 1; i <= TOKURA_ZGEEV_BATCHED_MAX_MATRIX_SIZE; i++)
	{
		fprintf(fp, "handle->zhseqr_normal_MWB[%d] = TOKURA_ZHSEQR_NORMAL_MWB_%d;\n", i, i);
	}
	fclose(fp);


	sprintf(filename, "load_zhseqr_fastmethod_helper.h");
	fp = fopen(filename, "w");
	if (fp == NULL)
	{
		printf("FILE OPEN ERROR\n");
		exit(-1);
	}
	for (i = 1; i <= TOKURA_ZGEEV_BATCHED_MAX_MATRIX_SIZE; i++)
	{
		fprintf(fp, "handle->zhseqr_fastmethod[%d] = ZHSEQR_NORMAL_MWB;\n", i);
	}
	fclose(fp);
}

//for tuning of zhseqr
int main(void)
{
	int retval;
	tokurablas_t* handle;
	int batchcount = TOKURA_ZGEEV_BATCHED_GEHRD_TUNING_BATCHCOUNT;


	printf("TUNING of ZGEHRD START\n");
	retval = tokuraCreate(&handle);

	float* normal_MWB_time;
	normal_MWB_time = (float*)malloc(sizeof(float) * (TOKURA_ZGEEV_BATCHED_MAX_MATRIX_SIZE + 1));


	int* normal_MWB_threads;
	normal_MWB_threads = (int*)malloc(sizeof(int) * (TOKURA_ZGEEV_BATCHED_MAX_MATRIX_SIZE + 1));
	memset(normal_MWB_threads, 0, sizeof(int) * (TOKURA_ZGEEV_BATCHED_MAX_MATRIX_SIZE + 1));

	/*時間計測用*/
	cudaEvent_t start, stop;
	float elapsed_time_ms = 0.0f;

	/*時間計測用*/
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int matrix_size;

	for (matrix_size = 1; matrix_size <= TOKURA_ZGEEV_BATCHED_MAX_MATRIX_SIZE; matrix_size++)
	{

		printf("Matrix Size %d * %d\n", matrix_size, matrix_size);
		cuDoubleComplex* matrices_cpu;

		matrices_cpu = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex) * matrix_size * matrix_size * batchcount);
		if (matrices_cpu == NULL)
		{
			printf("Malloc fail\n");
			exit(-1);
		}
		set_mat(matrices_cpu, matrix_size, batchcount);


		//GPU memory allocate
		cuDoubleComplex* matrices_gpu;
		cuDoubleComplex* srcmatrices_gpu;
		cuDoubleComplex* workmatrices_gpu;
		cuDoubleComplex* myeigenvalues;
		char* flag;
		cudaMalloc((void**)& matrices_gpu, sizeof(cuDoubleComplex) * matrix_size * matrix_size * batchcount);
		cudaMalloc((void**)& srcmatrices_gpu, sizeof(cuDoubleComplex) * matrix_size * matrix_size * batchcount);
		cudaMalloc((void**)& workmatrices_gpu, tokura_get_zgeeveigenvaluesgetworspacesize(matrix_size, batchcount));
		cudaMalloc((void**)& myeigenvalues, sizeof(cuDoubleComplex) * matrix_size * batchcount);
		cudaMalloc((void**)& flag, sizeof(char) * batchcount);


		handle->zgeev_n = matrix_size;
		handle->zgeev_batchcount = batchcount;
		handle->zgeev_srcmatrices = srcmatrices_gpu;
		handle->zgeev_wrkmatrices = &workmatrices_gpu[0];
		handle->zgeev_srceigenvalues = &workmatrices_gpu[handle->zgeev_n * handle->zgeev_n * handle->zgeev_batchcount];
		handle->zgeev_wrkeigenvalues = myeigenvalues;
		handle->flags = flag;

		handle->currentmatrixarrange = MatrixWise;
		handle->nexttmatrixarrange = NotSet;
		handle->execstatus = TOKURA_BLAS_SUCCESS;
		handle->stream = NULL;

		//H2D
		cudaMemcpy(matrices_gpu, matrices_cpu, sizeof(cuDoubleComplex) * matrix_size * matrix_size * batchcount, cudaMemcpyHostToDevice);
		cudaMemcpy( handle->zgeev_srcmatrices, matrices_gpu, sizeof(cuDoubleComplex) * matrix_size * matrix_size * batchcount, cudaMemcpyDeviceToDevice);


		tokura_zgehrd_helper(handle);

		if (TOKURA_BLAS_SUCCESS != handle->execstatus)
		{
			return 1;
		}
		cudaMemcpy(matrices_gpu, handle->zgeev_srcmatrices, sizeof(cuDoubleComplex) * matrix_size * matrix_size * batchcount, cudaMemcpyDeviceToDevice);


		int upper_bound;
		int i;
		int threads_per_matrix;

		upper_bound = matrix_size < TARGET_CUDA_WARP ? matrix_size : TARGET_CUDA_WARP;

		upper_bound = upper_bound < 20 ? upper_bound : 20;



		float min_time;
		min_time = FLT_MAX;
		//tokura_zgehrd_shared_MWB_helper
		for (threads_per_matrix = 1; threads_per_matrix <= upper_bound; threads_per_matrix++)
		{
			handle->zhseqr_normal_MWB[handle->zgeev_n] = threads_per_matrix;

			normal_MWB_time[matrix_size] = 0.0;

			for (i = 0; i < TOKURA_ZGEEV_BATCHED_TUNING_COUNT; i++)
			{
				cudaMemcpy(handle->zgeev_srcmatrices, matrices_gpu, sizeof(cuDoubleComplex) * matrix_size * matrix_size * batchcount, cudaMemcpyDeviceToDevice);

				cudaEventRecord(start, 0);

				tokura_zhseqr_normal_MWB_helper(handle);
				cudaDeviceSynchronize();
				{
					cudaError_t err2 = cudaGetLastError();
					if (err2 != cudaSuccess) {
						exit(-1);
					}
				}

				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&elapsed_time_ms, start, stop);
				normal_MWB_time[matrix_size] += elapsed_time_ms;
				printf("%f\n", elapsed_time_ms);
			}
			normal_MWB_time[matrix_size] /= TOKURA_ZGEEV_BATCHED_TUNING_COUNT;
			if (normal_MWB_time[matrix_size] < min_time)
			{
				normal_MWB_threads[matrix_size] = threads_per_matrix;
				min_time = normal_MWB_time[matrix_size];
			}
		//	printf("%d: %f\n",  handle->zhseqr_normal_MWB[handle->zgeev_n], normal_MWB_time[matrix_size]);

		}
		normal_MWB_time[matrix_size] = min_time;
		handle->zhseqr_normal_MWB[handle->zgeev_n] = normal_MWB_threads[matrix_size];
	//	printf("%d %d %f\n", handle->zgeev_n, handle->zhseqr_normal_MWB[handle->zgeev_n], normal_MWB_time[matrix_size]);

		cudaFree(matrices_gpu);
		cudaFree(srcmatrices_gpu);
		cudaFree(workmatrices_gpu);
		cudaFree(myeigenvalues);
		cudaFree(flag);



	}



	write_zhseqr_parameters(handle);

	zhseqr_heder_gen();









	cudaEventDestroy(start);
	cudaEventDestroy(stop);


	free(normal_MWB_time);
	free(normal_MWB_threads);
	tokuraDestroy(handle);
}
#endif
#endif
