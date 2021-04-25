#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<omp.h>

#include <random>

#include<tokura_blas.h>

#define STREAM_NUM 8
#define MATRIXNUM_PER_STREAM (1024*8)
void set_mat(cuDoubleComplex* mat, const int matrix_size, const int mat_num)
{
	int i = 0;
	int j, k;

	std::random_device seed_gen;
  	std::mt19937 engine(seed_gen());
 	std::uniform_real_distribution<> dist1(-1.0, 1.0);
	for (k = 0; k < mat_num; k++)
	{
		for (i = 0; i < matrix_size; i++)
		{
			for (j = 0; j < matrix_size; j++)
			{
				mat[(j * matrix_size + i) + k * matrix_size * matrix_size].x = dist1(engine);
				mat[(j * matrix_size + i) + k * matrix_size * matrix_size].y = dist1(engine);
			}
		}
	}
	return;
}

void get_eig_main(int matrix_size, int mat_num)
{
	int i;
	double start;
	double end;




	tokurablas_t* handle;
	tokuraCreate(&handle);

	cuDoubleComplex* gpumatrix_cpu;
	cuDoubleComplex* gpueigenvalues_cpu;
	cudaMallocHost((void**)&gpumatrix_cpu, sizeof(cuDoubleComplex) * matrix_size * matrix_size * mat_num);
	cudaMallocHost((void**)&gpueigenvalues_cpu, sizeof(cuDoubleComplex) * matrix_size * mat_num);
	set_mat(gpumatrix_cpu, matrix_size, mat_num);

	//GPU malloc
	cuDoubleComplex* mymatrix_input_gpu[STREAM_NUM];
	cuDoubleComplex* myeigenvalues_gpu[STREAM_NUM];
	cuDoubleComplex* work_gpu[STREAM_NUM];
	char* flag_gpu[STREAM_NUM];

	for(i=0;i<STREAM_NUM;i++)
	{
		cudaMalloc((void**)& mymatrix_input_gpu[i], sizeof(cuDoubleComplex) * matrix_size * matrix_size * MATRIXNUM_PER_STREAM);
		cudaMalloc((void**)& myeigenvalues_gpu[i], sizeof(cuDoubleComplex) * matrix_size * MATRIXNUM_PER_STREAM);
		cudaMalloc((void**)& work_gpu[i], tokura_get_zgeeveigenvaluesgetworspacesize(matrix_size, MATRIXNUM_PER_STREAM));
		cudaMalloc((void**)& flag_gpu[i], sizeof(char) * MATRIXNUM_PER_STREAM);
	}

	cudaStream_t cudastream[STREAM_NUM];
	for(i=0;i<STREAM_NUM;i++)
	{
		cudaStreamCreate(&cudastream[i]);
	}		

	//計算
	int matrix_id=0;
	int stream_id=0;
	start = omp_get_wtime();
	while(matrix_id<mat_num)
	{

		int batchcount_per_stream;

		batchcount_per_stream=matrix_id+MATRIXNUM_PER_STREAM<mat_num?MATRIXNUM_PER_STREAM:mat_num-matrix_id;

		//printf("%d	%d	%d\n",matrix_id, batchcount_per_stream, mat_num);

		//データ転送: H2D
		cudaMemcpyAsync(
				mymatrix_input_gpu[stream_id],
				&gpumatrix_cpu[matrix_size * matrix_size*matrix_id],
				sizeof(cuDoubleComplex) * matrix_size * matrix_size * batchcount_per_stream,
				cudaMemcpyHostToDevice, 
				cudastream[stream_id]
			       );

		tokura_zgeev_batched_gpu
			(
			 handle,
			 matrix_size,
			 batchcount_per_stream,
			 mymatrix_input_gpu[stream_id],
			 myeigenvalues_gpu[stream_id],
			 work_gpu[stream_id],
			 flag_gpu[stream_id],
			 cudastream[stream_id]
			);


		//データ転送: D2H
		cudaMemcpyAsync(
				&gpueigenvalues_cpu[matrix_size*matrix_id],
				myeigenvalues_gpu[stream_id],
				sizeof(cuDoubleComplex) * matrix_size * batchcount_per_stream,
				cudaMemcpyHostToDevice, 
				cudastream[stream_id]
			       );

		stream_id=(stream_id+1)%STREAM_NUM;
		matrix_id+=batchcount_per_stream;
	}



	for(i=0;i<STREAM_NUM;i++)
	{
		cudaStreamSynchronize(cudastream[stream_id]);
	}
	end = omp_get_wtime();
	printf("%d,%lf,[s]\n",matrix_size, end - start);


	for(i=0;i<STREAM_NUM;i++)
	{
		cudaFree(mymatrix_input_gpu[i]);
		cudaFree(myeigenvalues_gpu[i]);
		cudaFree(work_gpu[i]);
		cudaFree(flag_gpu[i]);

		cudaStreamDestroy(cudastream[i]);

	}
	cudaFree(gpumatrix_cpu);
	cudaFree(gpueigenvalues_cpu);
}


int main(void)
{
	int n = 0;
	int mat_num = 50000;
	int i = 0;

	for (i = 64; i <= 64; i += 1)
	{
		get_eig_main(i, mat_num);
	}


	return 0;
}
