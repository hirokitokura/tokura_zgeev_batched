#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<omp.h>
#include<mkl.h>


#include <random>
#include<tokura_blas.h>

#define STREAM_NUM 8
#define MATRIXNUM_PER_STREAM (1024*8)

#include"get_eig_MKL.c"
int compare_MKL_Complex16(const void* a, const void* b)
{
	MKL_Complex16* tmp_a;
	MKL_Complex16* tmp_b;

	tmp_a = (MKL_Complex16*)a;
	tmp_b = (MKL_Complex16*)b;


	double norm_a;
	double norm_b;

	norm_a = tmp_a->real * tmp_a->real + tmp_a->imag * tmp_a->imag;
	norm_b = tmp_b->real * tmp_b->real + tmp_b->imag * tmp_b->imag;

	//if (norm_a == norm_b)
	{
		if (tmp_a->real == tmp_b->real)
		{
			//	return ((tmp_a->imag) < (tmp_b->imag)) ? -1 : 1;

			if (tmp_a->imag == tmp_b->imag)
			{
				double arg_a = atan2(tmp_a->imag, tmp_a->real);
				double arg_b = atan2(tmp_b->imag, tmp_b->real);

				return ((arg_a) < (arg_b)) ? -1 : 1;
				//return ((arg_a) - (arg_b)) ;
			}
			else
			{
				return ((tmp_a->imag) < (tmp_b->imag)) ? -1 : 1;
				//return ((tmp_a->imag) - (tmp_b->imag)) ;
			}
		}
		else
		{
			return ((tmp_a->real) < (tmp_b->real)) ? -1 : 1;
			//return !((tmp_a->real) - (tmp_b->real)) ;
		}
	}
	//else
	//{
	//	//return (norm_a < norm_b) ? -1 : 1;
	//	return (norm_a - norm_b);
	//}
}
void eigenvalues_sorter
(
	int matrix_size,
	int mat_num,
	MKL_Complex16* myeigenvalues
)
{

	MKL_Complex16* eigenvalues_tmp;
	eigenvalues_tmp = (MKL_Complex16*)malloc(sizeof(MKL_Complex16) * matrix_size);
	for (int k = 0; k < mat_num; k++)
	{
		for (int i = 0; i < matrix_size; i++)
		{
			eigenvalues_tmp[i] = myeigenvalues[i + k * matrix_size];
		}



		qsort(eigenvalues_tmp, matrix_size, sizeof(MKL_Complex16), compare_MKL_Complex16);

		for (int i = 0; i < matrix_size; i++)
		{
			myeigenvalues[i + k * matrix_size] = eigenvalues_tmp[i];
		}
	}
	free(eigenvalues_tmp);
}

double chack_ans
(
	int matrix_size,
	int mat_num,
	MKL_Complex16* myeigenvalues,
	MKL_Complex16* mkleigenvalues
)
{

	int error_mag[16]={0};
	MKL_Complex16* tmp_vec;
	tmp_vec = (MKL_Complex16*)malloc(sizeof(MKL_Complex16) * matrix_size);
	double error_max = -1.0;
	int matrix_index = 0;
	for (int k = 0; k < mat_num; k++)
	{


		double vec_sub = 0.0;
		double vec_ans = 0.0;
		for (int i = 0; i < matrix_size; i++)
		{
			double tmp;
			//差を計算
			tmp = (myeigenvalues[i + k * matrix_size].real - mkleigenvalues[i + k * matrix_size].real) * (myeigenvalues[i + k * matrix_size].real - mkleigenvalues[i + k * matrix_size].real)
				+ (myeigenvalues[i + k * matrix_size].imag - mkleigenvalues[i + k * matrix_size].imag) * (myeigenvalues[i + k * matrix_size].imag - mkleigenvalues[i + k * matrix_size].imag);

			vec_sub = tmp;

			//mkl側

			tmp = mkleigenvalues[i + k * matrix_size].real * mkleigenvalues[i + k * matrix_size].real + mkleigenvalues[i + k * matrix_size].imag * mkleigenvalues[i + k * matrix_size].imag;
			vec_ans = tmp;


			if (sqrt(vec_ans) == 0.0)
			{
				tmp = sqrt(vec_sub);
			}
			else
			{
				tmp = sqrt(vec_sub) / sqrt(vec_ans);
			}
			
			//printf("%e:%d\n",tmp, (int)log10(tmp));

			if(tmp==0.0)
			{
				error_mag[15]++;
			}
			else if((int)log10(tmp)<-15)
			{
				error_mag[15]++;
			}
			else if((int)log10(tmp)>0)
			{
				error_mag[0]++;
			}
			else
			{
				error_mag[(int)fabs(log10(tmp))]++;
			}

			if (error_max < tmp)
			{
				error_max = tmp;
				matrix_index = k;
			}

		}



	}

	printf("MAX ERROR: %e\n", error_max);

	for (int i = 0; i < matrix_size; i++)
	{
		//	printf("MY(%lf,%lf) (%lf,%lf)\n", myeigenvalues[i + matrix_index * matrix_size].real, myeigenvalues[i + matrix_index * matrix_size].imag, mkleigenvalues[i + matrix_index * matrix_size].real, mkleigenvalues[i + matrix_index * matrix_size].imag);
	}

	for (int i = 0; i < 16; i++)
	{
		printf("%03d:%08d\n",-i,error_mag[i]);
	}
	free(tmp_vec);

	return error_max;
}
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

		double gputime;
		double MKLtime;
		printf("GPU: %lf[s]\n", end - start);
		gputime=end - start;

		for(i=0;i<STREAM_NUM;i++)
		{
			cudaFree(mymatrix_input_gpu[i]);
			cudaFree(myeigenvalues_gpu[i]);
			cudaFree(work_gpu[i]);
			cudaFree(flag_gpu[i]);
			
			cudaStreamDestroy(cudastream[i]);

		}
		
		cudaFree(gpumatrix_cpu);
		tokuraDestroy(handle);
		//固有値の方はあとで解放





	MKL_Complex16* mklmatrix_input;

	MKL_Complex16* myeigenvalues;
	MKL_Complex16* mkleigenvalues;

	int alignment=64;
	mklmatrix_input = (MKL_Complex16*)mkl_malloc(sizeof(MKL_Complex16) * matrix_size * matrix_size * mat_num,alignment);

	myeigenvalues = (MKL_Complex16*)mkl_malloc(sizeof(MKL_Complex16) * matrix_size * mat_num,alignment);
	mkleigenvalues = (MKL_Complex16*)mkl_malloc(sizeof(MKL_Complex16) * matrix_size * mat_num,alignment);

		for (i = 0; i < matrix_size * matrix_size * mat_num; i++)
		{
			 mklmatrix_input[i].real=gpumatrix_cpu[i].x ;
			 mklmatrix_input[i].imag=gpumatrix_cpu[i].y ;
		}

	get_eig_MKL(matrix_size, mat_num, mklmatrix_input, mkleigenvalues, &MKLtime);


for (i = 0; i < matrix_size  * mat_num; i++)
		{
		myeigenvalues[i].real=gpueigenvalues_cpu[i].x;
		myeigenvalues[i].imag=gpueigenvalues_cpu[i].y;		
}


	eigenvalues_sorter(matrix_size, mat_num, myeigenvalues);
	eigenvalues_sorter(matrix_size, mat_num, mkleigenvalues);

	double error;
	error = chack_ans(matrix_size, mat_num, myeigenvalues, mkleigenvalues);


	mkl_free(mklmatrix_input);

	mkl_free(myeigenvalues);
	mkl_free(mkleigenvalues);


		cudaFree(gpueigenvalues_cpu);


	{

		FILE* fp;

		fp=fopen("exetime_precision.csv","a");
		if(fp==NULL)
		{
			printf("ERROR\n");
			exit(-1);
		}
		fprintf(fp,"%d,%f,%f,%e\n",matrix_size, gputime, MKLtime, error);
		fclose(fp);

	}

}


int main(void)
{
	int n = 0;
	int mat_num = 16000;
	int i = 0;

	for (i = 1; i <= TOKURA_ZGEEV_BATCHED_MAX_MATRIX_SIZE; i += 1)
	{
		get_eig_main(i, mat_num);
	}


	return 0;
}
