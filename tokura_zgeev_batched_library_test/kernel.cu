#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<omp.h>
#include<mkl.h>

#include <random>

#include<tokura_blas.h>

/*extern void get_eig_MKL
(
	int matrix_size,
	int mat_num,
	MKL_Complex16* mklmatrix_input,
	MKL_Complex16* mkleigenvalues,
	double* time
);*/

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

void chack_ans
(
	int matrix_size,
	int mat_num,
	MKL_Complex16* myeigenvalues,
	MKL_Complex16* mkleigenvalues
)
{
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
			//·‚ðŒvŽZ
			tmp = (myeigenvalues[i + k * matrix_size].real - mkleigenvalues[i + k * matrix_size].real) * (myeigenvalues[i + k * matrix_size].real - mkleigenvalues[i + k * matrix_size].real)
				+ (myeigenvalues[i + k * matrix_size].imag - mkleigenvalues[i + k * matrix_size].imag) * (myeigenvalues[i + k * matrix_size].imag - mkleigenvalues[i + k * matrix_size].imag);

			vec_sub = tmp;

			//mkl‘¤

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
	free(tmp_vec);
}


void set_mat(MKL_Complex16* mat, const int matrix_size, const int mat_num)
{
	int i = 0;
	int j, k;
	double tmp;

	std::random_device seed_gen;
	std::mt19937 engine(seed_gen());
        std::uniform_real_distribution<> dist1(-1.0, 1.0);

	for (k = 0; k < mat_num; k++)
	{
		for (i = 0; i < matrix_size; i++)
		{
			for (j = 0; j < matrix_size; j++)
			{
				mat[(j * matrix_size + i) + k * matrix_size * matrix_size].real = dist1(engine);
				mat[(j * matrix_size + i) + k * matrix_size * matrix_size].imag = dist1(engine);
			}
		}
	}

	return;
}
void get_eig_main(int matrix_size, int mat_num)
{
	double time[2][3];
	MKL_Complex16* mymatrix_input;
	MKL_Complex16* mklmatrix_input;

	MKL_Complex16* myeigenvalues;
	MKL_Complex16* mkleigenvalues;

	mymatrix_input = (MKL_Complex16*)malloc(sizeof(MKL_Complex16) * matrix_size * matrix_size * mat_num);
	mklmatrix_input = (MKL_Complex16*)malloc(sizeof(MKL_Complex16) * matrix_size * matrix_size * mat_num);

	myeigenvalues = (MKL_Complex16*)malloc(sizeof(MKL_Complex16) * matrix_size * mat_num);
	mkleigenvalues = (MKL_Complex16*)malloc(sizeof(MKL_Complex16) * matrix_size * mat_num);


	memset(myeigenvalues, 0x00, sizeof(MKL_Complex16) * matrix_size * mat_num);
	memset(mkleigenvalues, 0x00, sizeof(MKL_Complex16) * matrix_size * mat_num);

	set_mat(mymatrix_input, matrix_size, mat_num);

	memcpy(mklmatrix_input, mymatrix_input, sizeof(MKL_Complex16) * matrix_size * matrix_size * mat_num);

	printf("my_zgeev_gpu\n");
	{
		int i;

		tokurablas_t* handle;
		tokuraCreate(&handle);

		cuDoubleComplex* gpumatrix_cpu;
		gpumatrix_cpu = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex) * matrix_size * matrix_size * mat_num);

		for (i = 0; i < matrix_size * matrix_size * mat_num; i++)
		{
			gpumatrix_cpu[i].x = mklmatrix_input[i].real;
			gpumatrix_cpu[i].y = mklmatrix_input[i].imag;
		}

		//GPU malloc
		cuDoubleComplex* mymatrix_input_gpu;
		cuDoubleComplex* myeigenvalues_gpu;
		cuDoubleComplex* work_gpu;
		char* flag_gpu;

		cudaMalloc((void**)& mymatrix_input_gpu, sizeof(cuDoubleComplex) * matrix_size * matrix_size * mat_num);
		cudaMalloc((void**)& myeigenvalues_gpu, sizeof(cuDoubleComplex) * matrix_size * mat_num);
		cudaMalloc((void**)& work_gpu, tokura_get_zgeeveigenvaluesgetworspacesize(matrix_size, mat_num));
		cudaMalloc((void**)& flag_gpu, sizeof(char) * mat_num);



		cudaMemcpy(mymatrix_input_gpu, gpumatrix_cpu, sizeof(cuDoubleComplex) * matrix_size * matrix_size * mat_num, cudaMemcpyHostToDevice);
		printf("tokura_zgeev_batched_gpu start\n");
		double start;
		double end;


		start = omp_get_wtime();
		int retval =
			tokura_zgeev_batched_gpu
			(
				handle,
				matrix_size,
				mat_num,
				mymatrix_input_gpu,
				myeigenvalues_gpu,
				work_gpu,
				flag_gpu,
				(cudaStream_t)NULL
			);



		cudaDeviceSynchronize();
		{
			cudaError_t err2 = cudaGetLastError();
			if (err2 != cudaSuccess) {
				gpuErrchk(err2);
			}
		}

		end = omp_get_wtime();
		printf("GPU: %lf[s]\n", end - start);



		printf("tokura_zgeev_batched_gpu end %d\n", retval);


		cudaMemcpy(gpumatrix_cpu, myeigenvalues_gpu, sizeof(cuDoubleComplex) * matrix_size * mat_num, cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();

		for (i = 0; i < matrix_size * mat_num; i++)
		{
			myeigenvalues[i].real = gpumatrix_cpu[i].x;
			myeigenvalues[i].imag = gpumatrix_cpu[i].y;


			//	printf("(%lf,%lf)\n", myeigenvalues[i].real, myeigenvalues[i].imag);

		}



		cudaFree(mymatrix_input_gpu);
		cudaFree(myeigenvalues_gpu);
		cudaFree(work_gpu);
		cudaFree(flag_gpu);


		free(gpumatrix_cpu);
		tokuraDestroy(handle);
	}
	//my_zgeev_gpu(matrix_size, mat_num, mymatrix_input, myeigenvalues, time[0]);

	printf("get_eig_MKL\n");

	get_eig_MKL(matrix_size, mat_num, mklmatrix_input, mkleigenvalues, time[1]);

	eigenvalues_sorter(matrix_size, mat_num, myeigenvalues);
	eigenvalues_sorter(matrix_size, mat_num, mkleigenvalues);

	chack_ans(matrix_size, mat_num, myeigenvalues, mkleigenvalues);



	free(mymatrix_input);
	free(mklmatrix_input);

	free(myeigenvalues);
	free(mkleigenvalues);

}


int main(void)
{//2152
	int n = 0;
	int mat_num = 8192;
	int i = 0;

	for (i = 1; i <= TOKURA_ZGEEV_BATCHED_MAX_MATRIX_SIZE; i += 1)
	{
		get_eig_main(i, mat_num);
	}


	return 0;
}

