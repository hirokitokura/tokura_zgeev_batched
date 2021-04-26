#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuComplex.h>




__device__ double sum_reduction(double* reduction_shared, const int MWB_reduction_thread_num, const int  WARP_SIZE_ON_MWB)
{
	int i = MWB_reduction_thread_num;
	while (i != 0)
	{
		if (threadIdx.y < i && ((threadIdx.y + i) < blockDim.y))
		{
			double target;
			double source;

			reduction_shared[threadIdx.x + threadIdx.y * WARP_SIZE_ON_MWB]
				+= reduction_shared[threadIdx.x + (threadIdx.y + i) * WARP_SIZE_ON_MWB];
		}
		__syncthreads();
		i = i >> 1;
	}

	return reduction_shared[threadIdx.x + 0 * WARP_SIZE_ON_MWB];
}

__device__ cuDoubleComplex compute_alpha(const cuDoubleComplex target_element, const   double householder_norm)
{
	cuDoubleComplex alpha;
	double norm;
	norm = target_element.x * target_element.x + target_element.y * target_element.y;
	norm = sqrt(norm);

	//alpha.x = -householder_norm * (target_element.x / norm);
	//alpha.y = -householder_norm * (target_element.y / norm);

	norm = householder_norm / norm;

	if (isnan(norm) | isinf(norm))
	{
		norm = 0.0;
	}

	alpha.x = -norm * (target_element.x);
	alpha.y = -norm * (target_element.y);
	return alpha;
}

//ハウスホルダーベクトル生成デバイス関数
__device__ void construct_householdervector
(
	const int target_row_index,
	const int matrix_size,
	const int mat_num,
	const cuDoubleComplex* __restrict__  mymatrix,
	cuDoubleComplex* house_vector_shared,//ハウスホルダーベクトル
	double* reduction_shared,
	const int MWB_reduction_thread_num,
	const int WARP_SIZE_ON_MWB
)
{
	cuDoubleComplex tmp;
	double householder_norm = 0.0;
	cuDoubleComplex alpha;
	int index;
	int index_house;
	//ワークスペースにベクトルをコピーする
	//ベクトルの2乗和を計算

	index = ((target_row_index - 1) * matrix_size + (target_row_index + 1 + threadIdx.y)) * mat_num;
	index_house = (target_row_index + 1 + threadIdx.y - 1) * WARP_SIZE_ON_MWB;
	for (int i = target_row_index + 1 + threadIdx.y; i < matrix_size; i += blockDim.y)
	{
		tmp = mymatrix[index/*((target_row_index - 1) * matrix_size + i) * mat_num*/];
		house_vector_shared[index_house/*(i - 1) * MWB_WARP_SIZE*/] = tmp;
		householder_norm += tmp.x * tmp.x + tmp.y * tmp.y;

		index += blockDim.y * mat_num;
		index_house += blockDim.y * WARP_SIZE_ON_MWB;

	}


	reduction_shared[threadIdx.x + threadIdx.y * WARP_SIZE_ON_MWB] = householder_norm;
	__syncthreads();
	//リダクションを行う
	householder_norm = sum_reduction(reduction_shared, MWB_reduction_thread_num, WARP_SIZE_ON_MWB);
	__syncthreads();

	if (threadIdx.y == 0)
	{
		tmp = mymatrix[((target_row_index - 1) * matrix_size + target_row_index) * mat_num];
		//ベクトルのスカラ倍を計算
		//alpha = compute_alpha(tmp, sqrt(tmp.x * tmp.x + tmp.y * tmp.y + householder_norm));

		if (householder_norm != 0.0)
		{
			tmp = cuCsub(tmp, compute_alpha(tmp, sqrt(tmp.x * tmp.x + tmp.y * tmp.y + householder_norm)));
			house_vector_shared[(target_row_index - 1) * WARP_SIZE_ON_MWB] = tmp;

			householder_norm = sqrt(tmp.x * tmp.x + tmp.y * tmp.y + householder_norm);
			reduction_shared[threadIdx.x + 0 * WARP_SIZE_ON_MWB] = householder_norm;
		}
	}
	__syncthreads();

	householder_norm = reduction_shared[threadIdx.x + 0 * WARP_SIZE_ON_MWB];
	//if (householder_norm != 0.0)
	if(!((householder_norm + 1.0) == 1.0))
	{
		householder_norm = 1.0 / householder_norm;
	}
	else
	{
		householder_norm=0.0;
	}

	//正規化
	index_house = (target_row_index + threadIdx.y - 1) * WARP_SIZE_ON_MWB;
	for (int i = target_row_index + threadIdx.y; i < matrix_size; i += blockDim.y)
	{
		tmp = house_vector_shared[index_house/*(i - 1) * MWB_WARP_SIZE*/];
		tmp.x *= householder_norm;
		tmp.y *= householder_norm;


		house_vector_shared[index_house/*(i - 1) * MWB_WARP_SIZE*/] = tmp;

		index_house += blockDim.y * WARP_SIZE_ON_MWB;
	}
	__syncthreads();
}

__device__  int get_MWB_reduction_thread_num_device(const int threads_num_per_matrix)
{
	int retval = -1;


	if ((1 <= threads_num_per_matrix) && (threads_num_per_matrix <= (1 << 1)))
	{
		retval = 1;
	}
	else
	{
		for (int i = 1; i < sizeof(int) * CHAR_BIT - 1; i++)
		{
			if ((1 < (threads_num_per_matrix << i)) && (threads_num_per_matrix <= (1 << (i + 1))))
			{
				retval = 1 << i;
				break;
			}
		}
	}

	return retval;
}

template<const int MWB_WARP_SIZE>
__global__ void tokura_zgehrd_normal_MWB_kernel
(
	const int matrix_size,
	const int mat_num,
	cuDoubleComplex* __restrict__ mymatrix_input
)
{
	//	constexpr int MWB_reduction_thread_num_ = get_MWB_reduction_thread_num_device(MWB_reduction_thread_num);


	const int matrix_id = threadIdx.x;
	const int global_matrix_id = matrix_id + blockIdx.x * MWB_WARP_SIZE;
	const int matrix_size_mat_num = matrix_size * mat_num;

	cuDoubleComplex* mymatrix = &mymatrix_input[global_matrix_id];

	extern __shared__ cuDoubleComplex shared_dynamic[];
	cuDoubleComplex* house_vector_shared = &shared_dynamic[0];//ハウスホルダーベクトル保存用
	double* reduction_shared = (double*)& house_vector_shared[(matrix_size - 1) * MWB_WARP_SIZE];//リダクション用
	cuDoubleComplex inner_product;
	cuDoubleComplex tmp;
	int index;
	int index_house;

	house_vector_shared = &house_vector_shared[threadIdx.x];

	if (!(global_matrix_id < mat_num))
	{
		return;
	}

	/*if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0)
	{
		for (int i = 0; i < matrix_size; i++)
		{
			for (int j = 0; j < matrix_size; j++)
			{
				printf("(%e %e) ", mymatrix[(j * matrix_size + i) * mat_num].x, mymatrix[(j * matrix_size + i) * mat_num].y);
			}
			printf("\n");
		}
		printf("\n");
	}*/


	const int MWB_reduction_thread_num = get_MWB_reduction_thread_num_device(blockDim.y);
	__syncthreads();

	for (int target_row_index = 1; target_row_index < matrix_size - 1; target_row_index++)
	{

		//ハウスホルダーベクトルvを計算する
		//複素共役はしていない
		//最後に2倍すべき
		construct_householdervector
		(
			target_row_index,
			matrix_size,
			mat_num,
			mymatrix,
			house_vector_shared,
			reduction_shared,
			MWB_reduction_thread_num,
			MWB_WARP_SIZE
		);

		//相似変換
		//左からハウスホルダー行列をかける	
		for (int target_column_index_similar = target_row_index - 1 + threadIdx.y; target_column_index_similar < matrix_size; target_column_index_similar += blockDim.y)
		{
			//(v*,A)の計算
			inner_product.x = 0.0;
			inner_product.y = 0.0;

			index = ((target_column_index_similar)* matrix_size + target_row_index) * mat_num;
			index_house = (target_row_index - 1) * MWB_WARP_SIZE;
			for (int i = target_row_index; i < matrix_size; i++)
			{
				inner_product = cuCfma(cuConj(house_vector_shared[index_house/*(i - 1) * MWB_WARP_SIZE*/]), mymatrix[index/*((target_column_index_similar)* matrix_size + i) * mat_num*/], inner_product);
				index += mat_num;
				index_house += MWB_WARP_SIZE;
			}

			//A-2*V*(v*,A)の計算
			inner_product.x *= 2.0;
			inner_product.y *= 2.0;

			index = ((target_column_index_similar)* matrix_size + matrix_size - 1) * mat_num;
			index_house = (matrix_size - 1 - 1) * MWB_WARP_SIZE;
			//for (int i = target_row_index; i < matrix_size; i++)
			for (int i = matrix_size - 1; i >= target_row_index; i--)
			{

				tmp = mymatrix[index/*((target_column_index_similar)* matrix_size + i) * mat_num*/];
				tmp = cuCsub(tmp, cuCmul(house_vector_shared[index_house/*(i - 1) * MWB_WARP_SIZE*/], inner_product));
				mymatrix[index/*((target_column_index_similar)* matrix_size + i) * mat_num*/] = tmp;

				index -= mat_num;
				index_house -= MWB_WARP_SIZE;

			}
		}
		__syncthreads();


		//右からハウスホルダー行列をかける	
		for (int target_row_index_similar = threadIdx.y; target_row_index_similar < matrix_size; target_row_index_similar += blockDim.y)
		{
			//(A,v)の計算
			inner_product.x = 0.0;
			inner_product.y = 0.0;

			index = (target_row_index * matrix_size + target_row_index_similar) * mat_num;
			index_house = (target_row_index - 1) * MWB_WARP_SIZE;
			for (int j = target_row_index; j < matrix_size; j++)
			{
				inner_product = cuCfma(mymatrix[index/*(j * matrix_size + target_row_index_similar) * mat_num*/], house_vector_shared[index_house/*(j - 1) * MWB_WARP_SIZE*/], inner_product);
				index += matrix_size_mat_num/* matrix_size * mat_num*/;
				index_house += MWB_WARP_SIZE;
			}

			//A-2(A,v)v*の計算
			inner_product.x *= 2.0;
			inner_product.y *= 2.0;


			//for (int j = target_row_index; j < matrix_size; j++)

			index = ((matrix_size - 1) * matrix_size + target_row_index_similar) * mat_num;
			index_house = (matrix_size - 1 - 1) * MWB_WARP_SIZE;
			for (int j = matrix_size - 1; j >= target_row_index; j--)
			{
				tmp = mymatrix[(j * matrix_size + target_row_index_similar) * mat_num];
				tmp = cuCsub(tmp, cuCmul(inner_product, cuConj(house_vector_shared[index_house/*(j - 1) * MWB_WARP_SIZE*/])));
				mymatrix[(j * matrix_size + target_row_index_similar) * mat_num] = tmp;
				index -= matrix_size_mat_num/* matrix_size * mat_num*/;
				index_house -= MWB_WARP_SIZE;
			}
		}
		__syncthreads();
	}


	/*if ( threadIdx.y == 0 )
	{

		for (int i = 0; i < matrix_size; i++)
		{
			for (int j = 0; j < matrix_size; j++)
			{
				if(isnan(mymatrix[(j * matrix_size + i) * mat_num].x))
				{
					printf("NAN \n");
				}
				if(isnan(mymatrix[(j * matrix_size + i) * mat_num].y))
				{
					printf("NAN \n");
				}
				//printf("(%e %e) ", mymatrix[(j * matrix_size + i) * mat_num].x, mymatrix[(j * matrix_size + i) * mat_num].y);
			}
			//printf("\n");
		}
		//printf("\n");
	}*/

}



template<const int MWB_WARP_SIZE>
__global__ void tokura_zgehrd_shared_MWB_kernel
(
	const int matrix_size,
	const int mat_num,
	cuDoubleComplex* __restrict__ mymatrix_input
)
{
	//const int MWB_reduction_thread_num = 8;

	const int matrix_id = threadIdx.x;
	const int global_matrix_id = matrix_id + blockIdx.x * MWB_WARP_SIZE;

	cuDoubleComplex* mymatrix = &mymatrix_input[global_matrix_id];

	extern __shared__ cuDoubleComplex shared_dynamic[];
	cuDoubleComplex* matrix_shared = &shared_dynamic[0];
	cuDoubleComplex* house_vector_shared = &matrix_shared[matrix_size * matrix_size * MWB_WARP_SIZE];//ハウスホルダーベクトル保存用
	double* reduction_shared = (double*)& house_vector_shared[(matrix_size - 1) * MWB_WARP_SIZE];//リダクション用
	cuDoubleComplex inner_product;
	cuDoubleComplex tmp;
	int index;
	int index_house;

	matrix_shared = &matrix_shared[threadIdx.x];
	house_vector_shared = &house_vector_shared[threadIdx.x];
	if (!(global_matrix_id < mat_num))
	{
		return;
	}
	const int MWB_reduction_thread_num = get_MWB_reduction_thread_num_device(blockDim.y);

	/*if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0)
	{
		for (int i = 0; i < matrix_size; i++)
		{
			for (int j = 0; j < matrix_size; j++)
			{
				printf("(%e %e) ", mymatrix[(j * matrix_size + i) * mat_num].x, mymatrix[(j * matrix_size + i) * mat_num].y);
			}
			printf("\n");
		}
		printf("\n");
	}*/
	//行列を読み込む
	for (int k = threadIdx.y; k < matrix_size * matrix_size; k += blockDim.y)
	{
		/*int i = k % matrix_size;
		int j = k / matrix_size;*/
		matrix_shared[(k/*j * matrix_size + i*/)* MWB_WARP_SIZE]
			= mymatrix[(k/*j * matrix_size + i*/)* mat_num];

	}
	__syncthreads();
	//if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0)
	//{
	//	for (int i = 0; i < matrix_size; i++)
	//	{
	//		for (int j = 0; j < matrix_size; j++)
	//		{
	//			printf("(%e %e) ", mymatrix[(j * matrix_size + i) * mat_num].x, mymatrix[(j * matrix_size + i) * mat_num].y);
	//		}
	//		printf("\n");
	//	}
	//	printf("\n");
	//}
	//__syncthreads();
	for (int target_row_index = 1; target_row_index < matrix_size - 1; target_row_index++)
	{

		//ハウスホルダーベクトルvを計算する
		//複素共役はしていない
		//最後に2倍すべき
		/*construct_householdervector_shared
		(
			target_row_index,
			matrix_size,
			matrix_shared,
			house_vector_shared,
			reduction_shared,
			MWB_reduction_thread_num
		);*/
		construct_householdervector
		(
			target_row_index,
			matrix_size,
			MWB_WARP_SIZE,
			matrix_shared,
			house_vector_shared,
			reduction_shared,
			MWB_reduction_thread_num,
			MWB_WARP_SIZE
		);

		//相似変換
		//左からハウスホルダー行列をかける	
		for (int target_column_index_similar = target_row_index - 1 + threadIdx.y; target_column_index_similar < matrix_size; target_column_index_similar += blockDim.y)
		{
			//(v*,A)の計算
			inner_product.x = 0.0;
			inner_product.y = 0.0;

			index = ((target_column_index_similar)* matrix_size + target_row_index) * MWB_WARP_SIZE;
			index_house = (target_row_index - 1) * MWB_WARP_SIZE;
			for (int i = target_row_index; i < matrix_size; i++)
			{
				//	inner_product = cuCfma(cuConj(house_vector_shared[ (i - 1) * blockDim.x]), matrix_shared[((target_column_index_similar)* matrix_size + i) * blockDim.x], inner_product);

				inner_product = cuCfma(cuConj(house_vector_shared[index_house/*(i - 1) * MWB_WARP_SIZE*/]), matrix_shared[index/*((target_column_index_similar)* matrix_size + i) * mat_num*/], inner_product);
				index += MWB_WARP_SIZE;
				index_house += MWB_WARP_SIZE;
			}

			//A-2*V*(v*,A)の計算
			inner_product.x *= 2.0;
			inner_product.y *= 2.0;

			index = ((target_column_index_similar)* matrix_size + matrix_size - 1) * MWB_WARP_SIZE;
			index_house = (matrix_size - 1 - 1) * MWB_WARP_SIZE;

			//for (int i = target_row_index; i < matrix_size; i++)
			for (int i = matrix_size - 1; i >= target_row_index; i--)
			{
				/*	tmp = matrix_shared[((target_column_index_similar)* matrix_size + i) * blockDim.x ];
					tmp = cuCsub(tmp, cuCmul(house_vector_shared[ (i - 1) * blockDim.x], inner_product));
					matrix_shared[((target_column_index_similar)* matrix_size + i) * blockDim.x ] = tmp;*/

				tmp = matrix_shared[index/*((target_column_index_similar)* matrix_size + i) * mat_num*/];
				tmp = cuCsub(tmp, cuCmul(house_vector_shared[index_house/*(i - 1) * MWB_WARP_SIZE*/], inner_product));
				matrix_shared[index/*((target_column_index_similar)* matrix_size + i) * mat_num*/] = tmp;

				index -= MWB_WARP_SIZE;
				index_house -= MWB_WARP_SIZE;
			}
		}
		__syncthreads();


		//右からハウスホルダー行列をかける	
		for (int target_row_index_similar = threadIdx.y; target_row_index_similar < matrix_size; target_row_index_similar += blockDim.y)
		{
			//(A,v)の計算
			inner_product.x = 0.0;
			inner_product.y = 0.0;

			index = (target_row_index * matrix_size + target_row_index_similar) * MWB_WARP_SIZE;
			index_house = (target_row_index - 1) * MWB_WARP_SIZE;
			for (int j = target_row_index; j < matrix_size; j++)
			{
				//	inner_product = cuCfma(matrix_shared[(j * matrix_size + target_row_index_similar) * blockDim.x ], house_vector_shared[ (j - 1) * blockDim.x], inner_product);

				inner_product = cuCfma(matrix_shared[index/*(j * matrix_size + target_row_index_similar) * mat_num*/], house_vector_shared[index_house/*(j - 1) * MWB_WARP_SIZE*/], inner_product);
				index += matrix_size * MWB_WARP_SIZE/* matrix_size * mat_num*/;
				index_house += MWB_WARP_SIZE;
			}

			//A-2(A,v)v*の計算
			inner_product.x *= 2.0;
			inner_product.y *= 2.0;

			index = ((matrix_size - 1) * matrix_size + target_row_index_similar) * MWB_WARP_SIZE;
			index_house = (matrix_size - 1 - 1) * MWB_WARP_SIZE;
			//for (int j = target_row_index; j < matrix_size; j++)
			for (int j = matrix_size - 1; j >= target_row_index; j--)
			{
				//	tmp = matrix_shared[(j * matrix_size + target_row_index_similar) * blockDim.x ];
				//	tmp = cuCsub(tmp, cuCmul(inner_product, cuConj(house_vector_shared[ (j - 1) * blockDim.x])));
				//	matrix_shared[(j * matrix_size + target_row_index_similar) * blockDim.x ] = tmp;

				tmp = matrix_shared[index/*(j * matrix_size + target_row_index_similar) * blockDim.x*/];
				tmp = cuCsub(tmp, cuCmul(inner_product, cuConj(house_vector_shared[index_house/*(j - 1) * MWB_WARP_SIZE*/])));
				matrix_shared[index/*(j * matrix_size + target_row_index_similar) * blockDim.x*/] = tmp;
				index -= matrix_size * MWB_WARP_SIZE/* matrix_size * mat_num*/;
				index_house -= MWB_WARP_SIZE;
			}

		}
		__syncthreads();

	}
	//行列を書き込む
	for (int k = threadIdx.y; k < matrix_size * matrix_size; k += blockDim.y)
	{
		/*int i = k % matrix_size;
		int j = k / matrix_size;*/
		mymatrix[(k/*j * matrix_size + i*/)* mat_num] =
			matrix_shared[(k/*j * matrix_size + i*/)* MWB_WARP_SIZE];


	}
	//__syncthreads();
	//if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.x == 0)
	//{
	//	for (int i = 0; i < matrix_size; i++)
	//	{
	//		for (int j = 0; j < matrix_size; j++)
	//		{
	//			printf("(%e %e) ", mymatrix[(j * matrix_size + i) * mat_num].x, mymatrix[(j * matrix_size + i) * mat_num].y);
	//		}
	//		printf("\n");
	//	}
	//	printf("\n");
	//}

}
