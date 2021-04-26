
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuComplex.h>

#include"tokura_blas.h"
//#include"mycuDoubleComplex.h"

constexpr int HOUSEHOLDER_VECTOR_SIZE = 2;
constexpr int MWB_WARP_SIZE = TARGET_CUDA_WARP;


__device__ static __inline__ double complexNormpower2(cuDoubleComplex a)
{
	double tmp;
	tmp = a.x * a.x + a.y * a.y;

	return tmp;
}
__device__ static __inline__ double complexNorm(cuDoubleComplex a)
{
	double tmp;
	tmp = complexNormpower2(a);
	tmp = sqrt(tmp);

	return tmp;
}


__device__ void compute_eigenvalues_2_2(const int matrix_size, const int mat_num, const cuDoubleComplex* mymatrix, const int lda, cuDoubleComplex* myeigenvalues)
{
	cuDoubleComplex tmp;
	cuDoubleComplex inside_root;
	cuDoubleComplex ans_root;
	cuDoubleComplex ans;




	tmp = cuCsub(mymatrix[((1) * lda + 1) * mat_num], mymatrix[((0) * lda + (0)) * mat_num]);
	inside_root = cuCmul(tmp, tmp);
	//tmp = complexMul(mymatrix[((submatrix_size)* matrix_size + submatrix_size)], mymatrix[((submatrix_size - 1) * matrix_size + (submatrix_size - 1))]);
	//tmp = complexFms(mymatrix[((submatrix_size - 1) * matrix_size + submatrix_size)], mymatrix[(submatrix_size * matrix_size + (submatrix_size - 1))], tmp);
	tmp = cuCmul(mymatrix[((0) * lda + 1) * mat_num], mymatrix[(1 * lda + (0)) * mat_num]);

	tmp.x *= 4.0;
	tmp.y *= 4.0;

	inside_root = cuCadd(inside_root, tmp);


	//�����v�Z
	ans_root.x = inside_root.x + sqrt(inside_root.x * inside_root.x + inside_root.y * inside_root.y);
	ans_root.x *= 0.5;
	ans_root.x = sqrt(ans_root.x);

	//�����v�Z
	ans_root.y = -inside_root.x + sqrt(inside_root.x * inside_root.x + inside_root.y * inside_root.y);
	ans_root.y *= 0.5;
	ans_root.y = sqrt(ans_root.y);
	if (inside_root.y < 0.0)
	{
		ans_root.y = -ans_root.y;
	}


	ans = cuCadd(mymatrix[((1) * lda + 1) * mat_num], mymatrix[((0) * lda + (0)) * mat_num]);
	ans = cuCadd(ans, ans_root);
	ans.x *= 0.5;
	ans.y *= 0.5;

	myeigenvalues[1] = ans;



	ans = cuCadd(mymatrix[((1) * lda + 1) * mat_num], mymatrix[((0) * lda + (0)) * mat_num]);
	ans = cuCsub(ans, ans_root);
	ans.x *= 0.5;
	ans.y *= 0.5;



	myeigenvalues[0] = ans;
}

__device__ void compute_submatrix_eigenvalues
(
	const int matrix_size,
	const int mat_num,
	const cuDoubleComplex* mymatrix,
	const int* zero_index,
	cuDoubleComplex* myeigenvalues
)
{
	int diag_index = matrix_size - 1;
	int submatrix_size = matrix_size;
	int computed_eigenvalues_num = 0;
	int prev_diag_index;
	diag_index = 0;
	submatrix_size = matrix_size;
	//MKL_Complex16 x, y, z;

	submatrix_size = matrix_size - 1;
	while (submatrix_size >= 0)
	{

		diag_index = (zero_index[(submatrix_size)* MWB_WARP_SIZE] != 0) || (submatrix_size == 0) ? submatrix_size : submatrix_size - 1;
		if (diag_index == submatrix_size)
		{
			myeigenvalues[submatrix_size * mat_num] = mymatrix[(submatrix_size * matrix_size + submatrix_size) * mat_num];
			submatrix_size--;
		}
		else
		{
			cuDoubleComplex eig_tmp[2];
			compute_eigenvalues_2_2(matrix_size, mat_num, &mymatrix[((submatrix_size - 1) * matrix_size + (submatrix_size - 1)) * mat_num], matrix_size, eig_tmp /*&myeigenvalues[(submatrix_size - 1)*mat_num]*/);


			myeigenvalues[(submatrix_size)* mat_num] = eig_tmp[1];
			myeigenvalues[(submatrix_size - 1) * mat_num] = eig_tmp[0];

			submatrix_size -= 2;
		}


	}
}
__device__ int ignore_test(double dest, double sorce)
{
	return (((dest /16.0) + sorce) == sorce);
}

__device__ int ignore_test_elementbyelement(cuDoubleComplex dest, cuDoubleComplex sorce)
{
	int flag=0;
	double tmp=0;
	//����
	if(sorce.x==0.0)
	{
		tmp=1.0;
	}
	else
	{
		tmp=sorce.x;
	}
	if((fabs(dest.x)+fabs(tmp))==fabs(tmp))
	{
		flag=1;
	}

	//����
	if(sorce.y==0.0)
	{
		tmp=1.0;
	}
	else
	{
		tmp=sorce.y;
	}
	if((fabs(dest.y)+fabs(tmp))==fabs(tmp))
	{
		flag=flag&&1;
	}
	return flag;
	//return (((dest /16.0) + sorce) == sorce);
}
__device__ double compute_norm2
(
	const cuDoubleComplex* workspace_zgehrd,
	const int matrix_size
)
{
	double householder_norm = 0.0;

	//���a�̌v�Z
	//for (int i = target_row_index; i < matrix_size; i++)
	for (int i = matrix_size - 1; i >= 0; i--)
	{
		householder_norm += complexNormpower2(workspace_zgehrd[i]);
	}

	return householder_norm;
}
__device__ double compute_norm
(
	const cuDoubleComplex* workspace_zgehrd,
	const int matrix_size
)
{
	double householder_norm = 0.0;

	householder_norm = compute_norm2(workspace_zgehrd, matrix_size);
	//���������Ƃ�(��ьv�Z)
	householder_norm = sqrt(householder_norm);

	return householder_norm;
}
__device__ cuDoubleComplex compute_alpha_singleqr(const cuDoubleComplex target_element, const double householder_norm)
{
	cuDoubleComplex alpha;
	double norm;

	norm = complexNorm(target_element);
	//alpha.x = -householder_norm * (target_element.x / norm);
	//alpha.y = -householder_norm * (target_element.y / norm);
	//if (norm != 0.0)
	{
		norm = householder_norm / norm;
		/*if (isnan(norm) | isinf(norm))
		{
			norm = 0.0;
		}*/
	}

	alpha.x = -norm * (target_element.x);
	alpha.y = -norm * (target_element.y);
	return alpha;
}


__device__ void bulge_generation_singleqr
(
	const int matrix_size,
	const int mat_num,
	cuDoubleComplex* __restrict__ mymatrix,
	int* __restrict__ zero_index,
	const int iteration,
	const int max_n
)
{
	cuDoubleComplex householder_tau;
	double householder_tau_double;
	cuDoubleComplex householder_vector[HOUSEHOLDER_VECTOR_SIZE];
	int diag_index;
	int submatrix_size = matrix_size;
	int computed_eigenvalues_num = 0;
	int prev_diag_index;
	diag_index = 0;
	submatrix_size = matrix_size;

	const cuDoubleComplex ZERO_CUDOUBLECOMPLEX = make_cuDoubleComplex(0.0, 0.0);
	householder_tau = ZERO_CUDOUBLECOMPLEX;


	int start_diag_index = 0;

	for (diag_index = 0; diag_index + 2 < max_n/*matrix_size*/; diag_index++)
	{
		submatrix_size = matrix_size;
		for (int i = 0; i < HOUSEHOLDER_VECTOR_SIZE; i++)
		{
			householder_vector[i] = ZERO_CUDOUBLECOMPLEX;
		}
		householder_tau = ZERO_CUDOUBLECOMPLEX;
		__syncthreads();
		/*ZERO_FLAG==1�̂Ƃ��̂݃o���W���ł���*/
		int ZERO_FLAG = ((zero_index[diag_index * MWB_WARP_SIZE] != 0) || (diag_index == 0))
			&& ((zero_index[(diag_index + 1) * MWB_WARP_SIZE] == 0)
				&& (zero_index[(diag_index + 2) * MWB_WARP_SIZE] == 0));

		if (ZERO_FLAG == 1)
		{
			for (int i = diag_index + 3; i < matrix_size + 1; i++)
			{
				if (zero_index[(i)* MWB_WARP_SIZE] != 0)
				{
					submatrix_size = zero_index[i * MWB_WARP_SIZE];
					break;
				}
			}
		}
		__syncthreads();
		if (ZERO_FLAG == 1)
		{
			//if (threadIdx.y < (submatrix_size - diag_index))
			{
				cuDoubleComplex tmp;
				if ((iteration + 1) % (47) == 0)
				{

					householder_tau = ZERO_CUDOUBLECOMPLEX;
				}
				else if ((iteration + 1) % (25) == 0)
				{
					cuDoubleComplex eig_tmp[2];
					compute_eigenvalues_2_2(matrix_size, mat_num, &mymatrix[((submatrix_size - 1 - 1) * matrix_size + (submatrix_size - 1 - 1)) * mat_num], matrix_size, eig_tmp);


					tmp = mymatrix[(((diag_index)* matrix_size + (diag_index))) * mat_num];
					double norm0 = complexNormpower2(cuCsub(tmp/*mymatrix[(((diag_index)* matrix_size + (diag_index))) * mat_num]*/, eig_tmp[0]));
					double norm1 = complexNormpower2(cuCsub(tmp/*mymatrix[(((diag_index)* matrix_size + (diag_index))) * mat_num]*/, eig_tmp[1]));
					householder_tau = norm0 < norm1 ? eig_tmp[1] : eig_tmp[0];

				}
				else if ((iteration + 1) % (20) == 0)
				{
					householder_tau.x = -mymatrix[((diag_index)* matrix_size + (diag_index)) * mat_num].x;
					householder_tau.y = -mymatrix[((diag_index)* matrix_size + (diag_index )) * mat_num].y;
				}
				else
				{
					cuDoubleComplex eig_tmp[2];
					compute_eigenvalues_2_2(matrix_size, mat_num, &mymatrix[((submatrix_size - 1 - 1) * matrix_size + (submatrix_size - 1 - 1)) * mat_num], matrix_size, eig_tmp);


					tmp = mymatrix[(((diag_index)* matrix_size + (diag_index))) * mat_num];
					double norm0 = complexNormpower2(cuCsub(tmp/*mymatrix[(((diag_index)* matrix_size + (diag_index))) * mat_num]*/, eig_tmp[0]));
					double norm1 = complexNormpower2(cuCsub(tmp/*mymatrix[(((diag_index)* matrix_size + (diag_index))) * mat_num]*/, eig_tmp[1]));
					householder_tau = norm0 < norm1 ? eig_tmp[0] : eig_tmp[1];
				}

				


			}
		}
		__syncthreads();
		start_diag_index = submatrix_size;
		if (ZERO_FLAG == 1)
		{
			ZERO_FLAG = ZERO_FLAG && (complexNorm(mymatrix[((diag_index)* matrix_size + (diag_index + 1)) * mat_num]) != 0.0);
		}
		__syncthreads();
		householder_tau_double = 0.0;
		/*�n�E�X�z���_�[�ϊ��\�z�J�n*/
		if (ZERO_FLAG == 1)
		{
			//	if (threadIdx.y < (submatrix_size - diag_index))
			{
				//p�̌v�Z
				householder_vector[0] = cuCsub(mymatrix[(((diag_index)* matrix_size + (diag_index))) * mat_num], householder_tau);
				//q�̌v�Z//
				householder_vector[1] = mymatrix[(((diag_index)* matrix_size + (diag_index + 1))) * mat_num];

				if(complexNormpower2(householder_vector[0])+complexNormpower2(householder_vector[1])==complexNormpower2(householder_vector[0]))
				{
					//householder_vector[0].x=0.0;
					//householder_vector[0].y=0.0;
					//householder_vector[1].x=0.0;
					//householder_vector[1].y=0.0;
				}
				else
				{

				}
				cuDoubleComplex alpha;
				//�ΏۂƂȂ�����т��v�Z����
				double householder_norm = compute_norm(householder_vector, HOUSEHOLDER_VECTOR_SIZE);
				//�x�N�g���̃X�J���{���v�Z
				alpha = compute_alpha_singleqr(householder_vector[0], householder_norm);


				householder_vector[0] = cuCsub(householder_vector[0], alpha);
				householder_norm = 0.5 * compute_norm2(householder_vector, HOUSEHOLDER_VECTOR_SIZE);

				householder_tau_double = 1.0 / (householder_norm);

				if(!((householder_norm + 1.0) == 1.0))
				{
					householder_tau_double = 1.0 / householder_norm;
				}
				else
				{
					householder_tau_double=0.0;
					ZERO_FLAG=0;
				}
				ZERO_FLAG = ZERO_FLAG && (!(isnan(householder_norm) || isnan(householder_tau_double) || isnan(householder_vector[0].x) || isnan(householder_vector[1].y)));

			}
		}
		ZERO_FLAG = ZERO_FLAG && (!(isnan(householder_tau_double) || isnan(householder_vector[0].x) || isnan(householder_vector[1].y)));
		ZERO_FLAG = ZERO_FLAG && (!(isinf(householder_tau_double) || isinf(householder_vector[0].x) || isinf(householder_vector[1].y)));

		if ((ZERO_FLAG == 0)&&(threadIdx.y == 0))
		{
			//mymatrix[(diag_index * matrix_size + (diag_index + 1 + 0)) * mat_num].x=0.0;
			//mymatrix[(diag_index * matrix_size + (diag_index + 1 + 0)) * mat_num].y=0.0;
			householder_tau_double=0.0;
		}
		__syncthreads();
		if (ZERO_FLAG == 1)
		{
			//�����ϊ�
			//������n�E�X�z���_�[�s���������	
			for (int target_column_index_similar = diag_index + threadIdx.y; target_column_index_similar < submatrix_size; target_column_index_similar += blockDim.y)
			{
				//(v*,A)�̌v�Z
				cuDoubleComplex inner_product;
				inner_product = ZERO_CUDOUBLECOMPLEX;

				int index;

				index = ((target_column_index_similar)* matrix_size + (diag_index + 0)) * mat_num;
				for (int i = 0; i < HOUSEHOLDER_VECTOR_SIZE; i++)
				{
					inner_product = cuCfma(cuConj(householder_vector[i]), (mymatrix[index/*((target_column_index_similar)* matrix_size + (diag_index + i)) * mat_num*/]), inner_product);
					index += mat_num;
				}
				//A-2*V*(v*,A)�̌v�Z
				inner_product.x *= householder_tau_double;
				inner_product.y *= householder_tau_double;

				index = ((target_column_index_similar)* matrix_size + (diag_index + 0)) * mat_num;
				for (int i = 0; i < HOUSEHOLDER_VECTOR_SIZE; i++)
				{

					mymatrix[index/*((target_column_index_similar)* matrix_size + (diag_index + i)) * mat_num*/]
						= cuCsub(mymatrix[index/*((target_column_index_similar)* matrix_size + (diag_index + i)) * mat_num*/], cuCmul(householder_vector[i], inner_product));

					index += mat_num;
				}
			}
		}
		__syncthreads();

		if (ZERO_FLAG == 1)
		{
			int mmin = ((submatrix_size < (diag_index + 3 + 1)) ? submatrix_size : diag_index + 3 + 1);
			//�E����n�E�X�z���_�[�s���������	
			for (int target_row_index_similar = diag_index + threadIdx.y; target_row_index_similar < mmin; target_row_index_similar += blockDim.y)
			{
				//(A,v)�̌v�Z
				cuDoubleComplex inner_product;
				inner_product = ZERO_CUDOUBLECOMPLEX;

				int index;
				index = ((diag_index + 0) * matrix_size + target_row_index_similar) * mat_num;
				for (int j = 0; j < HOUSEHOLDER_VECTOR_SIZE; j++)
				{
					inner_product = cuCfma(mymatrix[index/*((diag_index + j) * matrix_size + target_row_index_similar)* mat_num*/], (householder_vector[j]), inner_product);
					index += matrix_size * mat_num;

				}

				//A-2(A,v)v*�̌v�Z
				inner_product.x *= householder_tau_double;
				inner_product.y *= householder_tau_double;

				index = ((diag_index + 0) * matrix_size + target_row_index_similar) * mat_num;
				for (int j = 0; j < HOUSEHOLDER_VECTOR_SIZE; j++)
				{
					mymatrix[index/*((diag_index + j) * matrix_size + target_row_index_similar) * mat_num*/]
						= cuCsub(mymatrix[index/*((diag_index + j) * matrix_size + target_row_index_similar) * mat_num*/], cuCmul(inner_product, cuConj(householder_vector[j])));

					index += matrix_size * mat_num;
				}
			}
		}
		__syncthreads();
	}
}


__device__ void bulge_chasing_singleqr
(
	const int matrix_size,
	const int mat_num,
	cuDoubleComplex* __restrict__ mymatrix,
	const int* __restrict__ zero_index,
	const int max_n
)
{
	cuDoubleComplex householder_tau;
	double householder_tau_double;

	cuDoubleComplex householder_vector[HOUSEHOLDER_VECTOR_SIZE];
	int submatrix_size = matrix_size;
	int computed_eigenvalues_num = 0;
	int prev_diag_index;
	submatrix_size = matrix_size;
	int start_diag_index = 0;
	const cuDoubleComplex ZERO_CUDOUBLECOMPLEX = make_cuDoubleComplex(0.0, 0.0);
	for (int diag_index = 0; diag_index < max_n/*matrix_size*/ - 2/*m < nn-1-1*/; diag_index++)
	{
		if (zero_index[((diag_index + 1)) * MWB_WARP_SIZE] != 0)
		{
			start_diag_index = diag_index + 1;

		}
		else if (zero_index[((diag_index + 1 + 1)) * MWB_WARP_SIZE] != 0)
		{
			start_diag_index = diag_index + 1 + 1;
		}

		__syncthreads();
		int ZERO_FLAG = (zero_index[((diag_index + 1)) * MWB_WARP_SIZE] == 0) && (zero_index[((diag_index + 1 + 1)) * MWB_WARP_SIZE] == 0);
		if (ZERO_FLAG == 1)
		{
			/*�o���W�𐶐��ł��邩���f*/
			for (int i = diag_index + 1 + 1 + 1; i < matrix_size + 1; i++)
			{
				if (zero_index[(i)* MWB_WARP_SIZE] != 0)
				{
					submatrix_size = zero_index[(i)* MWB_WARP_SIZE];
					break;
				}
			}
		}
		__syncthreads();
		householder_tau_double = 0.0;
		if (ZERO_FLAG == 1)
		{
			householder_vector[0] = mymatrix[(diag_index * matrix_size + (diag_index + 1 + 0)) * mat_num];
			householder_vector[1] = mymatrix[(diag_index * matrix_size + (diag_index + 1 + 1)) * mat_num];
			double householder_norm;
			cuDoubleComplex alpha;


			if(complexNormpower2(householder_vector[0])+complexNormpower2(householder_vector[1])==complexNormpower2(householder_vector[0]))
			{
			//	householder_vector[0].x=0.0;
			//	householder_vector[0].y=0.0;
			//	householder_vector[1].x=0.0;
			//	householder_vector[1].y=0.0;
			}
			else
			{

			}
			//�ΏۂƂȂ�����т��v�Z����
			householder_norm = compute_norm(householder_vector, HOUSEHOLDER_VECTOR_SIZE);
			//�x�N�g���̃X�J���{���v�Z
			alpha = compute_alpha_singleqr(householder_vector[0], householder_norm);
			householder_vector[0] = cuCsub(householder_vector[0], alpha);

			//�ΏۂƂȂ�����т��v�Z����
			householder_norm = compute_norm2(householder_vector, HOUSEHOLDER_VECTOR_SIZE);
			if ((householder_norm != 0.0))
			//if (!((householder_norm +2.0) == 2.0))
			{
				//���K��
				householder_tau_double = 2.0 / (householder_norm);
			}
			else
			{
				householder_tau_double = 0.0;
			}
			ZERO_FLAG = ZERO_FLAG && (householder_norm != 0.0);
			//ZERO_FLAG = ZERO_FLAG && (!((householder_norm +2.0) == 2.0));

			ZERO_FLAG = ZERO_FLAG && (!(isnan(householder_tau_double)||isnan(householder_norm) || isnan(householder_tau_double) || isnan(householder_vector[0].x) || isnan(householder_vector[1].y)));

		}
		ZERO_FLAG = ZERO_FLAG && (!( isinf(householder_tau_double) || isinf(householder_vector[0].x) || isinf(householder_vector[1].y)));
if ((ZERO_FLAG == 0)&&(threadIdx.y == 0))
		{
			householder_tau_double=0.0;
		}
		__syncthreads();
		if ((ZERO_FLAG == 1))
		{
			//�����ϊ�
			//������n�E�X�z���_�[�s���������	
			for (int target_column_index_similar = diag_index + threadIdx.y; target_column_index_similar < submatrix_size; target_column_index_similar += blockDim.y)
			{
				//(v*,A)�̌v�Z
				cuDoubleComplex inner_product;
				inner_product = ZERO_CUDOUBLECOMPLEX;

				inner_product = cuCfma(cuConj(householder_vector[0]), (mymatrix[((target_column_index_similar)* matrix_size + (diag_index + 0 + 1)) * mat_num]), inner_product);
				inner_product = cuCfma(cuConj(householder_vector[1]), (mymatrix[((target_column_index_similar)* matrix_size + (diag_index + 1 + 1)) * mat_num]), inner_product);

				//A-2*V*(v*,A)�̌v�Z
				inner_product.x *= householder_tau_double;
				inner_product.y *= householder_tau_double;


				mymatrix[((target_column_index_similar)* matrix_size + (diag_index + 1 + 1)) * mat_num] =
					cuCsub(mymatrix[((target_column_index_similar)* matrix_size + (diag_index + 1 + 1)) * mat_num], cuCmul(householder_vector[1], inner_product));
				mymatrix[((target_column_index_similar)* matrix_size + (diag_index + 0 + 1)) * mat_num] =
					cuCsub(mymatrix[((target_column_index_similar)* matrix_size + (diag_index + 0 + 1)) * mat_num], cuCmul(householder_vector[0], inner_product));
			}
		}
		__syncthreads();
		if ((ZERO_FLAG == 1))
		{
			int mmin = (submatrix_size < (diag_index + 3 + 1 + 1) ? submatrix_size : (diag_index + 3 + 1 + 1));


			//�E����n�E�X�z���_�[�s���������	
			for (int target_row_index_similar = start_diag_index + threadIdx.y; target_row_index_similar < mmin; target_row_index_similar += blockDim.y)
			{
				//(A,v)�̌v�Z
				cuDoubleComplex inner_product;
				inner_product = ZERO_CUDOUBLECOMPLEX;



				inner_product = cuCfma(mymatrix[((diag_index + 1 + 0) * matrix_size + target_row_index_similar) * mat_num], (householder_vector[0]), inner_product);
				inner_product = cuCfma(mymatrix[((diag_index + 1 + 1) * matrix_size + target_row_index_similar) * mat_num], (householder_vector[1]), inner_product);

				//A-2(A,v)v*�̌v�Z
				inner_product.x *= householder_tau_double;
				inner_product.y *= householder_tau_double;

				mymatrix[((diag_index + 1 + 1) * matrix_size + target_row_index_similar) * mat_num] =
					cuCsub(mymatrix[((diag_index + 1 + 1) * matrix_size + target_row_index_similar) * mat_num], cuCmul(inner_product, cuConj(householder_vector[1])));

				mymatrix[((diag_index + 1 + 0) * matrix_size + target_row_index_similar) * mat_num] =
					cuCsub(mymatrix[((diag_index + 1 + 0) * matrix_size + target_row_index_similar) * mat_num], cuCmul(inner_product, cuConj(householder_vector[0])));

			}

		}
		__syncthreads();


	}
}

__global__ void tokura_zhseqr_normal_MWB_kernel
(
	const int matrix_size,
	const int mat_num,
	cuDoubleComplex* __restrict__ mymatrix_input,
	cuDoubleComplex* __restrict__ myeigenvalues_cuDoubleComplex_gpu,
	char* __restrict__ flags
)
{
	const int matrix_id = threadIdx.x;
	const int global_matrix_id = matrix_id + blockIdx.x * MWB_WARP_SIZE;


	cuDoubleComplex* mymatrix = &mymatrix_input[global_matrix_id];
	cuDoubleComplex* myeigenvalues = &myeigenvalues_cuDoubleComplex_gpu[global_matrix_id];

	int prev_computed_eigenvalues_num = 0;
	int submatrix_size = matrix_size;
	int iteration = 0;
	extern __shared__ int shared_dynamic[];
	int* zero_index = &shared_dynamic[0];
	int* eig_num_shared = &zero_index[(matrix_size + 1) * MWB_WARP_SIZE];
	zero_index = &zero_index[threadIdx.x];
	eig_num_shared = &eig_num_shared[threadIdx.x];


	const cuDoubleComplex ZERO_CUDOUBLECOMPLEX = make_cuDoubleComplex(0.0, 0.0);

	if (!(global_matrix_id < mat_num))
	{
		return;
	}

	if (threadIdx.y == 0)
	{
		zero_index[matrix_size * MWB_WARP_SIZE] = matrix_size;
	}
	for (int i = threadIdx.y; i < matrix_size; i += blockDim.y)
	{
		zero_index[i * MWB_WARP_SIZE] = 0;
	}
	__syncthreads();
	//	return;
	int its = 0;
	while (its != matrix_size * matrix_size * matrix_size *matrix_size)
	{
		its++;
		/*�s����S�~�̒l�����J�n*/
		for (int i = 2 + threadIdx.y; i < matrix_size; i += blockDim.y)
		{
			mymatrix[((i - 2) * matrix_size + i) * mat_num] = ZERO_CUDOUBLECOMPLEX;
		}

		__syncthreads();

		//���̊i�v�f���\���Ɏ����������𒲂ׂ�
		for (int diag_index = matrix_size - 1 - threadIdx.y; 0 < diag_index; diag_index -= blockDim.y)
		{
			if (zero_index[diag_index * MWB_WARP_SIZE] == 0)
			{
				double tmp;
				tmp = sqrt(complexNormpower2(mymatrix[(diag_index * matrix_size + diag_index) * mat_num]) + complexNormpower2(mymatrix[((diag_index - 1) * matrix_size + (diag_index - 1)) * mat_num]));

				if (tmp == 0.0)
				{
					tmp = 1.0;
				}

				cuDoubleComplex tmp_complex;

				tmp_complex.x=fabs(mymatrix[(diag_index * matrix_size + diag_index) * mat_num].x)+fabs(mymatrix[((diag_index - 1) * matrix_size + (diag_index - 1)) * mat_num].x);
				tmp_complex.y=fabs(mymatrix[(diag_index * matrix_size + diag_index) * mat_num].y)+fabs(mymatrix[((diag_index - 1) * matrix_size + (diag_index - 1)) * mat_num].y);

				tmp_complex.x*=0.5;
				tmp_complex.y*=0.5;
				if(ignore_test_elementbyelement(mymatrix[((diag_index - 1) * matrix_size + diag_index) * mat_num], tmp_complex))
				{
					zero_index[diag_index * MWB_WARP_SIZE] = diag_index;
				}
			}
		}
		__syncthreads();
		//���ۂɌŗL�l�������܂������𒲂ׂ�
		int computed_eigenvalues_num = 0;
		int prev_diag_index;
		if (threadIdx.y == 0)
		{
			prev_diag_index = matrix_size;
			for (int diag_index = matrix_size - 1; 0 < diag_index; diag_index--)
			{
				if (zero_index[diag_index * MWB_WARP_SIZE] != 0)
				{
					//ZERO_CUDOUBLECOMPLEX
					mymatrix[((diag_index - 1) * matrix_size + diag_index) * mat_num] = ZERO_CUDOUBLECOMPLEX;
					submatrix_size = prev_diag_index - diag_index;
					prev_diag_index = diag_index;
					if (submatrix_size < 3)
					{
						computed_eigenvalues_num += submatrix_size;
					}
				}
			}

			submatrix_size = prev_diag_index - 0;
			if (submatrix_size < 3)
			{
				computed_eigenvalues_num += submatrix_size;
			}

			eig_num_shared[0] = computed_eigenvalues_num;
		}
		__syncthreads();
		computed_eigenvalues_num = eig_num_shared[0];
		//�ŗL�l�����ׂČv�Z�����΁A���[�v�𔲂���
		if (computed_eigenvalues_num == matrix_size)
		{
			break;
		}

		__syncthreads();

		iteration++;
		if (iteration < 0)
		{
			iteration = 0;
		}
		if (prev_computed_eigenvalues_num != computed_eigenvalues_num)
		{
			iteration = 0;
		}
		prev_computed_eigenvalues_num = computed_eigenvalues_num;

		__syncthreads();


		//�o���W����
		bulge_generation_singleqr
		(
			matrix_size,
			mat_num,
			mymatrix,
			zero_index,
			iteration,
			matrix_size/*max_n*/
		);

		__syncthreads();
		bulge_chasing_singleqr
		(
			matrix_size,
			mat_num,
			mymatrix,
			zero_index,
			matrix_size/*max_n*/
		);
	}

	if ((its != matrix_size * matrix_size * matrix_size*matrix_size))
	{
		flags[global_matrix_id] = 0;
	}
	else
	{
		//printf("NOT COMPUTED\n");
		flags[global_matrix_id] = 1;
	}
	__syncthreads();

	if (threadIdx.y == 0)
	{
		compute_submatrix_eigenvalues
		(
			matrix_size,
			mat_num,
			mymatrix,
			zero_index,
			myeigenvalues
		);
	}

}

