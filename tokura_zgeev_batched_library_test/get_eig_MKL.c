#include<stdio.h>
#include<stdlib.h>


#include<omp.h>
#include<mkl.h>
#include "mkl_lapacke.h"

 lapack_int LAPACKE_zgeev( int matrix_layout, char jobvl, char jobvr, lapack_int n, lapack_complex_double* a, lapack_int lda, lapack_complex_double* w, lapack_complex_double* vl, lapack_int ldvl, lapack_complex_double* vr, lapack_int ldvr );


void get_eig_MKL
(
	int matrix_size,
	int mat_num,
	MKL_Complex16* mklmatrix_input,
	MKL_Complex16* mkleigenvalues,
	double* time
)
{
	
	double start;
	double end;


	start = omp_get_wtime();
#pragma omp parallel default(shared)
	{
#pragma omp for
		for (int k = 0; k < mat_num; k++)
		{

			LAPACKE_zgeev
			(
				LAPACK_COL_MAJOR,
				'N',
				'N',
				matrix_size,
				&mklmatrix_input[k * matrix_size * matrix_size],
				matrix_size,
				&mkleigenvalues[k * matrix_size],
				NULL,
				1,
				NULL,
				1
			);
		}
	}

	end = omp_get_wtime();
	time[0] = end - start;

	printf("MKL: %lf[s]\n", time[0]);
	//for (int i = 0; i < matrix_size; i++)
	//{
	//	printf("%e %e\n", mkleigenvalues[i].real, mkleigenvalues[i].imag);
	//}
	//printf("\n");

}
