#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>

#include <random>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuComplex.h>

#include <tokura_blas.h>


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
