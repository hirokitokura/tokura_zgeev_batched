# A GPU Implementation of Bulk Computation of the Eigenvalue Problem for Many Small Non-Hermitian matrices
We provide the CUDA-program for bulk computation of the eigenvalue problem for many small real non-hermitian matrices in the GPU.

Our program supports eigenvalue computation of COMPLEX matrices of which size is equal or less than 64. 
The size of all metrices should be the same.

This is related with a following paper.
http://www.ijnc.org/index.php/ijnc/article/view/152

# License
MIT

# Compile an example for Linux
* `mkdir $HOME/tokura_zgeev_batched`
* git clone https://github.com/hirokitokura/tokura_zgeev_batched.git
* cd tokura_zgeev_batched/
* ./compile.sh
  *  Auto tuning program is compiled and executed, so please wait several time.
  *  libtokurablas.so will be generated at tokura_zgeev_batched_library/bin .

# Example source code
An exmaple code is provided at stream_test_with_MKL which computes all eigenvalues of many small non-hermitian matrices on the gpu and output a maximum relative error vs intel MKL.


# Functions
`int tokuraCreate(tokurablas_t** handle)`
This function allocates a handler for this library.

__Arguments__
* [in/out] `handle` tokurablas_t*, This is a handler for this library.
---

`int tokuraDestroy(tokurablas_t* handle)`
This function frees a handler.

__Arguments__
* [in/out] `handle` tokurablas_t*, This is a handler for this library.
---
`size_t tokura_get_zgeeveigenvaluesgetworspacesize(int n, int batch_count)`
This function returns a temporary work space size for tokura_zgeev_batched_gpu.

__Arguments__
* [in] `n` int, The order of a matrix. Should be 1<=n<=64.
* [in] `batch_count` int, The number of metrices.

__Return value__
* size_t, a temporary work space size for tokura_zgeev_batched_gpu.

---
`int tokura_zgeev_batched_gpu
(
	tokurablas_t* handle,
	int n,
	int batch_count,
	cuDoubleComplex* A,
	cuDoubleComplex* eigenvalues,
	cuDoubleComplex* work,
	char* flag,
	cudaStream_t cudastream
 )`
 
 This function computes all eigenvalues for n-by-n non-hermitian matrices.
 No eigenvectors are computed.
 
 __Arguments__
 * [in] `handle` tokurablas_t*, This is a handler for tokura_zgeev_batched_gpu.
 * [in] `n` int, The order of a matrix. Should be 1<=n<=64.
 * [in] `batch_count` int, The number of metrices.
 * [in/out] `A` cuDoubleComplex*, A pointer to cuDoubleComplex array which contain all metrices. Each matrix is stored one by one and elements of each matrix are stored in column-major order. After call tokura_zgeev_batched_gpu, do not use any element of A.
 * [out] `eigenvalues` cuDoubleComplex*, A pointer to cuDoubleComplex array which contain all eigenvalues. i-th eigenvalue of j-th matrix can be access by `eigenvalues[i+j*n]`.
 * [in/out] `work` cuDoubleComplex*, This is a temporary work space. Should be allocated a memory size which is equal or more than  `tokura_get_zgeeveigenvaluesgetworspacesize`.
 * [out] `flag` char*, This is flags. If eigenvalues of j-th metrix is computed, flag[j] is 0, otherwise flag[j] is not 0.
 * [in] `cudastream` cudaStream_t, This is used for asynchronous computation. If synchronous computation is preferred, cudastream should be NULL.
