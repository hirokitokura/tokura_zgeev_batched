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
An exmaple code is provided at stream_test_with_MKL which computes all eigenvalues of many small real non-hermitian matrices on the gpu and output a maximum relative error vs intel MKL.


# Functions
