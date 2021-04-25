nvcc main.cu -L../tokura_zgeev_batched_library/bin/ -ltokurablas -I../tokura_zgeev_batched_library/include  -Xcompiler "-fopenmp"
