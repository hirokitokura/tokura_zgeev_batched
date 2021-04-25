#gcc -c get_eig_MKL.c  -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl  -m64 -I${MKLROOT}/include -fopenmp 


#nvcc  get_eig_MKL.o kernel.cu -L../tokura_zgeev_batched_library/bin/ -ltokurablas -L${MKLROOT}/lib/intel64  -I../tokura_zgeev_batched_library/include  -m64 -I${MKLROOT}/include -Xcompiler "-fopenmp" 

#nvcc get_eig_MKL.c kernel.cu -L../tokura_zgeev_batched_library/bin/ -ltokurablas   -I../tokura_zgeev_batched_library/include  -m64 -I${MKLROOT}/include -Xcompiler "-fopenmp"  -Xcompiler "-Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group" -lpthread -lm -ldl
nvcc  kernel.cu -L../tokura_zgeev_batched_library/bin/ -ltokurablas   -I../tokura_zgeev_batched_library/include  -m64 -I${MKLROOT}/include -Xcompiler "-fopenmp"  --linker-options ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a,${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a


./a.out
