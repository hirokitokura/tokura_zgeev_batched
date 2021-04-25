cd tokura_zgeev_batched_library
cd src

nvcc *.cu -I/home/tokura/tokura_blas/tokura_zgeev_batched_library/include     -DTOKURA_ZGEEV_BATCHED_GEHRD_TUNING -o ../bin/zgehrd_tune.out
cd ../bin

./zgehrd_tune.out

cp load_zgehrd_fastmethod_helper.h ../src
cp load_zgehrd_normal_MWB_helper.h ../src
cp load_zgehrd_shared_MWB_helper.h ../src
cp tokura_zgehrd_parameters.h ../include

rm *

cd ../src


nvcc *.cu -I/home/tokura/tokura_blas/tokura_zgeev_batched_library/include     -DTOKURA_ZGEEV_BATCHED_HSEQR_TUNING -o ../bin/zhseqr_tune.out

cd ../bin

./zhseqr_tune.out

cp load_zhseqr_normal_MWB_helper.h ../src
cp load_zhseqr_fastmethod_helper.h ../src
cp tokura_zhseqr_parameters.h ../include

rm *

cd ../src

nvcc -shared *.cu -I/home/tokura/tokura_blas/tokura_zgeev_batched_library/include  -o ../bin/libtokurablas.so -Xcompiler "-fPIC" 


cd ../../
cd tokura_zgeev_batched_library_test

./compile.sh
