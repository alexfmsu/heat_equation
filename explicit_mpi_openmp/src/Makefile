compile_XL_O3:
	mpixlcxx_r explicit_single.cpp -O3 -qhot -qstrict           -o single.x;
	mpixlcxx_r explicit_mpi.cpp    -O3 -qhot -qstrict -qsmp=omp -o mpi.x;

compile_XL_O4:
	mpixlcxx_r explicit_single.cpp -O4 -qhot -qstrict           -o single.x;
	mpixlcxx_r explicit_mpi.cpp    -O4 -qhot -qstrict -qsmp=omp -o mpi.x;

compile_XL_O5:
	mpixlcxx_r explicit_single.cpp -O5 -qhot -qstrict           -o single.x;
	mpixlcxx_r explicit_mpi.cpp    -O5 -qhot -qstrict -qsmp=omp -o mpi.x;

run_omp_1:
	mpisubmit.bg -n 1 ./single.x data --stdout=1.out --stderr=1.err

run_omp_2:
	mpisubmit.bg -n 2 ./mpi.x data -mode SMP -env OMP_NUM_THREADS=4 --stdout=2.out --stderr=2.err

run_omp_4:
	mpisubmit.bg -n 4 ./mpi.x data -mode SMP -env OMP_NUM_THREADS=4 --stdout=4.out --stderr=4.err

run_omp_8:
	mpisubmit.bg -n 8 ./mpi.x data -mode SMP -env OMP_NUM_THREADS=4 --stdout=8.out --stderr=8.err

run_omp_16:
	mpisubmit.bg -n 16 ./mpi.x data -mode SMP -env OMP_NUM_THREADS=4 --stdout=16.out --stderr=16.err

run_omp_32:
	mpisubmit.bg -n 32 ./mpi.x data -mode SMP -env OMP_NUM_THREADS=4 --stdout=32.out --stderr=32.err

run_omp_64:
	mpisubmit.bg -n 64 ./mpi.x data -mode SMP -env OMP_NUM_THREADS=4 --stdout=64.out --stderr=64.err

run_omp_128:
	mpisubmit.bg -n 128 ./mpi.x data -mode SMP -env OMP_NUM_THREADS=4 --stdout=128.out --stderr=128.err

run_omp_256:
	mpisubmit.bg -n 256 -w 00:10:00 ./mpi.x data -mode SMP -env OMP_NUM_THREADS=4 --stdout=256.out --stderr=256.err

run_omp_512:
	mpisubmit.bg -n 512 -w 00:05:00 ./mpi.x data -mode SMP -env OMP_NUM_THREADS=4 --stdout=512.out --stderr=512.err

cancel:
	llcancel -u edu-cmc-ski16-052

clean:
	rm *.out *.err core* 2>/dev/null || echo > /dev/null

llq:
	llq -u edu-cmc-ski16-052

accel:
	perl accel.pl

run1_16_O3:
	make cancel
	make clean
	make compile_XL_O3
	make run_omp_1
	for i in 2 4 8 16; do make run_omp_$$i; done;

run1_16_O4:
	make cancel
	make clean
	make compile_XL_O4
	make run_omp_1
	for i in 2 4 8 16; do make run_omp_$$i; done;

run1_16_O5:
	make cancel
	make clean
	make compile_XL_O5
	make run_omp_1
	for i in 2 4 8 16; do make run_omp_$$i; done;

run1_32_O3:
	make cancel
	make clean
	make compile_XL_O3
	make run_omp_1
	for i in 2 4 8 16 32; do make run_omp_$$i; done;

run1_32_O4:
	make cancel
	make clean
	make compile_XL_O4
	make run_omp_1
	for i in 2 4 8 16 32; do make run_omp_$$i; done;

run1_32_O5:
	make cancel
	make clean
	make compile_XL_O5
	make run_omp_1
	for i in 2 4 8 16 32; do make run_omp_$$i; done;

run1_64_O3:
	make cancel
	make clean
	make compile_XL_O3
	make run_omp_1
	for i in 2 4 8 16 32 64; do make run_omp_$$i; done;

run1_64_O4:
	make cancel
	make clean
	make compile_XL_O4
	make run_omp_1
	for i in 2 4 8 16 32 64; do make run_omp_$$i; done;

run1_64_O5:
	make cancel
	make clean
	make compile_XL_O5
	make run_omp_1
	for i in 2 4 8 16 32 64; do make run_omp_$$i; done;

run1_128_O3:
	make cancel
	make clean
	make compile_XL_O3
	make run_omp_1
	for i in 2 4 8 16 32 64 128; do make run_omp_$$i; done;

run1_128_O4:
	make cancel
	make clean
	make compile_XL_O4
	make run_omp_1
	for i in 2 4 8 16 32 64 128; do make run_omp_$$i; done;

run1_128_O5:
	make cancel
	make clean
	make compile_XL_O5
	make run_omp_1
	for i in 2 4 8 16 32 64 128; do make run_omp_$$i; done;
	
run1_256_O3:
	make cancel
	make clean
	make compile_XL_O3
	make run_omp_1
	for i in 2 4 8 16 32 64 128 256; do make run_omp_$$i; done;

run1_256_O4:
	make cancel
	make clean
	make compile_XL_O4
	make run_omp_1
	for i in 2 4 8 16 32 64 128 256; do make run_omp_$$i; done;

run1_256_O5:
	make cancel
	make clean
	make compile_XL_O5
	make run_omp_1
	for i in 2 4 8 16 32 64 128 256; do make run_omp_$$i; done;

run1_512_O3:
	make cancel
	make clean
	make compile_XL_O3
	make run_omp_1
	for i in 2 4 8 16 32 64 128 256 512; do make run_omp_$$i; done;

run1_512_O4:
	make cancel
	make clean
	make compile_XL_O4
	make run_omp_1
	for i in 2 4 8 16 32 64 128 256 512; do make run_omp_$$i; done;

run1_512_O5:
	make cancel
	make clean
	make compile_XL_O5
	make run_omp_1
	for i in 2 4 8 16 32 64 128 256 512; do make run_omp_$$i; done;
