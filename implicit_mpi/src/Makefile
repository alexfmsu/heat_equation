compile:
	mpicxx implicit_single.cpp -o single.x;
	mpicxx implicit_mpi.cpp -o mpi.x;

compile_O3:
	mpicxx implicit_single.cpp -O3 -o single.x;
	mpicxx implicit_mpi.cpp -O3 -o mpi.x;

compile_XL:
	mpixlcxx implicit_single.cpp -o single.x;
	mpixlcxx implicit_mpi.cpp -o mpi.x;

compile_XL_O4:
	mpixlcxx implicit_single.cpp -O4 -o single.x;
	mpixlcxx implicit_mpi.cpp -O4 -o mpi.x;

compile_XL_O5:
	mpixlcxx implicit_single.cpp -O5 -o single.x;
	mpixlcxx implicit_mpi.cpp -O5 -o mpi.x;

run1:
	mpisubmit.bg -n 1 ./single.x data --stdout=1.out --stderr=1.err
	
run2:
	mpisubmit.bg -n 2 ./mpi.x data --stdout=2.out --stderr=2.err

run4:
	mpisubmit.bg -n 4 ./mpi.x data --stdout=4.out --stderr=4.err

run8:
	mpisubmit.bg -n 8 ./mpi.x data --stdout=8.out --stderr=8.err

run16:
	mpisubmit.bg -n 16 ./mpi.x data --stdout=16.out --stderr=16.err

run32:
	mpisubmit.bg -n 32 ./mpi.x data --stdout=32.out --stderr=32.err

run64:
	mpisubmit.bg -n 64 ./mpi.x data --stdout=64.out --stderr=64.err

run128:
	mpisubmit.bg -n 128 ./mpi.x data --stdout=128.out --stderr=128.err

run256:
	mpisubmit.bg -n 256 -w 00:10:00 ./mpi.x data --stdout=256.out --stderr=256.err

run512:
	mpisubmit.bg -n 512 -w 00:05:00 ./mpi.x data --stdout=512.out --stderr=512.err

cancel:
	llcancel -u edu-cmc-ski16-052

clean:
	rm *.out *.err core* 2>/dev/null || echo > /dev/null

llq:
	llq -u edu-cmc-ski16-052

accel:
	perl accel.pl

run1_16:
	make cancel
	make clean
	make compile
	make run1
	for i in 2 4 8 16; do make run$$i; done;

run1_16_O3:
	make cancel
	make clean
	make compile_O3
	make run1
	for i in 2 4 8 16; do make run$$i; done;

run1_16_XL:
	make cancel
	make clean
	make compile_XL
	make run1
	for i in 2 4 8 16; do make run$$i; done;

run1_32:
	make cancel
	make clean
	make compile
	make run1
	for i in 2 4 8 16 32; do make run$$i; done;

run1_32_O3:
	make cancel
	make clean
	make compile_O3
	make run1
	for i in 2 4 8 16 32; do make run$$i; done;

run1_32_XL:
	make cancel
	make clean
	make compile_XL
	make run1
	for i in 2 4 8 16 32; do make run$$i; done;

run1_64:
	make cancel
	make clean
	make compile
	make run1
	for i in 2 4 8 16 32 64; do make run$$i; done;

run1_64_O3:
	make cancel
	make clean
	make compile_O3
	make run1
	for i in 2 4 8 16 32 64; do make run$$i; done;

run1_64_XL:
	make cancel
	make clean
	make compile_XL
	make run1
	for i in 2 4 8 16 32 64; do make run$$i; done;

run1_64_XL_O4:
	make cancel
	make clean
	make compile_XL_O4
	make run1
	for i in 2 4 8 16 32 64; do make run$$i; done;

run1_64_XL_O5:
	make cancel
	make clean
	make compile_XL_O5
	make run1
	for i in 2 4 8 16 32 64; do make run$$i; done;

run1_128:
	make cancel
	make clean
	make compile
	make run1
	for i in 2 4 8 16 32 64 128; do make run$$i; done;
	
run1_128_O3:
	make cancel
	make clean
	make compile_O3
	make run1
	for i in 2 4 8 16 32 64 128; do make run$$i; done;

run1_128_XL:
	make cancel
	make clean
	make compile_XL
	make run1
	for i in 2 4 8 16 32 64 128; do make run$$i; done;

run1_256:
	make cancel
	make clean
	make compile
	make run1
	for i in 2 4 8 16 32 64 128 256; do make run$$i; done;
	
run1_256_O3:
	make cancel
	make clean
	make compile_O3
	make run1
	for i in 2 4 8 16 32 64 128 256; do make run$$i; done;

run1_256_XL:
	make cancel
	make clean
	make compile_XL
	make run1
	for i in 2 4 8 16 32 64 128 256; do make run$$i; done;

run1_512:
	make cancel
	make clean
	make compile
	make run1
	for i in 2 4 8 16 32 64 128 256 512; do make run$$i; done;
	
run1_512_O3:
	make cancel
	make clean
	make compile_O3
	make run1
	for i in 2 4 8 16 32 64 128 256 512; do make run$$i; done;

run1_512_XL:
	make cancel
	make clean
	make compile_XL
	make run1
	for i in 2 4 8 16 32 64 128 256 512; do make run$$i; done;
