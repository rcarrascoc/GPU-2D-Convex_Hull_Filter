all: 
	nvcc -O3 -arch sm_70 --extended-lambda -o prog main.cu

alt:
	test index

test:
	nvcc -O3 -o prog main.cpp
	
scan.o:
	nvcc -O3 -arch sm_70 -c src/scan.cu

index: kernel.o
	ar rc scan.a scan.o

clean: 
	-rm *.o *.a
