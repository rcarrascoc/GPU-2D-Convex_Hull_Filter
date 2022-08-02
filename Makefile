all: link
	nvcc -O3 -o prog main.cpp -lgmp filter.a

filter.o:
	nvcc -O3 -arch sm_70 --extended-lambda -c src/filter.cu

link: filter.o
	ar rc filter.a filter.o

alt: 
	nvcc -O3 -arch sm_70 --extended-lambda -o prog main.cu -lgmp

clean:
	rm -f prog filter.o filter.a