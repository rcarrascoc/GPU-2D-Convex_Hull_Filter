all: link
	nvcc -O3 -o prog4 main.cpp -lgmp filter.a

filter.o:
	nvcc -O3 -Xcompiler -fopenmp -arch sm_70 --extended-lambda -c src/filter.cu

link: filter.o
	ar rc filter.a filter.o

alt: clean link
	nvcc -O3 -Xcompiler -fopenmp -o prog3 main.cpp filter.a -I/home/linuxbrew/.linuxbrew/opt/cgal/include/ -I/home/linuxbrew/.linuxbrew/opt/boost/include 
raw: 
	nvcc -O3 -arch sm_70 --extended-lambda -o prog main.cu -lgmp

clean:
	rm -f prog3 filter.o filter.a