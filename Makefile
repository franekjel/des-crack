CUDAPATH = /opt/cuda
CC = g++
NVCC = $(CUDAPATH)/bin/nvcc
CFLAGS =  -O2 -I$(CUDAPATH)/include
NVCCFLAGS = -O2 -maxrregcount 32 -m64 -arch=sm_61 -I$(CUDAPATH)/include -L$(CUDA_ROOT_DIR)/lib64

C_HEADERS = cpu_constants.h cpu_algorithm.h
CUDA_HEADERS = device_algorithm.h

all: des-crack clean

device_algorithm: device_algorithm.cu device_algorithm.h
	$(NVCC) -c $(NVCCFLAGS) device_algorithm.cu

cpu_constants: cpu_constants.h cpu_constants.c
	$(CC) -c $(CFLAGS) cpu_constants.c

cpu_algorithm: cpu_constants cpu_algorithm.h cpu_algorithm.c
	$(CC) -c $(CFLAGS) cpu_algorithm.c

main: device_algorithm cpu_algorithm $(C_HEADERS) $(CUDA_HEADERS)
	$(NVCC) -c $(NVVCFLAGS) main.cu

des-crack: main $(C_HEADERS) $(CUDA_HEADERS)
	$(NVCC) $(NVCCFLAGS) main.o cpu_algorithm.o cpu_constants.o device_algorithm.o -o des-crack
	
clean:
	rm *.o
