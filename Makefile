all:
	nvcc -ccbin mpic++ -O3 -lineinfo -g -arch=sm_80 --extended-lambda -std=c++14 -lcusparse gpu_training.cu -o exec_gpu_training
	./exec_gpu_training

ncu: all
	ncu --set=full -o gpu_training_%p ./exec_gpu_training

cuda_malloc_memset:
	nvcc -O3 -g -arch=sm_80 --extended-lambda --std=c++14 -lcusparse cuda_malloc_memset.cu -o cuda_malloc_memset
	./exec_cuda_malloc_memset

cuda_malloc_managed:
	nvcc -O3 -g -arch=sm_80 --extended-lambda --std=c++14 -lcusparse cuda_malloc_managed.cu -o cuda_malloc_managed
	./exec_cuda_malloc_managed

hello_world:
	nvcc -O3 -g -arch=sm_80 --extended-lambda --std=c++14 -lcusparse hello_world.cu -o hello_world
	./exec_hello_world

