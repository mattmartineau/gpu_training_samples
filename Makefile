#PROF="nsys profile --trace=cuda,nvtx"

cuda_malloc_memset:
	nvcc -O3 -g -arch=sm_80 --extended-lambda --std=c++14 -lcusparse cuda_malloc_memset.cu -o exec_cuda_malloc_memset
	${PROF} ./exec_cuda_malloc_memset

cuda_malloc_managed:
	nvcc -O3 -g -arch=sm_80 --extended-lambda --std=c++14 -lcusparse cuda_malloc_managed.cu -o exec_cuda_malloc_managed
	${PROF} ./exec_cuda_malloc_managed

hello_world:
	nvcc -O3 -g -arch=sm_80 --extended-lambda --std=c++14 -lcusparse hello_world.cu -o exec_hello_world
	${PROF} ./exec_hello_world

hello_world_sync:
	nvcc -O3 -g -arch=sm_80 --extended-lambda --std=c++14 -lcusparse hello_world_sync.cu -o exec_hello_world_sync
	${PROF} ./exec_hello_world_sync

simple_stencil:
	nvcc -O3 -g -arch=sm_80 --extended-lambda --std=c++14 -lcusparse simple_stencil.cu -o exec_simple_stencil
	${PROF} ./exec_simple_stencil

simple_stencil_fixed:
	nvcc -O3 -g -arch=sm_80 --extended-lambda --std=c++14 -lcusparse simple_stencil_fixed.cu -o exec_simple_stencil_fixed
	${PROF} ./exec_simple_stencil_fixed

reduction_multipass_first:
	nvcc -O3 -g -arch=sm_80 --extended-lambda --std=c++14 -lcusparse reduction_multipass_first.cu -o exec_reduction_multipass_first
	${PROF} ./exec_reduction_multipass_first

memory_access_unrolling:
	nvcc -O3 -g -arch=sm_80 --extended-lambda --std=c++14 -lcusparse memory_access_unrolling.cu -o exec_memory_access_unrolling
	${PROF} ./exec_memory_access_unrolling

reduction_onepass:
	nvcc -O3 -g -arch=sm_80 --extended-lambda --std=c++14 -lcusparse reduction_onepass.cu -o exec_reduction_onepass
	${PROF} ./exec_reduction_onepass

