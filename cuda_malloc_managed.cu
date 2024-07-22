#include <thrust/random.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <cusparse.h>
#include <vector>
#include <fstream>
#include <nvtx3/nvToolsExt.h>
#include "common.h"

__global__ void warmup_kernel(int N, int* a)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  if(i < N) {
    a[i] = a[i];
  }
}

void print_ints(int N, int* a, const char* nvtx_name)
{
  nvtxRangePush(nvtx_name);
  for(int i = 0; i < N; ++i) {
    printf("element %d = %d\n", i, a[i]);
  }
  nvtxRangePop();
}

// CUDA example with managed memory
void cuda_malloc_managed_test(const int N)
{
  printf("\n\n***Running %s\n\n", __func__);

  // NVTX range (instrumentation) accepting function name
  nvtxRangePush(__func__);

  // Allocate some host memory and initialize
  // TODO Change this allocation to use unified memory
  int* a_h = (int*)malloc(sizeof(int)*N);
  for(int i = 0; i < N; ++i) {
    a_h[i] = -1;
  }

  // Initialize some device memory
  // TODO We don't need this anymore
  int* a_d;
  CHECK_CUDA(cudaMalloc(&a_d, sizeof(int)*N));

  // Overwite a_m with GPU to 0
  // a_m = 0
  CHECK_CUDA(cudaMemset(a_d, 0, sizeof(int)*N));

  printf("Printing host memory a_h pre-copy\n");
  print_ints(N, a_h, "pre_copy_print");

  // Copy device data to host memory
  // a_h = a_d
  // TODO We don't need this anymore!!
  CHECK_CUDA(cudaMemcpy(a_h, a_d, sizeof(int)*N, cudaMemcpyDefault));

  printf("\nPrinting host memory a_h post-copy\n");
  print_ints(N, a_h, "post_copy_print");

  // Free allocations on host and device
  // TODO Don't forget this one
  delete[] a_h;
  CHECK_CUDA(cudaFree(a_d));

  // End NVTX region
  nvtxRangePop();
}

int main(int argc, char** argv)
{
  CHECK_CUDA(cudaFree(0)); // Initialize the GPU context
  CHECK_CUDA(cudaSetDevice(0)); // Choose device 0 to execute on

  int N = 1024*1024*1024;
  int* a;
  CHECK_CUDA(cudaMalloc(&a, sizeof(int)*N));
  warmup_kernel<<<N/128, 128>>>(N, a);
  CHECK_CUDA(cudaFree(a));
  CHECK_CUDA(cudaDeviceSynchronize());
                                
  N = 10;

  cuda_malloc_managed_test(N);
}

