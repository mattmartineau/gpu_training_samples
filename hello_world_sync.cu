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

__global__ void hello_world_kernel(int N, int* a)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  if(i < N) {
    a[i] = 1337;
  }

  printf("Hello world from thread %d\n", i);
}

// CUDA example with device sync
void hello_world_sync_test(const int N)
{
  printf("\n\n***Running %s\n\n", __func__);
  nvtxRangePush(__func__);

  // Initialize some host memory and initialize
  int* a_m;
  CHECK_CUDA(cudaMallocManaged(&a_m, sizeof(int)*N));
  for(int i = 0; i < N; ++i) {
    a_m[i] = -1;
  }

  printf("Printing managed memory in a_m pre-kernel\n");
  print_ints(N, a_m, "pre_copy_print");

  // Overwrite a_m with GPU to 1337
  // a_m = 1337
  int block_size = 128;
  int nblocks = N/block_size + 1;
  hello_world_kernel<<<nblocks, block_size>>>(N, a_m);

  // XXX Add synchronization

  printf("\nPrinting managed memory post-kernel\n");
  print_ints(N, a_m, "post_copy_print");

  // Free allocations on host and device
  CHECK_CUDA(cudaFree(a_m));

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

  hello_world_sync_test(N);
}

