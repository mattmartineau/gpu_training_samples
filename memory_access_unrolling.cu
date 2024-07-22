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

__global__ void memory_access_test(int N, double* a, double val)
{
  int id = blockIdx.x*blockDim.x + threadIdx.x;

  // TODO Add unrolling
  for(int i = id; i < N; i += gridDim.x*blockDim.x) {
      a[i] = val;
  }
}

// CUDA example of memory access requiring unrolling
void memory_access_unrolling_test()
{
  printf("\n\n***Running %s\n\n", __func__);
  nvtxRangePush(__func__);

  // Large problem this time!
  int N = 1024*1024*1024;

  // Make space for the array we are reducing
  // Initialise all of the array to 1 (ignore sloppy launch params!)
  double* a;
  CHECK_CUDA(cudaMalloc(&a, sizeof(double)*N));

  cudaDeviceProp deviceProp;
  CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, 0));
  int nblocks = deviceProp.multiProcessorCount;

  TimingHelper outer_timer;
  outer_timer.begin();
  memory_access_test<<<nblocks, 32>>>(N, a, 1.0);
  float elapsed_ms = outer_timer.end();
  printf("elapsed_ms %.4f\n", elapsed_ms);

  CHECK_CUDA(cudaFree(a));
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
                                
  N = 512;
  memory_access_unrolling_test();
}

