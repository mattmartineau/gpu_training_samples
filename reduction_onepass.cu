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

__global__ void initialize_a_red(int N, double* a, double val)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  if(i < N) {
    a[i] = val;
  }
}

template <int block_size>
__global__ void onepass_reduction_kernel_test(int N, double* a, double* sum_private)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  __shared__ double sum[block_size];
  sum[threadIdx.x] = a[i];
  __syncthreads();

  for(int t = block_size/2; t > 0; t /= 2) {
      if(threadIdx.x < t) {
          sum[threadIdx.x] += sum[threadIdx.x + t];
      }
      __syncthreads();
  }

  // TODO Implement a one-pass reduction by using atomic here
}

// CUDA example of onepass reduction
void reduction_onepass_test()
{
  printf("\n\n***Running %s\n\n", __func__);
  nvtxRangePush(__func__);

  int N = 1024*1024*1024;

  double* a;
  CHECK_CUDA(cudaMalloc(&a, sizeof(double)*N));
  initialize_a_red<<<N/128, 128>>>(N, a, 1.0);

  // Only need a single element
  double* sum_private;
  CHECK_CUDA(cudaMalloc(&sum_private, sizeof(double)));
  CHECK_CUDA(cudaMemset(sum_private, 0, sizeof(double)));

  // Perform the reduction
  constexpr int block_size = 1024;
  int nblocks = N / block_size;

  TimingHelper outer_timer;
  outer_timer.begin();
  onepass_reduction_kernel_test<block_size><<<nblocks, block_size>>>(N, a, sum_private);
  float elapsed_ms = outer_timer.end();
  printf("elapsed_ms %.4f\n", elapsed_ms);

  double sum;
  CHECK_CUDA(cudaMemcpy(&sum, sum_private, sizeof(double), cudaMemcpyDefault));

  // Check we validate
  if(sum != (double)N) {
      printf("Failed validation sum=%.2f, n=%.2f\n", sum, (double)N);
  }
  else {
      printf("Successful validation sum=%.2f, n=%.2f\n", sum, (double)N);
  }

  CHECK_CUDA(cudaFree(a));
  CHECK_CUDA(cudaFree(sum_private));
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
                                
  reduction_onepass_test();
}

