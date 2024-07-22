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
__global__ void multipass_reduction_shared_kernel(int N, double* a, double* sum_private)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  __shared__ double sum[block_size];
  sum[threadIdx.x] = a[i];
  __syncthreads();
  
  // TODO Implement shared reduction 
}

// CUDA example of multipass reduction
void reduction_multipass_shared_test()
{
  printf("\n\n***Running %s\n\n", __func__);
  nvtxRangePush(__func__);

  // Large problem this time!
  int N = 1024*1024*1024;

  double* a;
  CHECK_CUDA(cudaMalloc(&a, sizeof(double)*N));
  initialize_a_red<<<N/128, 128>>>(N, a, 1.0);
  CHECK_CUDA(cudaDeviceSynchronize());

  // Only need nblocks private space
  constexpr int block_size = 128;
  int nblocks = N / block_size;

  double* sum_private;
  CHECK_CUDA(cudaMalloc(&sum_private, sizeof(double)*nblocks));

  // Perform the reduction
  multipass_reduction_shared_kernel<block_size><<<nblocks, block_size>>>(N, a, sum_private);

  // Let thrust reduce the private sums
  double sum = thrust::reduce(thrust::device,
          sum_private, sum_private + nblocks, 0, thrust::plus<double>());

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

  N = 512;
  
  reduction_multipass_shared_test();
}

