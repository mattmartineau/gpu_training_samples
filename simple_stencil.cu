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

// CUDA example with stencil
void simple_stencil_test(const int N)
{
  printf("\n\n***Running %s\n\n", __func__);
  nvtxRangePush(__func__);

  int* a_out;
  int* a_ref;
  int* a_orig;
  CHECK_CUDA(cudaMallocManaged(&a_out, sizeof(int)*N));
  CHECK_CUDA(cudaMallocManaged(&a_ref, sizeof(int)*N));
  CHECK_CUDA(cudaMallocManaged(&a_orig, sizeof(int)*N));

  int block_size = 128;
  int nblocks = N/block_size + 1;
  initialize_a<<<nblocks, block_size>>>(N, a_orig);

  // cudaMemcpy is naturally synchronous...
  CHECK_CUDA(cudaMemcpy(a_out, a_orig, sizeof(int)*N, cudaMemcpyDefault));
  CHECK_CUDA(cudaMemcpy(a_ref, a_orig, sizeof(int)*N, cudaMemcpyDefault));

  // CPU stencil kernel
  for(int i = 0; i < N; ++i) {
      if(i > 0) {
          a_ref[i-1] += a_orig[i];
      }
      if(i < N-1) {
          a_ref[i+1] += a_orig[i];
      }
  }

  // TODO implement the GPU kernel
  CHECK_CUDA(cudaDeviceSynchronize());

  // Validation
  for(int i = 0; i < N; ++i) {
      if(a_out[i] != a_ref[i]) {
          printf("a_out[%d]=%d != a_ref[%d]=%d !!\n", i, a_out[i], i, a_ref[i]);
      }
  }

  CHECK_CUDA(cudaFree(a_out));
  CHECK_CUDA(cudaFree(a_orig));
  CHECK_CUDA(cudaFree(a_ref));
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
  simple_stencil_test(N);
}

