#pragma once

#include <cusparse.h>
#include <vector>
#include <fstream>
#include <nvtx3/nvToolsExt.h>
#include <iostream>

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        exit(EXIT_FAILURE);                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        exit(EXIT_FAILURE);                                                   \
    }                                                                          \
}

struct MatVecCSR
{
  int nrows;
  int ncols;
  int nnz;
  int* rows;
  int* cols;
  double* vals;

  double alpha;
  double beta;
  double* x;
};

struct TimingHelper
{
  cudaEvent_t beg_event;
  cudaEvent_t end_event;

  TimingHelper() {
    CHECK_CUDA(cudaEventCreate(&beg_event));
    CHECK_CUDA(cudaEventCreate(&end_event));
  }

  void begin()
  {
    CHECK_CUDA(cudaEventRecord(beg_event));
  }

  float end()
  {
    CHECK_CUDA(cudaEventRecord(end_event));
    CHECK_CUDA(cudaEventSynchronize(end_event));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, beg_event, end_event));
    return elapsed_ms;
  }

  ~TimingHelper() {
    CHECK_CUDA(cudaEventDestroy(beg_event));
    CHECK_CUDA(cudaEventDestroy(end_event));
  }
};

struct CusparseData
{
    cudaDataType matType = CUDA_R_64F;
    cusparseHandle_t handle = nullptr;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    cusparseSpMVAlg_t alg = CUSPARSE_SPMV_CSR_ALG1;
    void* tmpBuf = nullptr;

    CusparseData(MatVecCSR& Ax, double* y)
    {
      CHECK_CUSPARSE( cusparseCreate(&handle) );

      CHECK_CUSPARSE( cusparseCreateCsr(
            &matA,
            Ax.nrows,
            Ax.ncols,
            Ax.nnz,
            Ax.rows,
            Ax.cols,
            Ax.vals,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO,
            matType));

      CHECK_CUSPARSE( cusparseCreateDnVec(&vecX, Ax.ncols, Ax.x, matType) );
      CHECK_CUSPARSE( cusparseCreateDnVec(&vecY, Ax.nrows, y, matType) );

      size_t bufferSize = 0;

      CHECK_CUSPARSE( cusparseSpMV_bufferSize(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &Ax.alpha,
            matA,
            vecX,
            &Ax.beta,
            vecY,
            matType,
            alg,
            &bufferSize));

      std::cout << "Initialising bufferSize " << bufferSize << "B" << std::endl;

      CHECK_CUDA(cudaMalloc(&tmpBuf, bufferSize));
      CHECK_CUDA(cudaDeviceSynchronize());
    }

    ~CusparseData()
    {
      CHECK_CUSPARSE( cusparseDestroySpMat(matA) );
      CHECK_CUSPARSE( cusparseDestroyDnVec(vecX) );
      CHECK_CUSPARSE( cusparseDestroyDnVec(vecY) );
      CHECK_CUDA(cudaFree(tmpBuf));
    }
};

void read_matrix(MatVecCSR& Ax, std::string& fname);

void validate_spmv(const char* kernel_name, int nrows, int nnz, double* y, double* y_ref);

void initialize_poisson_nz_counts(
  int nx, int ny, int nz, int Anrows, int& Axnnz, int* Arows, int* nz_counts)
{
  thrust::counting_iterator<int> ci(0);
  thrust::for_each(ci, ci + Anrows,
      [=] __device__ (int id)
      {
        int i = id % nx;
        int j = (id / nx) % ny;
        int k = id / (nx*ny);

        int nnz = 0;

        auto test_non_zero = [=](int ioff, int joff, int koff, int& nnz)
        {
          int ind = (i + ioff) + (j + joff)*nx + (k + koff)*nx*ny;
          if(ind >= 0 && ind < Anrows)
          {
            nnz++;
          }
        };

        test_non_zero(0, 0, 0, nnz);
        test_non_zero(1, 0, 0, nnz);
        test_non_zero(0, 1, 0, nnz);
        test_non_zero(0, 0, 1, nnz);
        test_non_zero(-1, 0, 0, nnz);
        test_non_zero(0, -1, 0, nnz);
        test_non_zero(0, 0, -1, nnz);

        nz_counts[id] = nnz;
      });

  Axnnz = thrust::reduce(thrust::device, nz_counts, nz_counts + Anrows);
  thrust::exclusive_scan(thrust::device, nz_counts, nz_counts + Anrows, Arows);
  CHECK_CUDA(cudaMemcpy(&Arows[Anrows], &Axnnz, sizeof(int), cudaMemcpyDefault));
}

void initialize_poisson_matrix(
  int nx, int ny, int nz, int Anrows, int* nz_counts,
  int* Acols, int* Arows, double* Avals)
{
  thrust::counting_iterator<int> ci(0);
  thrust::for_each(ci, ci + Anrows,
      [=] __device__ (int id)
      {
          int i = id % nx;
          int j = (id / nx) % ny;
          int k = id / (nx*ny);

          int nnz = 0;
          auto set_non_zero = [=](int ioff, int joff, int koff, int& nnz)
          {
              int ind = (i + ioff) + (j + joff)*nx + (k + koff)*nx*ny;
              if(ind >= 0 && ind < Anrows)
              {
                  int row_offset = Arows[id];
                  int col = row_offset + nnz;
                  Acols[col] = ind;
                  nnz++;

                  // Populate Ax with random numbers between 1 and 2
                  thrust::default_random_engine rng;
                  thrust::uniform_real_distribution<double> dist(1.0, 2.0);
                  rng.discard(col);
                  Avals[col] = dist(rng);
              }
          };

          set_non_zero(0, 0, 0, nnz);
          set_non_zero(1, 0, 0, nnz);
          set_non_zero(0, 1, 0, nnz);
          set_non_zero(0, 0, 1, nnz);
          set_non_zero(-1, 0, 0, nnz);
          set_non_zero(0, -1, 0, nnz);
          set_non_zero(0, 0, -1, nnz);
      });
}

void initialize_poisson_vectors(
  int Anrows, double* x)
{
  thrust::counting_iterator<int> ci(0);
  thrust::for_each(ci, ci + Anrows,
      [=] __device__ (int id)
      {
          // Populate A with random numbers between 1 and 2
          thrust::default_random_engine rng;
          thrust::uniform_real_distribution<double> dist(1.0, 2.0);
          rng.discard(id);
          x[id] = dist(rng);
      });
}


