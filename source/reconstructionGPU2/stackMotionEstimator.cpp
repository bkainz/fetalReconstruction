/*=========================================================================
* GPU accelerated motion compensation for MRI
*
* Copyright (c) 2016 Bernhard Kainz, Amir Alansary, Maria Kuklisova-Murgasova,
* Kevin Keraudren, Markus Steinberger
* (b.kainz@imperial.ac.uk)
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
* IN THE SOFTWARE.
=========================================================================*/

#if HAVE_CULA

#include "stackMotionEstimator.h"
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/extrema.h>
#include <thrust/count.h>

stackMotionEstimator::stackMotionEstimator()
{
  status = culaInitialize();
  checkStatus(status);
}


stackMotionEstimator::~stackMotionEstimator()
{
  culaShutdown();
}

struct is_larger_eps
{
  float alpha;
  is_larger_eps(float eps){alpha = eps;};
  __host__ __device__
    bool operator()(float &x)
  {
    return x > alpha;
  }
};

struct is_larger_zero
{
  __host__ __device__
    bool operator()(const int &x)
  {
    return (x > 0);
  }
};

float stackMotionEstimator::evaluateStackMotion(irtkRealImage* img)
{

  //TODO do this non static, move shtdown and init of CULA in cons/destructor

  double min_, max_;
  img->GetMinMax(&min_,&max_);
  printf("min: %f, max: %f \n", min_, max_);

  irtkGenericImage<float> imgN;
  imgN.Initialize(img->GetImageAttributes());
  for(int i = 0; i < imgN.GetNumberOfVoxels(); i++)
  {
    imgN.GetPointerToVoxels()[i] = (float)((img->GetPointerToVoxels()[i]-min_)/(max_-min_));
  }

  //we take only central slices into account
  unsigned int slicesElems = imgN.GetX()*imgN.GetY();
  unsigned int aThirdSlices = (imgN.GetZ()/3.0);

  //use low rank approximation paradigm
  int M = slicesElems; //one image per row
  int N = aThirdSlices; // num rows

  int LDA = M;
  int LDU = M;
  int LDVT = N;

  culaDeviceFloat* A = NULL;
  culaDeviceFloat* S = NULL;
  culaDeviceFloat* U = NULL;
  culaDeviceFloat* VT = NULL;

  char jobu = 'N';
  char jobvt = 'N';

  printf("testing motion %d %d %d MB\n", M, N, LDU, LDU*M*sizeof(culaDeviceFloat)/1024/1024);

  unsigned int num_ev = imin(M,N);
  //cudaMalloc output
  checkCudaErrors(cudaMalloc((void**)&A, M*N*sizeof(culaDeviceFloat))); 
  checkCudaErrors(cudaMalloc((void**)&S, num_ev*sizeof(culaDeviceFloat))); 

  checkCudaErrors(cudaMemcpy(A, imgN.GetPointerToVoxels() /*+ slicesElems*aThirdSlices*/, M*N*sizeof(culaDeviceFloat), cudaMemcpyHostToDevice));


  //TODO #if USE_CPU
  status = culaDeviceSgesvd(jobu, jobvt, M, N, A, LDA, S, U, LDU, VT, LDVT);
  checkStatus(status);

  float* S_host = (float*)malloc(num_ev*sizeof(culaDeviceFloat));
  checkCudaErrors(cudaMemcpy(S_host, S, num_ev*sizeof(culaDeviceFloat), cudaMemcpyDeviceToHost));

  //why does thrust not work here? do we need to compile that with nvcc
  //int result = thrust::count_if(S, S+N, is_larger_zero());
  // thrust::transform(A, A+N, A, ptr_addon, plus<float>());

  double normAll = 0;
  for (int i = 0; i < num_ev; i++)
  {
    normAll += S_host[i] * S_host[i];
  }
  normAll = sqrt(normAll);
  std::cout << "normAll: " << normAll << std::endl;

  double t = 0.99;
  double et = 0;
  int r_min = -1;

  for (int r = 0; r < num_ev; r++)
  {
    double normA = 0;
    for (int i = 0; i < r; i++)
    {
      normA += S_host[i] * S_host[i];
    }
    normA = sqrt(normA);

    double error = normA / normAll;

    if (error < t)
    {
      et = error;
      r_min = r;
    }

    std::cout << "error: " << error << std::endl;
  }

  std::cout << "error fin: " << et << " r_min: " << r_min << std::endl;

  cudaFree(A);
  cudaFree(S); 

  printf("done!\n");

  return (et) * r_min;
}

void stackMotionEstimator::checkStatus(culaStatus status)
{
  char buf[256];

  if(!status)
    return;

  culaGetErrorInfoString(status, culaGetErrorInfo(), buf, sizeof(buf));
  printf("%s\n", buf);

  culaShutdown();
  exit(EXIT_FAILURE);
}

#endif