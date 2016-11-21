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

//! Functionality for performing gaussian filtering

#ifndef GAUSSFILTER_CU
#define GAUSSFILTER_CU

#include <stdio.h>
//#include <npp.h>
#include "gaussFilterConvolution.cuh"
//#include "gaussfilter_kernel.cu"
#include "helper_cuda.h"


int iDivUp(int a, int b)
{
  return (a + b - 1) / b;
  //return (a % b != 0) ? (a / b + 1) : (a / b);
}


//!/////////////////////////////////////////////////////////////////////////////
//! General Functions
//!/////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//! Generate 1D Gaussian convolution kernel
//! @param kernel    resulting kernel (necassary memory will be allocated)
//! @param sigma     sigma
//! @param klength   klength of the kernel
////////////////////////////////////////////////////////////////////////////////
int generateGaussianKernel(float** kernel, float sigma, int klength)
{
  // check for valid filter length
  if ((klength % 2) == 0)
  {
    fprintf(stderr, "Error: Convolution Kernel length even\n");
    return -1;
  }

  // allocate memory for kernel
  *kernel = (float*)malloc(sizeof(float) * klength);

  // sum for normalization
  float sum = 0;

  // compute kernel values
  int mid_point = (int)floor(klength/2.0f);
  for( int i = 0; i < klength; i++)
  {
    // generate value
    (*kernel)[i] = exp(-(float)abs(i-mid_point)*(float)abs(i-mid_point)/(2*sigma*sigma));

    // update sum for normalization
    sum += (*kernel)[i];
  }

  // normalize kernel
  for(int i = 0; i < klength; i++)
    (*kernel)[i] /= sum;

  return 0;
}

texture<float, cudaTextureType1D, cudaReadModeElementType> gaussKernelTex_;

template<int klength, typename T>
__global__ void GaussXKernel(T* in, T* out, uint3 vSize)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z;

  unsigned int idx = x + y*vSize.x + z*vSize.x*vSize.y;

  if (idx >= (vSize.x*vSize.y*vSize.z))
    return;
  
  float v = in[idx];//surf2DLayeredread<float>(in, x*4, y, z, cudaBoundaryModeClamp);
  
  if(v != -1)
  {
    v = v * tex1Dfetch(gaussKernelTex_, 0);
 
    #pragma unroll
    for (int i = 1; i < (klength + 1) / 2; ++i)
    {
      T kv1 = (x + i) < vSize.x ? max(0.0f, in[(x + i) + y*vSize.x + z*vSize.x*vSize.y]) : 0.0f;
      T kv2 = (x - i) >= 0 ? max(0.0f, in[(x - i) + y*vSize.x + z*vSize.x*vSize.y]) : 0.0f;
      v = v + tex1Dfetch(gaussKernelTex_, i) * (kv1 + kv2);
    }
  }

 // surf2DLayeredwrite(v, out, x*4, y, z, cudaBoundaryModeZero);
  out[idx] = v;
}

template<typename T>
__global__ void GaussXKernelGeneral(int klength, T* in, T* out, uint3 vSize)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z;

  unsigned int idx = x + y*vSize.x + z*vSize.x*vSize.y;

  if (idx >= (vSize.x*vSize.y*vSize.z))
    return;
  
  float v = in[idx];//float v = surf2DLayeredread<float>(in, x*4, y, z, cudaBoundaryModeClamp);
  
  if(v != -1)
  {
    v = v * tex1Dfetch(gaussKernelTex_, 0);

    for (int i = 1; i < (klength + 1) / 2; ++i)
    {
      T kv1 = (x + i) < vSize.x ? max(0.0f, in[(x + i) + y*vSize.x + z*vSize.x*vSize.y]) : 0.0f;
      T kv2 = (x - i) >= 0 ? max(0.0f, in[(x - i) + y*vSize.x + z*vSize.x*vSize.y]) : 0.0f;
      v = v + tex1Dfetch(gaussKernelTex_, i) * (kv1 + kv2);
    }
  }
 
  out[idx] = v; //surf2DLayeredwrite(v, out, x*4, y, z, cudaBoundaryModeZero);
}

template<int klength, typename T>
__global__ void GaussYKernel(T* in, T* out, uint3 vSize)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z;

  unsigned int idx = x + y*vSize.x + z*vSize.x*vSize.y;
  
  if (idx >= (vSize.x*vSize.y*vSize.z))
    return;

  float v = in[idx];//float v = surf2DLayeredread<float>(in, x*4, y, z, cudaBoundaryModeClamp);
  
  if(v != -1)
  {
    v = v * tex1Dfetch(gaussKernelTex_, 0);
    #pragma unroll
    for (int i = 1; i < (klength + 1) / 2; ++i)
    {
      /////////////////////////////////////
      // test code to solve mem-check error
      if ( ( (x + (y + i)*vSize.x + z*vSize.x*vSize.y) >= (vSize.x*vSize.y*vSize.z) )
        || ( (x + (y - i)*vSize.x + z*vSize.x*vSize.y) < 0 ) )
        continue;
      /////////////////////////////////////

      T kv1 = (y + i) < vSize.y ? max(0.0f, in[(x) + (y + i)*vSize.x + z*vSize.x*vSize.y]) : 0.0f;
      T kv2 = (y - i) >= 0 ? max(0.0f, in[(x) + (y - i)*vSize.x + z*vSize.x*vSize.y]) : 0.0f;
      v = v + tex1Dfetch(gaussKernelTex_, i) * (kv1 + kv2);
    }
  }
 
  out[idx] = v;//surf2DLayeredwrite(v, out, x*4, y, z, cudaBoundaryModeZero);
}

template<typename T>
__global__ void GaussYKernelGeneral(int klength, T* in, T* out, uint3 vSize)
{ 
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z;

  unsigned int idx = x + y*vSize.x + z*vSize.x*vSize.y;

  if (idx >= (vSize.x*vSize.y*vSize.z))
    return;

  float v = in[idx];//float v = surf2DLayeredread<float>(in, x*4, y, z, cudaBoundaryModeClamp);
  
  if(v != -1)
  {
    v = v * tex1Dfetch(gaussKernelTex_, 0);
  
    for (int i = 1; i < (klength + 1) / 2; ++i)
    {
      /////////////////////////////////////
      // test code to solve mem-check error
      if ( ( (x + (y + i)*vSize.x + z*vSize.x*vSize.y) >= (vSize.x*vSize.y*vSize.z) )
        || ( (x + (y - i)*vSize.x + z*vSize.x*vSize.y) < 0 ) )
        continue;
      /////////////////////////////////////

      T kv1 = (y + i) < vSize.y ? max(0.0f, in[(x)+(y + i)*vSize.x + z*vSize.x*vSize.y]) : 0.0f;
      T kv2 = (y - i) >= 0 ? max(0.0f, in[(x)+(y - i)*vSize.x + z*vSize.x*vSize.y]) : 0.0f;
      v = v + tex1Dfetch(gaussKernelTex_, i) * (kv1 + kv2);
    }
  }
  out[idx] = v;// surf2DLayeredwrite(v, out, x * 4, y, z, cudaBoundaryModeZero);
}


////////////////////////////////////////////////////////////////////////////////
//! Performes optimized gaussian filtering of a stack of image (x,y direction
//! while slices are stacked up along z
//! @param input          pointer to input image stack
//! @param output         pointer to output image stack
//! @param temp           pointer to temp image stack
//! @param width          width of the image
//! @param height         height of the image
//! @param slices         num slices
//! @param pitchX/Y       image sizes
//! @param num_ch         number of channels in the image
//! @param sigma          sigma parameter to construct kernel
////////////////////////////////////////////////////////////////////////////////
template <typename T>
int patchBasedFilterGaussStack(int dev, T* input, T* output, uint3 vSize,
                 unsigned int slices, float sigma)
{
  int ret = 0;
  unsigned int width = vSize.x;
  unsigned int height = vSize.y;

  //determine filter length
  int klength = max(min((int)(sigma*5),MAX_LENGTH_SK),7);
  //printf("klength %d \n", klength);
  klength -= 1-klength%2;

  // get device number from the main function 
  // int dev;
  // cudaGetDevice(&dev);
  // printf("dev = %d\n", dev );
  checkCudaErrors(cudaSetDevice(dev));

  static int lastKLength[128] =      {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
  static float lastsigma[128] =      { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  static float* d_GaussKoeffs[128] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  if(lastKLength[dev] != klength || lastsigma[dev] != sigma)
  {
    if(lastKLength[dev] != klength)
    {
      if(d_GaussKoeffs[dev] != 0)
        cudaFree(d_GaussKoeffs[dev]);
      cudaMalloc(&d_GaussKoeffs[dev], sizeof(float)*(klength+1)/2);
    }

    // generate kernel
    float* kernel = NULL;
    ret = generateGaussianKernel(&kernel, sigma, klength);
    if (ret)
    {
      fprintf(stderr, "Error in CUDA FilterGaussStack(): Could not generate Kernel\n");
      return ret;
    }

    cudaMemcpy(d_GaussKoeffs[dev], kernel + klength/2, (klength+1)/2*sizeof(float), cudaMemcpyHostToDevice);

    free(kernel);

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    checkCudaErrors(cudaBindTexture(0, gaussKernelTex_, d_GaussKoeffs[dev], cudaCreateChannelDesc<float>(),(klength+1)/2*sizeof(float)));

    gaussKernelTex_.addressMode[0] = cudaAddressModeClamp;
    gaussKernelTex_.filterMode = cudaFilterModePoint;
    gaussKernelTex_.normalized = false;

    lastsigma[dev] = sigma;
    lastKLength[dev] = klength;
  }

  T* d_temp;
  checkCudaErrors(cudaMalloc(&d_temp, sizeof(T)*vSize.x*vSize.y*vSize.z));
  checkCudaErrors(cudaMemset(d_temp, 0, sizeof(T)*vSize.x*vSize.y*vSize.z));

  //filter (with optimizations for special cases)
  const int blockSize1 = 32;
  const int blockSize2 = 32;

  dim3 blockx(blockSize1, blockSize2);
  dim3 gridx(iDivUp(width, blockSize1), iDivUp(height, blockSize2), slices);
  dim3 blocky(blockSize2, blockSize1);
  dim3 gridy(iDivUp(width, blockSize2), iDivUp(height, blockSize1), slices);

  switch(klength)
  {
  case 7:
    GaussXKernel<7> << <gridx, blockx >> >(input, d_temp, vSize);
    GaussYKernel<7> << <gridy, blocky >> >(d_temp, output, vSize);
    break;
  case 9:
    GaussXKernel<9> << <gridx, blockx >> >(input, d_temp, vSize);
    GaussYKernel<9> << <gridy, blocky >> >(d_temp, output, vSize);
    break;
  case 11:
    GaussXKernel<11> << <gridx, blockx >> >(input, d_temp, vSize);
    GaussYKernel<11> << <gridy, blocky >> >(d_temp, output, vSize);
    break;
  case 13:
    GaussXKernel<13> << <gridx, blockx >> >(input, d_temp, vSize);
    GaussYKernel<14> << <gridy, blocky >> >(d_temp, output, vSize);
    break;
  case 15:
    GaussXKernel<15> << <gridx, blockx >> >(input, d_temp, vSize);
    GaussYKernel<15> << <gridy, blocky >> >(d_temp, output, vSize);
    break;
  default:
    GaussXKernelGeneral << <gridx, blockx >> >(klength, input, d_temp, vSize);
    GaussYKernelGeneral << <gridy, blocky >> >(klength, d_temp, output, vSize);
    break;
  }

  checkCudaErrors(cudaFree(d_temp));
  return ret;
}


template int patchBasedFilterGaussStack<float>(int dev, float* input, float* output, uint3 vSize, unsigned int slices, float sigma);
template int patchBasedFilterGaussStack<double>(int dev, double* input, double* output, uint3 vSize, unsigned int slices, float sigma);

#endif // GAUSSFILTER_CU
