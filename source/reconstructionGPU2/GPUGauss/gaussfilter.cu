/*=========================================================================
Library   : Image Registration Toolkit (IRTK)
Copyright : Imperial College, Department of Computing
Visual Information Processing (VIP), 2011 onwards
Date      : $Date: 2013-11-15 14:36:30 +0100 (Fri, 15 Nov 2013) $
Version   : $Revision: 1 $
Changes   : $Author: bkainz $

Copyright (c) 2014, Bernhard Kainz, Markus Steinberger,
Maria Murgasova, Kevin Keraudren
All rights reserved.

If you use this work for research we would very much appreciate if you cite
Bernhard Kainz, Markus Steinberger, Maria Kuklisova-Murgasova, Christina Malamateniou,
Wolfgang Wein, Thomas Torsney-Weir, Torsten Moeller, Mary Rutherford,
Joseph V. Hajnal and Daniel Rueckert:
Fast Volume Reconstruction from Motion Corrupted 2D Slices.
IEEE Transactions on Medical Imaging, in press, 2015

IRTK IS PROVIDED UNDER THE TERMS OF THIS CREATIVE
COMMONS PUBLIC LICENSE ("CCPL" OR "LICENSE"). THE WORK IS PROTECTED BY
COPYRIGHT AND/OR OTHER APPLICABLE LAW. ANY USE OF THE WORK OTHER THAN
AS AUTHORIZED UNDER THIS LICENSE OR COPYRIGHT LAW IS PROHIBITED.

BY EXERCISING ANY RIGHTS TO THE WORK PROVIDED HERE, YOU ACCEPT AND AGREE
TO BE BOUND BY THE TERMS OF THIS LICENSE. TO THE EXTENT THIS LICENSE MAY BE
CONSIDERED TO BE A CONTRACT, THE LICENSOR GRANTS YOU THE RIGHTS CONTAINED
HERE IN CONSIDERATION OF YOUR ACCEPTANCE OF SUCH TERMS AND CONDITIONS.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
=========================================================================*/

//! Functionality for performing gaussian filtering

#ifndef GAUSSFILTER_CU
#define GAUSSFILTER_CU

#include <stdio.h>
#include <npp.h>
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


template<int klength>
__global__ void GaussXKernel(cudaSurfaceObject_t in, cudaSurfaceObject_t out)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z;

  float v = surf2DLayeredread<float>(in, x*4, y, z, cudaBoundaryModeClamp);
  
  if(v != -1)
  {
    v = v * tex1Dfetch(gaussKernelTex_, 0);
 
    #pragma unroll
    for(int i = 1; i < (klength+1)/2; ++i)
      v = v + tex1Dfetch(gaussKernelTex_, i) * (max(0.0f,surf2DLayeredread<float>(in, (x + i)*4, y, z, cudaBoundaryModeClamp)) + 
                                                max(0.0f,surf2DLayeredread<float>(in, (x - i)*4, y, z, cudaBoundaryModeClamp)) );
  }

  surf2DLayeredwrite(v, out, x*4, y, z, cudaBoundaryModeZero);
}


__global__ void GaussXKernelGeneral(int klength, cudaSurfaceObject_t in, cudaSurfaceObject_t out)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z;

  float v = surf2DLayeredread<float>(in, x*4, y, z, cudaBoundaryModeClamp);
  
  if(v != -1)
  {
    v = v * tex1Dfetch(gaussKernelTex_, 0);

    for(int i = 1; i < (klength+1)/2; ++i)
      v = v + tex1Dfetch(gaussKernelTex_, i) * (max(0.0f,surf2DLayeredread<float>(in, (x + i)*4, y, z, cudaBoundaryModeClamp)) + 
                                                max(0.0f,surf2DLayeredread<float>(in, (x - i)*4, y, z, cudaBoundaryModeClamp)) );
  }
 
   surf2DLayeredwrite(v, out, x*4, y, z, cudaBoundaryModeZero);
}

template<int klength>
__global__ void GaussYKernel(cudaSurfaceObject_t in, cudaSurfaceObject_t out)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z;

  float v = surf2DLayeredread<float>(in, x*4, y, z, cudaBoundaryModeClamp);
  
  if(v != -1)
  {
    v = v * tex1Dfetch(gaussKernelTex_, 0);
    #pragma unroll
    for(int i = 1; i < (klength+1)/2; ++i)
      v = v + tex1Dfetch(gaussKernelTex_, i) * (max(0.0f,surf2DLayeredread<float>(in, x*4, y + i, z, cudaBoundaryModeClamp)) + 
                                                max(0.0f,surf2DLayeredread<float>(in, x*4, y - i, z, cudaBoundaryModeClamp)) );
  }
 
  surf2DLayeredwrite(v, out, x*4, y, z, cudaBoundaryModeZero);
}

__global__ void GaussYKernelGeneral(int klength, cudaSurfaceObject_t in, cudaSurfaceObject_t out)
{ 
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z;

  float v = surf2DLayeredread<float>(in, x*4, y, z, cudaBoundaryModeClamp);
  
  if(v != -1)
  {
    v = v * tex1Dfetch(gaussKernelTex_, 0);
  
    for(int i = 1; i < (klength+1)/2; ++i)
      v = v + tex1Dfetch(gaussKernelTex_, i) * (max(0.0f,surf2DLayeredread<float>(in, x*4, y + i, z, cudaBoundaryModeClamp)) + 
                                                max(0.0f,surf2DLayeredread<float>(in, x*4, y - i, z, cudaBoundaryModeClamp)) );
  }
  surf2DLayeredwrite(v, out, x*4, y, z, cudaBoundaryModeZero);
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
int FilterGaussStack(cudaSurfaceObject_t input, cudaSurfaceObject_t output, cudaSurfaceObject_t temp, 
                 unsigned int width, unsigned int height, unsigned int slices, float sigma)
{
  int ret = 0;

  //determine filter length
  int klength = max(min((int)(sigma*5),MAX_LENGTH_SK),7);
  klength -= 1-klength%2;

  int dev;
  cudaGetDevice(&dev);

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
    GaussXKernel<7><<<gridx, blockx>>>(input, temp);
    GaussYKernel<7><<<gridy, blocky>>>(temp, output);
    break;
  case 9:
    GaussXKernel<9><<<gridx, blockx>>>(input, temp);
    GaussYKernel<9><<<gridy, blocky>>>(temp, output);
    break;
  case 11:
    GaussXKernel<11><<<gridx, blockx>>>(input, temp);
    GaussYKernel<11><<<gridy, blocky>>>(temp, output);
    break;
  case 13:
    GaussXKernel<13><<<gridx, blockx>>>(input, temp);
    GaussYKernel<14><<<gridy, blocky>>>(temp, output);
    break;
  case 15:
    GaussXKernel<15><<<gridx, blockx>>>(input, temp);
    GaussYKernel<15><<<gridy, blocky>>>(temp, output);
    break;
  default:
    GaussXKernelGeneral<<<gridx, blockx>>>(klength, input, temp);
    GaussYKernelGeneral<<<gridy, blocky>>>(klength, temp, output);
    break;
  }
  return ret;
}


#endif // GAUSSFILTER_CU
