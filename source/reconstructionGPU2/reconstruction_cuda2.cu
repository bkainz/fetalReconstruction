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

#include "reconstruction_cuda2.cuh"
#include "gaussFilterConvolution.cuh"
#define _USE_MATH_DEFINES
#include <math.h>
#include <thrust/inner_product.h>
#include <stdint.h>
#include <algorithm>

using namespace thrust;

__constant__ int d_directions[13][3];
__constant__ float d_factor[13];
__constant__ Matrix4 d_reconstructedW2I;
__constant__ Matrix4 d_reconstructedI2W;
__constant__ float3 d_PSFdim;
__constant__ uint3 d_PSFsize;
__constant__ Matrix4 d_PSFI2W;
__constant__ Matrix4 d_PSFW2I;
__constant__ float d_quality_factor;
__constant__ float d_use_SINC; //constant flag because of efficency reasons

texture<float, 3, cudaReadModeElementType> psfTex;
texture<float, 3, cudaReadModeElementType > reconstructedTex_;

inline __host__ __device__ int reflect(int M, int x)
{
  return max(0, min(M - 1, x));
}

__forceinline__ __device__ float sq(const float x){
  return x*x;
}

inline __host__ __device__ float G_(float x, float s)
{
  return __step*exp(-x*x / (2.0f*s)) / (sqrt(6.28f*s));
}

inline __host__ __device__ float M_(float m)
{
  return m*__step;
}

void checkGPUMemory()
{
  size_t free_byte;
  size_t total_byte;
  cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);

  if (cudaSuccess != cuda_status){
    printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status));
  }

  float free_db = (float)free_byte;
  float total_db = (float)total_byte;
  float used_db = total_db - free_db;
  printf("GPU memory usage: \nused = %f, free = %f MB, total = %f MB\n",
    used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);
}

__device__ float round_(float x)
{
  return roundf(x);
}

__device__ double atomicAdd(double* address, double val)
{
  unsigned long long int* address_as_ull =
    (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
      __double_as_longlong(val +
      __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}


///////////////////////////////test continous PSF kernels
//in slice coords
__device__ float inline calcPSF(float3 sPos, float3 dim)
{
  const float sigmaz = dim.z / 2.3548f;

#if 1

  if(d_use_SINC)
  {
    // sinc is already 1.2 FWHM
    sPos.x = sPos.x * dim.x / 2.3548f;
    sPos.y = sPos.y * dim.y / 2.3548f;
    float x = sqrt(sPos.x*sPos.x + sPos.y*sPos.y);
    float R = 3.14159265359f * x;
    float si = sin(R) / (R);
    return  si*si * exp((-sPos.z * sPos.z) / (2.0f * sigmaz * sigmaz)); //Bartlett positive sinc
  }
  else
  {
    const float sigmax = 1.2f * dim.x / 2.3548f;
    const float sigmay = 1.2f * dim.y / 2.3548f;

    return exp((-sPos.x * sPos.x) / (2.0f * sigmax * sigmax) - (sPos.y * sPos.y) / (2.0f * sigmay * sigmay)
      - (sPos.z * sPos.z) / (2.0f * sigmaz * sigmaz));
  }

#else
  float val = exp(-sPos.x * sPos.x / (2.0f * sigmax * sigmax) - sPos.y * sPos.y / (2.0f * sigmay * sigmay)
    - sPos.z * sPos.z / (2.0f * sigmaz * sigmaz));
  if (abs(val) < 0.6)
    val = 0;

  return val;
#endif


}

__device__ float inline getPSFParams(float3 &ofsPos, Matrix4 sliceTrans, Matrix4 sliceInvTrans, Matrix4 sliceI2W, Matrix4 sliceW2I,
  int3 cur_ofs, int centre, int dim, float3 slicePos, float3 reconDim, float3 sliceDim/*, bool useTex = true*/)
{
  //optimized version, less readable, saves registers
  float3 psfxyz = d_reconstructedW2I*(sliceTrans*(sliceI2W*slicePos));
  psfxyz = make_float3(round_(psfxyz.x), round_(psfxyz.y), round_(psfxyz.z));

  ofsPos = make_float3(cur_ofs.x + psfxyz.x - centre, cur_ofs.y + psfxyz.y - centre, cur_ofs.z + psfxyz.z - centre);

  psfxyz = sliceW2I*(sliceInvTrans*(d_reconstructedI2W*ofsPos));

  float3 psfofs = psfxyz - slicePos; //only in slice coordintes we are sure about z
  psfofs = make_float3(psfofs.x*sliceDim.x, psfofs.y*sliceDim.y, psfofs.z*sliceDim.z);

  psfxyz = psfofs - d_PSFI2W*make_float3(((d_PSFsize.x - 1) / 2.0), ((d_PSFsize.y - 1) / 2.0), ((d_PSFsize.z - 1) / 2.0));
  return calcPSF(make_float3(psfxyz.x, psfxyz.y, psfxyz.z), sliceDim);
}

__device__ float inline getPSFParamsPrecomp(float3 &ofsPos, const float3& psfxyz, int3 currentoffsetMCenter, Matrix4 combInvTrans, float3 slicePos, float3 sliceDim)
{
  ofsPos = make_float3(currentoffsetMCenter.x + psfxyz.x, currentoffsetMCenter.y + psfxyz.y, currentoffsetMCenter.z + psfxyz.z);

  float3 psfxyz2 = combInvTrans*ofsPos;
  float3 psfofs = psfxyz2 - slicePos; //only in slice coordintes we are sure about z
  psfofs = make_float3(psfofs.x*sliceDim.x, psfofs.y*sliceDim.y, psfofs.z*sliceDim.z);

  psfxyz2 = psfofs - d_PSFI2W*make_float3(((d_PSFsize.x - 1)*0.5f), ((d_PSFsize.y - 1)*0.5f), ((d_PSFsize.z - 1)*0.5f));
  return calcPSF(make_float3(psfxyz2.x, psfxyz2.y, psfxyz2.z), sliceDim);
}

__global__ void gaussianReconstructionKernel3D_tex(int numSlices, float* __restrict slices,
  float* __restrict bias2D, float* __restrict scales, Volume<float> reconstructed,
  Volume<float> reconstructed_volWeigths, Volume<int> sliceVoxel_count,
  Volume<float> v_PSF_sums, float* __restrict mask, uint2 vSize,
  Matrix4* __restrict sliceI2W, Matrix4*  __restrict sliceW2I, Matrix4* __restrict slicesTransformation,
  Matrix4* __restrict slicesInvTransformation, const float3* __restrict d_slicedim, const int* __restrict d_dims, float reconVSize, short step, short slicesPerRun)
{

  const uint3 pos = make_uint3(blockIdx.x* blockDim.x + threadIdx.x,
    blockIdx.y* blockDim.y + threadIdx.y,
    blockIdx.z* blockDim.z + threadIdx.z + step);

  //z is slice index
  if (pos.z >= numSlices || (pos.z - step) >= slicesPerRun)
    return;

  unsigned int idx = pos.x + pos.y*vSize.x + pos.z*vSize.x*vSize.y;
  float s = slices[idx];
  if ((s == -1.0f))
    return;

  float3 sliceDim = d_slicedim[pos.z];

  s = s * exp(-bias2D[idx]) * scales[pos.z];

  float3 slicePos = make_float3(pos.x, pos.y, 0);


  float size_inv = 2.0f * d_quality_factor / reconstructed.dim.x;
  int xDim = round_((sliceDim.x * size_inv));
  int yDim = round_((sliceDim.y * size_inv));
  int zDim = round_((sliceDim.z * size_inv));

#if !USE_INFINITE_PSF_SUPPORT
  int dim = (floor(ceil(sqrt(float(xDim * xDim + yDim * yDim + zDim * zDim)) / d_quality_factor) * 0.5f)) * 2.0f + 3.0f;
  int centre = (dim - 1) / 2;
#else 
  //truncate if value gets close to epsilon
  int dim = MAX_PSF_SUPPORT;
  int centre = (MAX_PSF_SUPPORT - 1) / 2;
#endif

  Matrix4 combInvTrans;
  combInvTrans = sliceW2I[pos.z] * slicesInvTransformation[pos.z] * d_reconstructedI2W;
  float3 psfxyz;
  float3 _psfxyz = d_reconstructedW2I*(slicesTransformation[pos.z] * (sliceI2W[pos.z] * slicePos));
  psfxyz = make_float3(round_(_psfxyz.x), round_(_psfxyz.y), round_(_psfxyz.z));

  float sume = 0;
  for (int z = 0; z < dim; z++)
  {
    for (int y = 0; y < dim; y++)
    {
      float oldPSF = FLT_MAX;
      for (int x = 0; x < dim; x++)
      {
        float3 ofsPos;
        float psfval = getPSFParamsPrecomp(ofsPos, psfxyz, make_int3(x - centre, y - centre, z - centre), combInvTrans, slicePos, sliceDim);
        if (abs(oldPSF - psfval) < PSF_EPSILON) continue;
        oldPSF = psfval;

        uint3 apos = make_uint3(ofsPos.x, ofsPos.y, ofsPos.z);
        if (apos.x < reconstructed.size.x && apos.y < reconstructed.size.y && apos.z < reconstructed.size.z)
        {
          sume += psfval;
        }
      }
    }
  }

  //fix for crazy values at the border -> too accurate ;)
  if (sume > 0.5f)
  {
    v_PSF_sums.set(pos, sume);
  }
  else
  {
    return;
  }

  bool addvoxNum = false;

  for (float z = 0; z < dim; z++)
  {
    for (float y = 0; y < dim; y++)
    {
      float oldPSF = FLT_MAX;
      for (float x = 0; x < dim; x++)
      {
        float3 ofsPos;
        float psfval = getPSFParamsPrecomp(ofsPos, psfxyz, make_int3(x - centre, y - centre, z - centre), combInvTrans, slicePos, sliceDim);
        if (abs(oldPSF - psfval) < PSF_EPSILON) continue;
        oldPSF = psfval;

        uint3 apos = make_uint3(round_(ofsPos.x), round_(ofsPos.y), round_(ofsPos.z)); //NN
        if (apos.x < reconstructed.size.x && apos.y < reconstructed.size.y && apos.z < reconstructed.size.z
          && mask[apos.x + apos.y*reconstructed.size.x + apos.z*reconstructed.size.x*reconstructed.size.y] != 0)
        {
          psfval /= sume;
          atomicAdd(&(reconstructed_volWeigths.data[apos.x + apos.y*reconstructed.size.x + apos.z*reconstructed.size.x*reconstructed.size.y]),
            psfval);
          atomicAdd(&(reconstructed.data[apos.x + apos.y*reconstructed.size.x + apos.z*reconstructed.size.x*reconstructed.size.y]),
            psfval*s);
          addvoxNum = true;
        }

      }
    }
  }


  if (addvoxNum)
  {
    atomicAdd(&(sliceVoxel_count.data[idx]), 1);
  }
}


__global__ void simulateSlicesKernel3D_tex(int numSlices,
  float* __restrict slices, Volume<float> simslices, Volume<float> simweights, Volume<char> siminside,
  Volume<float> reconstructed, float* __restrict mask, float* __restrict v_PSF_sums, uint2 vSize,
  Matrix4* sliceI2W, Matrix4* sliceW2I, Matrix4* slicesTransformation, Matrix4* slicesInvTransformation,
  float3* __restrict d_slicedim, float reconVSize, int step, int slicesPerRun)
{
  const uint3 pos = make_uint3(blockIdx.x * blockDim.x + threadIdx.x,
    blockIdx.y * blockDim.y + threadIdx.y,
    blockIdx.z * blockDim.z + threadIdx.z + step);

  if (pos.z >= numSlices || (pos.z - step) >= slicesPerRun)
    return;

  unsigned int idx = (pos.x + pos.y*vSize.x + pos.z*vSize.x*vSize.y);
  float s = slices[idx];
  if (s == -1.0f)
    return;

  float sume = v_PSF_sums[idx];
  if (sume == 0.0f)
    return;

  float simulated_sliceV = 0;
  bool slice_inside = 0;
  float weight = 0;
  float3 sliceDim = d_slicedim[pos.z];

  float3 slicePos = make_float3(pos.x, pos.y, 0);
  float size_inv = 2.0f * d_quality_factor / reconstructed.dim.x;
  int xDim = round_((sliceDim.x * size_inv));
  int yDim = round_((sliceDim.y * size_inv));
  int zDim = round_((sliceDim.z * size_inv));

#if !USE_INFINITE_PSF_SUPPORT
  int dim = (floor(ceil(sqrt(float(xDim * xDim + yDim * yDim + zDim * zDim)) / d_quality_factor) * 0.5f)) * 2.0f + 3.0f;
  int centre = (dim - 1) / 2;
#else 
  //truncate if value gets close to epsilon
  int dim = MAX_PSF_SUPPORT;
  int centre = (MAX_PSF_SUPPORT - 1) / 2;
#endif

#if 0
  //TODO try if this could be replaced similar to reg slices generation with lin texture
  for (int z = 0; z < dim; z++)
  {
    for (int y = 0; y < dim; y++)
    {
      for (int x = 0; x < dim; x++)
      {
        float3 ofsPos;
        float psfval = getPSFParams(ofsPos, slicesTransformation[pos.z], slicesInvTransformation[pos.z],
          sliceI2W[pos.z], sliceW2I[pos.z], make_int3(x, y, z), centre, dim, slicePos, reconstructed.dim, sliceDim);

        uint3 apos = make_uint3(round_(ofsPos.x), round_(ofsPos.y), round_(ofsPos.z)); //NN
        if (apos.x < reconstructed.size.x && apos.y < reconstructed.size.y && apos.z < reconstructed.size.z &&
          mask[apos.x + apos.y*reconstructed.size.x + apos.z*reconstructed.size.x*reconstructed.size.y] != 0)
        {
          psfval /= sume;
          simulated_sliceV += psfval * reconstructed[apos];
          weight += psfval;

          slice_inside = 1;
}
      }
    }
  }
#else
  Matrix4 combInvTrans;
  combInvTrans = sliceW2I[pos.z] * slicesInvTransformation[pos.z] * d_reconstructedI2W;
  float3 psfxyz;
  float3 _psfxyz = d_reconstructedW2I*(slicesTransformation[pos.z] * (sliceI2W[pos.z] * slicePos));
  psfxyz = make_float3(round_(_psfxyz.x), round_(_psfxyz.y), round_(_psfxyz.z));

  for (int z = 0; z < dim; z++) {
    for (int y = 0; y < dim; y++) {
      float oldPSF = FLT_MAX;
      for (int x = 0; x < dim; x++)
      {
        float3 ofsPos;
        float psfval = getPSFParamsPrecomp(ofsPos, psfxyz, make_int3(x - centre, y - centre, z - centre), combInvTrans, slicePos, sliceDim);
        if (abs(oldPSF - psfval) < PSF_EPSILON) continue;
        oldPSF = psfval;

        uint3 apos = make_uint3(round_(ofsPos.x), round_(ofsPos.y), round_(ofsPos.z)); //NN
        if (apos.x < reconstructed.size.x && apos.y < reconstructed.size.y && apos.z < reconstructed.size.z &&
          mask[apos.x + apos.y*reconstructed.size.x + apos.z*reconstructed.size.x*reconstructed.size.y] != 0)
        {
          psfval /= sume;
          simulated_sliceV += psfval * reconstructed[apos];
          weight += psfval;

          slice_inside = 1;
        }
      }
    }
  }

#endif

  if (weight > 0)
  {
    simslices.set(pos, simulated_sliceV / weight);
    simweights.set(pos, weight);
    siminside.set(pos, slice_inside);
  }
}



__global__ void SuperresolutionKernel3D_tex(unsigned short numSlices, float* __restrict slices, float* __restrict bias, float* __restrict weights,
  float* __restrict simslices, float* __restrict slice_weights, float* __restrict scales, float* __restrict mask,
  float* __restrict v_PSF_sums, Volume<float> addon, float* confidence_map, Matrix4* __restrict sliceI2W, Matrix4* __restrict sliceW2I,
  Matrix4* slicesTransformation, Matrix4* slicesInvTransformation, float3* __restrict d_slicedim, float reconVSize,
  unsigned short step, int slicesPerRun, ushort2 sliceSize, float size)
{

  const uint3 pos = make_uint3(blockIdx.x* blockDim.x + threadIdx.x,
    blockIdx.y* blockDim.y + threadIdx.y,
    blockIdx.z* blockDim.z + threadIdx.z + step);

  //z is slice index
  if (pos.z >= numSlices || (pos.z - step) >= slicesPerRun)
    return;

  unsigned int idx = pos.x + pos.y*sliceSize.x + pos.z*sliceSize.x*sliceSize.y;
  float s = slices[idx];
  if ((s == -1.0f))
    return;

  float sume = v_PSF_sums[idx];
  if ((sume == 0.0))
    return;

  float b = bias[idx];
  float w = weights[idx];
  float ss = simslices[idx];
  float slice_weight = slice_weights[pos.z];
  float scale = scales[pos.z];

  float sliceVal = s * exp(-b) * scale;
  if (ss > 0.0f)
    sliceVal = (sliceVal - ss);
  else
    sliceVal = 0.0f;

  float3 sliceDim = d_slicedim[pos.z];

  float size_inv = 2.0f * d_quality_factor / addon.dim.x;
  int xDim = round_((sliceDim.x * size_inv));
  int yDim = round_((sliceDim.y * size_inv));
  int zDim = round_((sliceDim.z * size_inv));

#if !USE_INFINITE_PSF_SUPPORT
  int dim = (floor(ceil(sqrt(float(xDim * xDim + yDim * yDim + zDim * zDim)) / d_quality_factor) * 0.5f)) * 2.0f + 3.0f;
  int centre = (dim - 1) / 2;
#else 
  //truncate if value gets close to epsilon
  int dim = MAX_PSF_SUPPORT;
  int centre = (MAX_PSF_SUPPORT - 1) / 2;
#endif

#if 0
  //TODO try if this could be replaced similar to reg slices generation with lin texture
  for (short z = 0; z < dim; z++)
  {
    for (short y = 0; y < dim; y++)
    {
      for (short x = 0; x < dim; x++)
      {
        float3 ofsPos;
        float psfval = getPSFParams(ofsPos, slicesTransformation[pos.z], slicesInvTransformation[pos.z],
          sliceI2W[pos.z], sliceW2I[pos.z], make_int3(x, y, z), centre, dim, make_float3((float)pos.x, (float)pos.y, 0), addon.dim, sliceDim);

        ushort3 apos = make_ushort3(round_(ofsPos.x), round_(ofsPos.y), round_(ofsPos.z)); //NN
        if (apos.x < addon.size.x && apos.y < addon.size.y && apos.z < addon.size.z &&
          mask[apos.x + apos.y*addon.size.x + apos.z*addon.size.x*addon.size.y] != 0)
        {
          psfval /= sume;
          atomicAdd(&(addon.data[apos.x + apos.y*addon.size.x + apos.z*addon.size.x*addon.size.y]),
            psfval * w * slice_weight * sliceVal);
          atomicAdd(&(confidence_map[apos.x + apos.y*addon.size.x + apos.z*addon.size.x*addon.size.y]),
            psfval * w * slice_weight);
}
      }
    }
  }
#else
  Matrix4 combInvTrans;
  combInvTrans = sliceW2I[pos.z] * slicesInvTransformation[pos.z] * d_reconstructedI2W;
  float3 psfxyz;
  float3 slicePos = make_float3(pos.x, pos.y, 0);
  float3 _psfxyz = d_reconstructedW2I*(slicesTransformation[pos.z] * (sliceI2W[pos.z] * slicePos));
  psfxyz = make_float3(round_(_psfxyz.x), round_(_psfxyz.y), round_(_psfxyz.z));

  for (int z = 0; z < dim; z++){
    for (int y = 0; y < dim; y++) {
      float oldPSF = FLT_MAX;
      for (int x = 0; x < dim; x++)
      {
        float3 ofsPos;
        float psfval = getPSFParamsPrecomp(ofsPos, psfxyz, make_int3(x - centre, y - centre, z - centre), combInvTrans, slicePos, sliceDim);
        if (abs(oldPSF - psfval) < PSF_EPSILON) continue;
        oldPSF = psfval;

        uint3 apos = make_uint3(ofsPos.x, ofsPos.y, ofsPos.z);
        if (apos.x < addon.size.x && apos.y < addon.size.y && apos.z < addon.size.z &&
          mask[apos.x + apos.y*addon.size.x + apos.z*addon.size.x*addon.size.y] != 0)
        {
          psfval /= sume;
          atomicAdd(&(addon.data[apos.x + apos.y*addon.size.x + apos.z*addon.size.x*addon.size.y]),
            psfval * w * slice_weight * sliceVal);
          atomicAdd(&(confidence_map[apos.x + apos.y*addon.size.x + apos.z*addon.size.x*addon.size.y]),
            psfval * w * slice_weight);
        }
      }
    }
  }
#endif
}


__global__ void normalizeBiasKernel3D_tex(int numSlices, Volume<float> slices, Volume<float> bias2D,
  float* scales, float* __restrict mask, float* v_PSF_sums, Matrix4* __restrict sliceI2W, Matrix4* __restrict sliceW2I,
  Matrix4* slicesTransformation, Matrix4* slicesInvTransformation, float3* __restrict d_slicedim, float reconVSize,
  Volume<float> bias, Volume<float> reconVolWeights, Volume<float> volWeights, int step, int slicesPerRun)
{
  const uint3 pos = make_uint3(blockIdx.x * blockDim.x + threadIdx.x,
    blockIdx.y * blockDim.y + threadIdx.y,
    blockIdx.z * blockDim.z + threadIdx.z + step);

  //z is slice index
  if (pos.z >= numSlices || (pos.z - step) >= slicesPerRun)
    return;
  unsigned int idx = pos.x + pos.y*slices.size.x + pos.z*slices.size.x*slices.size.y;

  float s = slices[pos];
  if ((s == -1.0))
    return;
  float sume = v_PSF_sums[idx];
  if ((sume == 0.0))
    return;

  float nbias = bias2D[pos];
  float scale = scales[pos.z];

  if (scale > 0)
  {
    nbias -= log(scale);
}

  float3 sliceDim = d_slicedim[pos.z];

  float size_inv = 2.0f * d_quality_factor / bias.dim.x;
  int xDim = round_((sliceDim.x * size_inv));
  int yDim = round_((sliceDim.y * size_inv));
  int zDim = round_((sliceDim.z * size_inv));

#if !USE_INFINITE_PSF_SUPPORT
  int dim = (floor(ceil(sqrt(float(xDim * xDim + yDim * yDim + zDim * zDim)) / d_quality_factor) * 0.5f)) * 2.0f + 3.0f;
  int centre = (dim - 1) / 2;
#else 
  //truncate if value gets close to epsilon
  int dim = MAX_PSF_SUPPORT;
  int centre = (MAX_PSF_SUPPORT - 1) / 2;
#endif

  Matrix4 combInvTrans;
  combInvTrans = sliceW2I[pos.z] * slicesInvTransformation[pos.z] * d_reconstructedI2W;
  float3 psfxyz;
  float3 slicePos = make_float3(pos.x, pos.y, 0);
  float3 _psfxyz = d_reconstructedW2I*(slicesTransformation[pos.z] * (sliceI2W[pos.z] * slicePos));
  psfxyz = make_float3(round_(_psfxyz.x), round_(_psfxyz.y), round_(_psfxyz.z));

  //TODO try if this could be replaced similar to reg slices generation with lin texture
  for (int z = 0; z < dim; z++)
  {
    for (int y = 0; y < dim; y++)
    {
      float oldPSF = FLT_MAX;
      for (int x = 0; x < dim; x++)
      {
        float3 ofsPos;
        //float psfval = getPSFParams(ofsPos, slicesTransformation[pos.z], slicesInvTransformation[pos.z],
        //  sliceI2W[pos.z], sliceW2I[pos.z], make_int3(x,y,z), centre, dim, make_float3((float)pos.x, (float)pos.y, 0), bias.dim, sliceDim);
        float psfval = getPSFParamsPrecomp(ofsPos, psfxyz, make_int3(x - centre, y - centre, z - centre), combInvTrans, slicePos, sliceDim);
        if (abs(oldPSF - psfval) < PSF_EPSILON) continue;
        oldPSF = psfval;

        uint3 apos = make_uint3(round_(ofsPos.x), round_(ofsPos.y), round_(ofsPos.z)); //NN
        if (apos.x < bias.size.x && apos.y < bias.size.y && apos.z < bias.size.z &&
          apos.x >= 0 && apos.y >= 0 && apos.z >= 0 && mask[apos.x + apos.y*bias.size.x + apos.z*bias.size.x*bias.size.y] != 0)
        {
          psfval /= sume;
          float val = psfval*nbias;
          atomicAdd(&(bias.data[apos.x + apos.y*bias.size.x + apos.z*bias.size.x*bias.size.y]),
            val);
          atomicAdd(&(volWeights.data[apos.x + apos.y*volWeights.size.x + apos.z*volWeights.size.x*volWeights.size.y]),
            psfval);
        }
      }
    }
  }

}


///////////////////////////////test area PSF texture end

void threadStartCaller(int i, Reconstruction& reconstruction)
{
  reconstruction.startThread(i);
}

Reconstruction::Reconstruction(std::vector<int> dev, bool multiThreadedGPU) : multiThreadedGPU(multiThreadedGPU)
{
  devicesToUse = dev;

  _useCPUReg = false;
  _debugGPU = false;

  for (int d = 0; d < devicesToUse.size(); d++)
  {
    int dev = devicesToUse[d];
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, dev);
    printf("Going to use dev %d: %s\n", dev, props.name);
    checkCudaErrors(cudaSetDevice(dev));
    checkCudaErrors(cudaDeviceReset());
    checkCudaErrors(cudaStreamCreate(&streams[dev]));

    int can_access_peer;
    cudaDeviceCanAccessPeer(&can_access_peer, 0, dev);
    printf("peer access GPU 0 to %d \n", can_access_peer);
    if (can_access_peer == 1)
    {
      printf("enabeling peer access GPU 0 to %d \n", dev);
      checkCudaErrors(cudaSetDevice(0));
      checkCudaErrors(cudaDeviceEnablePeerAccess(dev, 0));
    }
  }


  if (multiThreadedGPU)
  {
    for (int d = 0; d < devicesToUse.size(); ++d)
    {
      //GPU_workers.emplace_back(*this, GPU_sync, devicesToUse[d]);
#if !USE_BOOST
      GPU_workers.push_back(std::make_shared<GPUWorkerCommunicator>(GPUWorkerCommunicator(*this, GPU_sync, devicesToUse[d])));
#else
      GPU_workers.push_back(boost::make_shared<GPUWorkerCommunicator>(GPUWorkerCommunicator(*this, GPU_sync, devicesToUse[d])));
#endif
    }
    // start the threads
    GPU_sync.startup<Reconstruction>(devicesToUse.size(), threadStartCaller, *this);
  }

  checkCudaErrors(cudaSetDevice(0));

  int h_directions[][3] = {
      { 1, 0, -1 },
      { 0, 1, -1 },
      { 1, 1, -1 },
      { 1, -1, -1 },
      { 1, 0, 0 },
      { 0, 1, 0 },
      { 1, 1, 0 },
      { 1, -1, 0 },
      { 1, 0, 1 },
      { 0, 1, 1 },
      { 1, 1, 1 },
      { 1, -1, 1 },
      { 0, 0, 1 }
  };
  // this is constant -> constant mem for regularization
  float factor[13];
  for (int i = 0; i < 13; i++) {
    factor[i] = 0;
  }
  for (int i = 0; i < 13; i++) {
    for (int j = 0; j < 3; j++)
    {
      factor[i] += fabs((float)(h_directions[i][j]));
    }
    factor[i] = 1.0f / factor[i];
  }

  checkCudaErrors(cudaMemcpyToSymbol(d_factor, (void*)factor, 13 * sizeof(float)));
  checkCudaErrors(cudaMemcpyToSymbol(d_directions, (void*)h_directions, 3 * 13 * sizeof(int)));

  d_scale_weight_allocated = false;
  d_sliceMatrices_allocated = false;
  d_slice_sized_allocated = false;
  d_sliceDims_allocated = false;
  d_sliceResMatrices_allocated = false;
  dev_d_slicesOfs_allocated = false;
  reconstructed_arrays_init = false;
  regStorageInit = false;

}

void Reconstruction::startThread(int i)
{
#if !USE_BOOST
  GPU_threads.push_back(std::make_shared<std::thread>(&GPUWorkerCommunicator::execute, std::ref(GPU_workers[i])));
#else
  GPU_threads.push_back(boost::make_shared<boost::thread>(&GPUWorkerCommunicator::execute, boost::ref(GPU_workers[i])));
#endif
}


void Reconstruction::generatePSFVolume(float* CPUPSF, uint3 PSFsize_, float3 sliceVoxelDim,
  float3 PSFdim, Matrix4 PSFI2W, Matrix4 PSFW2I, float _quality_factor, bool _use_SINC)
{
  std::cout << "generating PSF Volume..." << std::endl;
  PSFsize = PSFsize_;
  h_quality_factor = _quality_factor;

  if (multiThreadedGPU)
  {
    for (int d = 0; d < devicesToUse.size(); d++)
      GPU_workers[d]->prepareGeneratePSFVolume(CPUPSF, PSFsize_, sliceVoxelDim, PSFdim, PSFI2W, PSFW2I, _quality_factor, _use_SINC);
    GPU_sync.runNextRound();
  }
  else
  {
    for (int d = 0; d < devicesToUse.size(); d++)
    {
      int dev = devicesToUse[d];
      generatePSFVolumeOnX(CPUPSF, PSFsize_, sliceVoxelDim, PSFdim, PSFI2W, PSFW2I, _quality_factor, _use_SINC, dev);
    }
  }
}

void Reconstruction::generatePSFVolumeOnX(float* CPUPSF, uint3 PSFsize_, float3 sliceVoxelDim,
  float3 PSFdim, Matrix4 PSFI2W, Matrix4 PSFW2I, float _quality_factor, bool _use_SINC, int dev)
{
  checkCudaErrors(cudaSetDevice(dev));
  checkCudaErrors(cudaMemcpyToSymbol(d_PSFdim, &(PSFdim), sizeof(float3)));
  checkCudaErrors(cudaMemcpyToSymbol(d_PSFsize, &(PSFsize_), sizeof(uint3)));
  checkCudaErrors(cudaMemcpyToSymbol(d_PSFI2W, &(PSFI2W), sizeof(Matrix4)));
  checkCudaErrors(cudaMemcpyToSymbol(d_PSFW2I, &(PSFW2I), sizeof(Matrix4)));
  checkCudaErrors(cudaMemcpyToSymbol(d_quality_factor, &(_quality_factor), sizeof(float)));
  checkCudaErrors(cudaMemcpyToSymbol(d_use_SINC, &(_use_SINC), sizeof(bool)));
  if (_use_SINC && _debugGPU)
  {
    printf("using sinc like PSF.");
  }
}

template <typename T>
__global__ void initVolume(Volume<T> volume, const T val){
  uint2 a = thr2pos2();
  uint3 pos = make_uint3(a.x, a.y, 0);
  for (pos.z = 0; pos.z < volume.size.z; ++pos.z)
  {
    if (pos.x < volume.size.x && pos.y < volume.size.y && pos.z < volume.size.z &&
      pos.x >= 0 && pos.y >= 0 && pos.z >= 0)
      volume.set(pos, val);
  }
}

template <class T>
void initMem(T* mem, unsigned int N, T val)
{
  thrust::device_ptr<T> dev_p(mem);
  thrust::fill(dev_p, dev_p + N, val);
}


void Reconstruction::setSliceDims(std::vector<float3> slice_dims, float _quality_factor)
{

  std::vector<int> _sliceDim;
  for (int i = 0; i < slice_dims.size(); ++i)
  {
    float size = dev_reconstructed_[0].dim.x / _quality_factor;
    int xDim = 2.0f * slice_dims[i].x / size + 0.5f;
    int yDim = 2.0f * slice_dims[i].y / size + 0.5f;
    int zDim = 2.0f * slice_dims[i].z / size + 0.5f;
    int dim = (floor(ceil(sqrt(float(xDim * xDim + yDim * yDim + zDim * zDim)) / _quality_factor) / 2)) * 2 + 1 + 2;
    _sliceDim.push_back(dim);
  }

  bool allocate = !d_sliceDims_allocated;
  if (allocate)
  {
    int maxDevId = *std::max_element(devicesToUse.begin(), devicesToUse.end());
    dev_d_sliceDims.resize(maxDevId + 1);
    dev_d_sliceDim.resize(maxDevId + 1);
    d_sliceDims_allocated = true;
  }


  if (multiThreadedGPU)
  {
    for (int d = 0; d < devicesToUse.size(); d++)
      GPU_workers[d]->prepareSetSliceDims(slice_dims, _sliceDim, allocate);
    GPU_sync.runNextRound();
  }
  else
  {
    for (int d = 0; d < devicesToUse.size(); d++)
    {
      int dev = devicesToUse[d];
      setSliceDimsOnX(slice_dims, _sliceDim, allocate, dev);
    }
  }


  checkCudaErrors(cudaSetDevice(0));
}

void Reconstruction::setSliceDimsOnX(std::vector<float3>& slice_dims, std::vector<int>& sliceDim, bool allocate, int dev)
{
  checkCudaErrors(cudaSetDevice(dev));

  unsigned int start = dev_slice_range_offset[dev].x;
  unsigned int end = dev_slice_range_offset[dev].y;

  if (allocate)
  {
    float3* t1;
    checkCudaErrors(cudaMalloc((void**)&t1, dev_v_slices[dev].size.z*sizeof(float3)));
    dev_d_sliceDims[dev] = t1;
    int* t2;
    checkCudaErrors(cudaMalloc((void**)&t2, dev_v_slices[dev].size.z*sizeof(int)));
    dev_d_sliceDim[dev] = t2;
  }
  checkCudaErrors(cudaMemcpy(dev_d_sliceDims[dev], &slice_dims[start], dev_v_slices[dev].size.z*sizeof(float3), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dev_d_sliceDim[dev], &sliceDim[start], dev_v_slices[dev].size.z*sizeof(int), cudaMemcpyHostToDevice));
}

void Reconstruction::SetSliceMatrices(std::vector<Matrix4> matSliceTransforms, std::vector<Matrix4> matInvSliceTransforms,
  std::vector<Matrix4>& matsI2Winit, std::vector<Matrix4>& matsW2Iinit, std::vector<Matrix4>& matsI2W,
  std::vector<Matrix4>& matsW2I, Matrix4 reconI2W, Matrix4 reconW2I)
{
  bool alloc = !d_sliceMatrices_allocated;
  if (alloc)
  {
    int storagesize = 1 + *std::max_element(devicesToUse.begin(), devicesToUse.end());
    dev_d_slicesI2W.resize(storagesize);
    dev_d_slicesW2I.resize(storagesize);
    dev_d_slicesInvTransformation.resize(storagesize);
    dev_d_slicesTransformation.resize(storagesize);

    d_sliceMatrices_allocated = true;
  }

  if (multiThreadedGPU)
  {
    for (int d = 0; d < devicesToUse.size(); d++)
      GPU_workers[d]->prepareSetSliceMatrices(matSliceTransforms, matInvSliceTransforms, matsI2Winit, matsW2Iinit, matsI2W, matsW2I, reconI2W, reconW2I, alloc);
    GPU_sync.runNextRound();
  }
  else
  {
    for (int d = 0; d < devicesToUse.size(); d++)
    {
      int dev = devicesToUse[d];
      SetSliceMatricesOnX(matSliceTransforms, matInvSliceTransforms, matsI2Winit, matsW2Iinit, matsI2W, matsW2I, reconI2W, reconW2I, dev, alloc);
    }
  }


  checkCudaErrors(cudaSetDevice(0));
}

void Reconstruction::SetSliceMatricesOnX(std::vector<Matrix4> matSliceTransforms, std::vector<Matrix4> matInvSliceTransforms,
  std::vector<Matrix4>& matsI2Winit, std::vector<Matrix4>& matsW2Iinit, std::vector<Matrix4>& matsI2W,
  std::vector<Matrix4>& matsW2I, Matrix4 reconI2W, Matrix4 reconW2I, int dev, bool alloc)
{
  checkCudaErrors(cudaSetDevice(dev));

  unsigned int start = dev_slice_range_offset[dev].x;
  unsigned int end = dev_slice_range_offset[dev].y;

  // printf("SetSliceMatrices device %d from %d to %d size %d\n", dev, start, end, end-start);

  checkCudaErrors(cudaMemcpyToSymbol(d_reconstructedW2I, &(reconW2I), sizeof(Matrix4)));
  checkCudaErrors(cudaMemcpyToSymbol(d_reconstructedI2W, &(reconI2W), sizeof(Matrix4)));
  CHECK_ERROR(cudaMemcpyToSymbol);

  if (alloc)
  {
    Matrix4* t1;
    checkCudaErrors(cudaMalloc((void**)&t1, (end - start)*sizeof(Matrix4)));
    dev_d_slicesI2W[dev] = (t1);

    Matrix4* t2;
    checkCudaErrors(cudaMalloc((void**)&t2, (end - start)*sizeof(Matrix4)));
    dev_d_slicesW2I[dev] = (t2);

    Matrix4* t3;
    checkCudaErrors(cudaMalloc((void**)&t3, (end - start)*sizeof(Matrix4)));
    dev_d_slicesInvTransformation[dev] = (t3);

    Matrix4* t4;
    checkCudaErrors(cudaMalloc((void**)&t4, (end - start)*sizeof(Matrix4)));
    dev_d_slicesTransformation[dev] = (t4);
  }
  checkCudaErrors(cudaMemcpy(dev_d_slicesI2W[dev], &matsI2W[start], (end - start)*sizeof(Matrix4), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dev_d_slicesW2I[dev], &matsW2I[start], (end - start)*sizeof(Matrix4), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dev_d_slicesInvTransformation[dev], &matInvSliceTransforms[start], (end - start)*sizeof(Matrix4), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dev_d_slicesTransformation[dev], &matSliceTransforms[start], (end - start)*sizeof(Matrix4), cudaMemcpyHostToDevice));
}

template <typename T>
__global__ void GaussianConvolutionKernel(int numSlices, int* d_sliceSicesX, int* d_sliceSicesY,
  T* input, Volume<T> output, float3* __restrict d_slicedim, uint2 vSize, T sigma, bool horizontal, int cofs)
{
  const uint3 pos = make_uint3(__umul24(blockIdx.x, blockDim.x) + threadIdx.x,
    __umul24(blockIdx.y, blockDim.y) + threadIdx.y,
    __umul24(blockIdx.z, blockDim.z) + threadIdx.z + cofs);

  if (pos.z >= numSlices || pos.z < 0)
    return;

  int ssizeX = vSize.x;
  int ssizeY = vSize.y;
  if (pos.x >= ssizeX || pos.y >= ssizeY || pos.x < 0 || pos.y < 0)
    return;

  float3 slicedim = d_slicedim[pos.z];
  T sigma2 = sigma / (T)slicedim.x;
  int klength = 2 * round_(4 * sigma2) + 1;
  klength -= 1 - klength % 2;
  unsigned int idx = pos.x + __umul24(pos.y, output.size.x) + __umul24(pos.z, __umul24(output.size.x, output.size.y));
  //////////////////
  //shared memory would be good, but the kernel size is really large... (memory problem)

  T sum = 0;
  int half_kernel_elements = (klength - 1) / 2;
  T outv = 0.0;

  if (horizontal) {
    // convolve horizontally
    T g0 = 1.0 / (sqrt(2.0 * M_PI) * sigma2); //norm
    T g1 = exp(-0.5 / (sigma2 * sigma2));
    T g2 = g1 * g1;
    sum = g0 * input[idx];
    T sum_coeff = g0;
    for (unsigned int i = 1; i <= half_kernel_elements; i++){
      g0 *= g1;
      g1 *= g2;
      //border repeat
      unsigned int src_x = reflect(ssizeX, (int)pos.x + (int)i);
      unsigned int idx2 = src_x + __umul24(pos.y, output.size.x) + __umul24(pos.z, __umul24(output.size.x, output.size.y));
      sum += g0 * input[idx2];
      src_x = reflect(ssizeX, (int)pos.x - (int)i);
      idx2 = src_x + __umul24(pos.y, output.size.x) + __umul24(pos.z, __umul24(output.size.x, output.size.y));
      sum += g0 * input[idx2];
      sum_coeff += 2 * g0;
    }
    outv = sum / sum_coeff;
  }
  else {
    // convolve vertically
    T g0 = 1.0 / (sqrt(2.0 * M_PI) * sigma2);
    T g1 = exp(-0.5 / (sigma2 * sigma2));
    T g2 = g1 * g1;
    sum = g0 * input[idx];
    T sum_coeff = g0;
    for (unsigned int j = 1; j <= half_kernel_elements; j++){
      g0 *= g1;
      g1 *= g2;
      //border repeat
      unsigned int src_y = reflect(ssizeY, (int)pos.y + (int)j);
      unsigned int idx2 = pos.x + __umul24(src_y, output.size.x) + __umul24(pos.z, __umul24(output.size.x, output.size.y));
      sum += g0 * input[idx2];
      src_y = reflect(ssizeY, (int)pos.y - (int)j);
      idx2 = pos.x + __umul24(src_y, output.size.x) + __umul24(pos.z, __umul24(output.size.x, output.size.y));
      sum += g0 * input[idx2];
      sum_coeff += 2 * g0;
    }
    outv = sum / sum_coeff;
  }

  if (outv != 0)
  {
    output.set(pos, outv);
  }

}


__global__ void GaussianConvolutionKernel3D(float* input, float* output, float sigma, short dir, float3 inputDim, uint3 inputSize)
{
  const uint3 pos = make_uint3(__umul24(blockIdx.x, blockDim.x) + threadIdx.x,
    __umul24(blockIdx.y, blockDim.y) + threadIdx.y,
    __umul24(blockIdx.z, blockDim.z) + threadIdx.z);

  if (pos.z >= inputSize.z || pos.z < 0)
    return;

  int ssizeX = inputSize.x;
  int ssizeY = inputSize.y;
  int ssizeZ = inputSize.z;
  if (pos.x >= ssizeX || pos.y >= ssizeY || pos.x < 0 || pos.y < 0)
    return;

  unsigned int idx = pos.x + __umul24(pos.y, inputSize.x) + __umul24(pos.z, __umul24(inputSize.x, inputSize.y));
  //////////////////

  //shared memory would be good, but the kernel size is really large... (memory problem)
  float sum = 0;
  float outv = 0.0;

  if (dir == 0) {
    float sigma2 = sigma / (float)inputDim.x;
    int klength = 2 * round_(4 * sigma2) + 1;
    klength -= 1 - klength % 2;
    int half_kernel_elements = (klength - 1) / 2;
    // convolve horizontally
    float g0 = 1.0 / (sqrt(2.0 * M_PI) * sigma2); //norm
    float g1 = exp(-0.5 / (sigma2 * sigma2));
    float g2 = g1 * g1;
    sum = g0 * input[idx];
    float sum_coeff = g0;
    for (unsigned int i = 1; i <= half_kernel_elements; i++){
      g0 *= g1;
      g1 *= g2;
      //border repeat
      unsigned int src_x = reflect(ssizeX, (int)pos.x + (int)i);
      unsigned int idx2 = src_x + __umul24(pos.y, inputSize.x) + __umul24(pos.z, __umul24(inputSize.x, inputSize.y));
      sum += g0 * input[idx2];
      src_x = reflect(ssizeX, (int)pos.x - (int)i);
      idx2 = src_x + __umul24(pos.y, inputSize.x) + __umul24(pos.z, __umul24(inputSize.x, inputSize.y));
      sum += g0 * input[idx2];
      sum_coeff += 2 * g0;
    }
    outv = sum / sum_coeff;
  }
  else if (dir == 1){
    float sigma2 = sigma / (float)inputDim.y;
    int klength = 2 * round_(4 * sigma2) + 1;
    klength -= 1 - klength % 2;
    int half_kernel_elements = (klength - 1) / 2;
    // convolve vertically
    float g0 = 1.0 / (sqrt(2.0 * M_PI) * sigma2);
    float g1 = exp(-0.5 / (sigma2 * sigma2));
    float g2 = g1 * g1;
    sum = g0 * input[idx];
    float sum_coeff = g0;
    //#pragma unroll 10
    for (unsigned int j = 1; j <= half_kernel_elements; j++){
      g0 *= g1;
      g1 *= g2;
      //border repeat
      unsigned int src_y = reflect(ssizeY, (int)pos.y + (int)j);
      unsigned int idx2 = pos.x + __umul24(src_y, inputSize.x) + __umul24(pos.z, __umul24(inputSize.x, inputSize.y));
      sum += g0 * input[idx2];
      src_y = reflect(ssizeY, (int)pos.y - (int)j);
      idx2 = pos.x + __umul24(src_y, inputSize.x) + __umul24(pos.z, __umul24(inputSize.x, inputSize.y));
      sum += g0 * input[idx2];
      sum_coeff += 2 * g0;
    }
    outv = sum / sum_coeff;
  }
  else if (dir == 2){
    float sigma2 = sigma / (float)inputDim.z;
    int klength = 2 * round_(4 * sigma2) + 1;
    klength -= 1 - klength % 2;
    int half_kernel_elements = (klength - 1) / 2;
    // convolve vertically
    float g0 = 1.0 / (sqrt(2.0 * M_PI) * sigma2);
    float g1 = exp(-0.5 / (sigma2 * sigma2));
    float g2 = g1 * g1;
    sum = g0 * input[idx];
    float sum_coeff = g0;
    //#pragma unroll 10
    for (unsigned int k = 1; k <= half_kernel_elements; k++){
      g0 *= g1;
      g1 *= g2;
      //border repeat
      unsigned int src_z = reflect(ssizeZ, (int)pos.z + (int)k);
      unsigned int idx2 = pos.x + __umul24(pos.y, inputSize.x) + __umul24(src_z, __umul24(inputSize.x, inputSize.y));
      sum += g0 * input[idx2];
      src_z = reflect(ssizeZ, (int)pos.z - (int)k);
      idx2 = pos.x + __umul24(pos.y, inputSize.x) + __umul24(src_z, __umul24(inputSize.x, inputSize.y));
      sum += g0 * input[idx2];
      sum_coeff += 2 * g0;
    }
    outv = sum / sum_coeff;
  }

  if (outv == outv)
  {
    output[idx] = outv;
  }

}

void Reconstruction::setMask(uint3 s, float3 dim, float* data, float sigma_bias)
{
  int storagesize = 1 + *std::max_element(devicesToUse.begin(), devicesToUse.end());
  dev_mask_.resize(storagesize);
  if (multiThreadedGPU)
  {
    for (int d = 0; d < devicesToUse.size(); d++)
      GPU_workers[d]->prepareSetMask(s, dim, data, sigma_bias);
    GPU_sync.runNextRound();
  }
  else
  {
    for (int d = 0; d < devicesToUse.size(); d++)
    {
      int dev = devicesToUse[d];
      setMaskOnX(s, dim, data, sigma_bias, dev);
    }
  }
  checkCudaErrors(cudaSetDevice(0));
}


void Reconstruction::setMaskOnX(uint3 s, float3 dim, float* data, float sigma_bias, int dev)
{
  printf("setMask %d\n", dev);
  checkCudaErrors(cudaSetDevice(dev));

  //printf("l_mask %d\n",dev);
  Volume<float> l_mask;;
  l_mask.init(s, dim);
  checkCudaErrors(cudaMemcpy(l_mask.data, data, s.x*s.y*s.z*sizeof(float), cudaMemcpyHostToDevice));
  dev_mask_[dev] = l_mask;

  // FIXMEEEEEE: what if device 0 should not be used/is not the "primary device"
  if (dev == 0)
  {
    maskC_.init(s, dim);
    checkCudaErrors(cudaMemcpy(maskC_.data, data, s.x*s.y*s.z*sizeof(float), cudaMemcpyHostToDevice));

    //printf("filter Mask %d \n", dev);
    dim3 blockSize3 = dim3(4, 4, 4);
    dim3 gridSize3 = divup(dim3(maskC_.size.x, maskC_.size.y, maskC_.size.z), blockSize3);

    Volume<float> mbuf;
    mbuf.init(maskC_.size, maskC_.dim);

    initVolume<float> << <gridSize3, blockSize3, 0, streams[dev] >> >(mbuf, 0.0);
    CHECK_ERROR(initVolume);

    GaussianConvolutionKernel3D << <gridSize3, blockSize3, 0, streams[dev] >> >(maskC_.data, mbuf.data, sigma_bias, 0,
      maskC_.dim, maskC_.size);
    CHECK_ERROR(GaussianConvolutionKernel3D);
    GaussianConvolutionKernel3D << <gridSize3, blockSize3, 0, streams[dev] >> >(mbuf.data, maskC_.data, sigma_bias, 1,
      maskC_.dim, maskC_.size);
    CHECK_ERROR(GaussianConvolutionKernel3D);
    GaussianConvolutionKernel3D << <gridSize3, blockSize3, 0, streams[dev] >> >(maskC_.data, mbuf.data, sigma_bias, 2,
      maskC_.dim, maskC_.size);
    CHECK_ERROR(GaussianConvolutionKernel3D);
    checkCudaErrors(cudaMemcpy(maskC_.data, mbuf.data,
      maskC_.size.x*maskC_.size.y*maskC_.size.z*sizeof(float), cudaMemcpyDeviceToDevice));
    cudaDeviceSynchronize();
    mbuf.release();
  }
}

void Reconstruction::InitReconstructionVolume(uint3 s, float3 dim, float *data, float sigma_bias)
{
  printf("InitReconstructionVolume\n");
  int storagesize = 1 + *std::max_element(devicesToUse.begin(), devicesToUse.end());

  dev_reconstructed_.resize(storagesize);
  dev_bias_.resize(storagesize);
  dev_reconstructed_volWeigths.resize(storagesize);
  dev_addon_.resize(storagesize);
  dev_confidence_map_.resize(storagesize);
  dev_volume_weights_.resize(storagesize);

  if (multiThreadedGPU)
  {
    for (int d = 0; d < devicesToUse.size(); d++)
      GPU_workers[d]->prepareInitReconstructionVolume(s, dim, data, sigma_bias);
    GPU_sync.runNextRound();
  }
  else
  {
    for (int d = 0; d < devicesToUse.size(); d++)
    {
      int dev = devicesToUse[d];
      InitReconstructionVolumeOnX(s, dim, data, sigma_bias, dev);
    }
  }
  checkCudaErrors(cudaSetDevice(0));
}

void Reconstruction::InitReconstructionVolumeOnX(uint3 s, float3 dim, float *data, float sigma_bias, int dev)
{
  checkCudaErrors(cudaSetDevice(dev));

  Volume<float> l_reconstructed_;
  l_reconstructed_.init(s, dim);
  unsigned int N = l_reconstructed_.size.x*l_reconstructed_.size.y*l_reconstructed_.size.z;
  cudaMemset(l_reconstructed_.data, 0, N*sizeof(float));
  if (data != NULL) checkCudaErrors(cudaMemcpy(l_reconstructed_.data, data, s.x*s.y*s.z*sizeof(float), cudaMemcpyHostToDevice));
  dev_reconstructed_[dev] = (l_reconstructed_);

  Volume<float> l_bias_;
  l_bias_.init(s, dim);
  cudaMemset(l_bias_.data, 0, N*sizeof(float));
  dev_bias_[dev] = (l_bias_);

  Volume<float> l_volume_weights_;
  l_volume_weights_.init(s, dim);
  cudaMemset(l_volume_weights_.data, 0, N*sizeof(float));
  dev_volume_weights_[dev] = (l_volume_weights_);

  Volume<float> l_reconstructed_volWeigths;
  l_reconstructed_volWeigths.init(s, dim);
  cudaMemset(l_reconstructed_volWeigths.data, 0, N*sizeof(float));
  dev_reconstructed_volWeigths[dev] = (l_reconstructed_volWeigths);

  Volume<float> l_addon_;
  l_addon_.init(s, dim);
  cudaMemset(l_addon_.data, 0, N*sizeof(float));
  dev_addon_[dev] = (l_addon_);

  Volume<float> l_confidence_map_;
  l_confidence_map_.init(s, dim);
  cudaMemset(l_confidence_map_.data, 0, N*sizeof(float));
  dev_confidence_map_[dev] = (l_confidence_map_);
}

Reconstruction::~Reconstruction()
{

  //temporary
  v_weights.release();

  if (_debugGPU)
  {
    ////////single GPU
    //debug
    sliceVoxel_count_.release();
    v_bias.release();
    v_simulated_weights.release();
    v_simulated_slices.release();
    v_wresidual.release();
    v_wb.release();
    v_buffer.release();
    v_PSF_sums_.release();
  }


  if (multiThreadedGPU)
  {
    for (int d = 0; d < devicesToUse.size(); d++)
      GPU_workers[d]->end();
    GPU_sync.runNoSync();
    GPU_threads.clear();
  }
  v_slices.release();

  if (dev_reconstructed_.size() == 0)
    return;

  for (int d = 0; d < devicesToUse.size(); d++)
  {
    int dev = devicesToUse[d];
    cleanUpOnX(dev);

    checkCudaErrors(cudaStreamDestroy(streams[dev]));
  }

}

void Reconstruction::cleanUpOnX(int dev)
{
  checkCudaErrors(cudaSetDevice(dev));

  dev_reconstructed_[dev].release();
  dev_bias_[dev].release();
  dev_reconstructed_volWeigths[dev].release();
  dev_addon_[dev].release();
  dev_confidence_map_[dev].release();
  dev_volume_weights_[dev].release();

  dev_sliceVoxel_count_[dev].release();
  dev_v_slices[dev].release();
  dev_v_bias[dev].release();
  dev_v_weights[dev].release();
  dev_v_simulated_weights[dev].release();
  dev_v_simulated_slices[dev].release();
  dev_v_wresidual[dev].release();
  dev_v_wb[dev].release();
  dev_v_buffer[dev].release();
  dev_v_PSF_sums_[dev].release();

  if (!_useCPUReg)
  {
    dev_temp_slices[dev].release();
    dev_v_slices_resampled[dev].release();
    dev_v_slices_resampled_float[dev].release();
    dev_regSlices[dev].release();
  }
};

void Reconstruction::UpdateSliceWeights(std::vector<float> slices_weights)
{
  h_slices_weights = slices_weights;

  if (multiThreadedGPU)
  {
    for (int d = 0; d < devicesToUse.size(); d++)
      GPU_workers[d]->prepareUpdateSliceWeights(slices_weights);
    GPU_sync.runNextRound();
  }
  else
  {
    for (int d = 0; d < devicesToUse.size(); d++)
    {
      int dev = devicesToUse[d];
      UpdateSliceWeightsOnX(slices_weights, dev);
    }
  }

}

void Reconstruction::UpdateSliceWeightsOnX(std::vector<float>& slices_weights, int dev)
{
  checkCudaErrors(cudaSetDevice(dev));
  unsigned int start = dev_slice_range_offset[dev].x;
  unsigned int end = dev_slice_range_offset[dev].y;

  checkCudaErrors(cudaMemcpy(dev_d_slice_weights[dev], &slices_weights[start], dev_v_slices[dev].size.z*sizeof(float), cudaMemcpyHostToDevice));
}

void Reconstruction::UpdateScaleVector(std::vector<float> scales, std::vector<float> slices_weights)
{

  h_scales = scales;
  h_slices_weights = slices_weights;

  bool alloc = !d_scale_weight_allocated;
  if (alloc)
  {
    int storagesize = 1 + *std::max_element(devicesToUse.begin(), devicesToUse.end());
    dev_d_scales.resize(storagesize);
    dev_d_slice_weights.resize(storagesize);
    d_scale_weight_allocated = true;
  }

  if (multiThreadedGPU)
  {
    for (int d = 0; d < devicesToUse.size(); d++)
      GPU_workers[d]->prepareUpdateScaleVector(scales, slices_weights, alloc);
    GPU_sync.runNextRound();
  }
  else
  {
    for (int d = 0; d < devicesToUse.size(); d++)
    {
      int dev = devicesToUse[d];
      UpdateScaleVectorOnX(scales, slices_weights, dev, alloc);
    }
  }


}

void Reconstruction::UpdateScaleVectorOnX(std::vector<float>& scales, std::vector<float>& slices_weights, int dev, bool alloc)
{
  unsigned int start = dev_slice_range_offset[dev].x;
  unsigned int end = dev_slice_range_offset[dev].y;

  checkCudaErrors(cudaSetDevice(dev));
  if (alloc)
  {
    float* t1;
    checkCudaErrors(cudaMalloc((void **)&t1, dev_v_slices[dev].size.z*sizeof(float)));
    std::cout << dev << " " << dev_v_slices[dev].size.z << " " << h_scales.size() << std::endl;
    dev_d_scales[dev] = (t1);
    float* t2;
    checkCudaErrors(cudaMalloc((void **)&t2, dev_v_slices[dev].size.z*sizeof(float)));
    dev_d_slice_weights[dev] = (t2);
  }
  checkCudaErrors(cudaMemcpy(dev_d_scales[dev], &h_scales[start], dev_v_slices[dev].size.z*sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dev_d_slice_weights[dev], &h_slices_weights[start], dev_v_slices[dev].size.z*sizeof(float), cudaMemcpyHostToDevice));
}

__global__ void castToFloat(Volume<float> in, Volume<float> out)
{
  const uint3 pos = make_uint3(__umul24(blockIdx.x, blockDim.x) + threadIdx.x,
    __umul24(blockIdx.y, blockDim.y) + threadIdx.y,
    __umul24(blockIdx.z, blockDim.z) + threadIdx.z);

  if (pos.x < in.size.x && pos.y < in.size.y && pos.z < in.size.z &&
    pos.x >= 0 && pos.y >= 0 && pos.z >= 0)
  {
    out.set(pos, (float)(in[pos]));
  }
}

void Reconstruction::initStorageVolumes(uint3 size, float3 dim)
{
  printf("\nINIT STORAGE\n");
  num_slices_ = size.z;

  unsigned int start = 0;
  unsigned int end = 0;
  unsigned int step = floor(num_slices_ / (float)devicesToUse.size());

  int storagesize = 1 + *std::max_element(devicesToUse.begin(), devicesToUse.end());

  slices_per_device.resize(storagesize);
  dev_v_slices.resize(storagesize);
  dev_v_bias.resize(storagesize);
  dev_v_weights.resize(storagesize);
  dev_v_simulated_weights.resize(storagesize);
  dev_v_simulated_slices.resize(storagesize);
  dev_v_wresidual.resize(storagesize);
  dev_v_wb.resize(storagesize);
  dev_v_buffer.resize(storagesize);
  dev_v_simulated_inside.resize(storagesize);
  dev_sliceVoxel_count_.resize(storagesize);
  dev_v_PSF_sums_.resize(storagesize);
  dev_slice_range_offset.resize(storagesize);

  if (multiThreadedGPU)
  {
    for (int d = 0; d < devicesToUse.size(); d++)
    {
      end = start + step;
      if (end >= size.z) end = size.z - 1;
      GPU_workers[d]->prepareInitStorageVolumes(size, dim, start, end);
      start = end;
    }
    GPU_sync.runNextRound();
  }
  else
  {
    for (int d = 0; d < devicesToUse.size(); d++)
    {
      int dev = devicesToUse[d];
      end = start + step;
      if (end >= size.z) end = size.z - 1;

      initStorageVolumesOnX(size, dim, start, end, dev);
      start = end;
    }
  }

  checkCudaErrors(cudaSetDevice(0));

  v_slices.init(size, dim);

  //temporary
  v_weights.init(size, dim);

  if (_debugGPU)
  {
    ////////////////////////////////// single GPU
    //for debug
    v_bias.init(size, dim);
    // v_weights.init(size, dim);
    v_simulated_weights.init(size, dim);
    v_simulated_slices.init(size, dim);
    v_wresidual.init(size, dim);
    v_wb.init(size, dim);
    v_buffer.init(size, dim);
    sliceVoxel_count_.init(size, dim);
    v_simulated_inside.init(size, dim);
    v_PSF_sums_.init(size, dim);
    unsigned int N = size.x*size.y*size.z;

    cudaMemset(v_slices.data, 0, N*sizeof(float));
    cudaMemset(v_bias.data, 0, N*sizeof(float));
    cudaMemset(v_weights.data, 0, N*sizeof(float));
    cudaMemset(v_simulated_weights.data, 0, N*sizeof(float));
    cudaMemset(v_simulated_slices.data, 0, N*sizeof(float));
    cudaMemset(v_wresidual.data, 0, N*sizeof(float));
    cudaMemset(v_wb.data, 0, N*sizeof(float));
    cudaMemset(v_buffer.data, 0, N*sizeof(float));
    cudaMemset(v_simulated_inside.data, 0, N*sizeof(char));
    cudaMemset(v_PSF_sums_.data, 0, N*sizeof(float));
    cudaMemset(sliceVoxel_count_.data, 0, N*sizeof(int));
  }
}


void Reconstruction::initStorageVolumesOnX(uint3 size, float3 dim, int start, int end, int dev)
{
  checkCudaErrors(cudaSetDevice(dev));
  printf("INIT device: %d start: %d end: %d size: %d \n", dev, start, end, end - start);

  uint3 dsize = size;
  dsize.z = end - start;
  slices_per_device[dev] = dsize.z;

  Volume<float> l_slices;
  l_slices.init(dsize, dim);
  dev_v_slices[dev] = (l_slices);

  Volume<float> l_bias;
  l_bias.init(dsize, dim);
  dev_v_bias[dev] = (l_bias);

  Volume<float> l_weights;
  l_weights.init(dsize, dim);
  dev_v_weights[dev] = (l_weights);

  Volume<float> l_simulated_weights;
  l_simulated_weights.init(dsize, dim);
  dev_v_simulated_weights[dev] = (l_simulated_weights);

  Volume<float> l_simulated_slices;
  l_simulated_slices.init(dsize, dim);
  dev_v_simulated_slices[dev] = (l_simulated_slices);

  Volume<float> l_wresidual;
  l_wresidual.init(dsize, dim);
  dev_v_wresidual[dev] = (l_wresidual);

  Volume<float> l_wb;
  l_wb.init(dsize, dim);
  dev_v_wb[dev] = (l_wb);

  Volume<float> l_buffer;
  l_buffer.init(dsize, dim);
  dev_v_buffer[dev] = (l_buffer);

  Volume<char> l_simulated_inside;
  l_simulated_inside.init(dsize, dim);
  dev_v_simulated_inside[dev] = (l_simulated_inside);

  Volume<int> l_sliceVoxel_count_;
  l_sliceVoxel_count_.init(dsize, dim);
  dev_sliceVoxel_count_[dev] = (l_sliceVoxel_count_);

  Volume<float> l_PSF_sums_;
  l_PSF_sums_.init(dsize, dim);
  dev_v_PSF_sums_[dev] = (l_PSF_sums_);

  unsigned int N = dsize.x*dsize.y*dsize.z;

  cudaMemset(dev_v_slices[dev].data, 0, N*sizeof(float));
  cudaMemset(dev_v_bias[dev].data, 0, N*sizeof(float));
  cudaMemset(dev_v_weights[dev].data, 0, N*sizeof(float));
  cudaMemset(dev_v_simulated_weights[dev].data, 0, N*sizeof(float));
  cudaMemset(dev_v_simulated_slices[dev].data, 0, N*sizeof(float));
  cudaMemset(dev_v_wresidual[dev].data, 0, N*sizeof(float));
  cudaMemset(dev_v_wb[dev].data, 0, N*sizeof(float));
  cudaMemset(dev_v_buffer[dev].data, 0, N*sizeof(float));
  cudaMemset(dev_v_simulated_inside[dev].data, 0, N*sizeof(char));
  cudaMemset(dev_v_PSF_sums_[dev].data, 0, N*sizeof(float));
  cudaMemset(dev_sliceVoxel_count_[dev].data, 0, N*sizeof(int));

  dev_slice_range_offset[dev] = (make_uint2(start, end));

  checkGPUMemory();
}
void Reconstruction::FillSlices(float* sdata, std::vector<int> sizesX, std::vector<int> sizesY)
{

  float* ldata = sdata;
  float* d_ldata = v_slices.data;

  bool alloc = !d_slice_sized_allocated;
  if (alloc)
  {
    int storagesize = 1 + *std::max_element(devicesToUse.begin(), devicesToUse.end());
    dev_d_slice_sicesX.resize(storagesize);
    dev_d_slice_sicesY.resize(storagesize);
    d_slice_sized_allocated = true;
  }

  if (multiThreadedGPU)
  {
    for (int d = 0; d < devicesToUse.size(); d++)
    {
      int dev = devicesToUse[d];
      unsigned int start = dev_slice_range_offset[dev].x;
      unsigned int end = dev_slice_range_offset[dev].y;
      unsigned int num_elems_ = (end - start)*dev_v_slices[dev].size.x*dev_v_slices[dev].size.y;

      GPU_workers[d]->prepareFillSlices(sdata, sizesX, sizesY, d_ldata, ldata, alloc);

      d_ldata += num_elems_;
      ldata += num_elems_;

    }
    GPU_sync.runNextRound();
  }
  else
  {
    for (int d = 0; d < devicesToUse.size(); d++)
    {
      int dev = devicesToUse[d];
      unsigned int start = dev_slice_range_offset[dev].x;
      unsigned int end = dev_slice_range_offset[dev].y;
      unsigned int num_elems_ = (end - start)*dev_v_slices[dev].size.x*dev_v_slices[dev].size.y;

      FillSlicesOnX(sdata, sizesX, sizesY, d_ldata, ldata, dev, alloc);

      d_ldata += num_elems_;
      ldata += num_elems_;
    }
  }


  checkCudaErrors(cudaSetDevice(0));
}

void Reconstruction::FillSlicesOnX(float* sdata, std::vector<int>& sizesX, std::vector<int>& sizesY, float* d_ldata, float* ldata, int dev, bool alloc)
{
  unsigned int start = dev_slice_range_offset[dev].x;
  unsigned int end = dev_slice_range_offset[dev].y;

  checkCudaErrors(cudaSetDevice(dev));

  printf("MEMCPY device: %d start: %d end: %d size: %d  nSlices: %d \n", dev, start, end, end - start, dev_v_slices.at(dev).size.z);
  checkGPUMemory();

  unsigned int num_elems_ = (end - start)*dev_v_slices[dev].size.x*dev_v_slices[dev].size.y;
  checkCudaErrors(cudaMemcpy(dev_v_slices[dev].data, ldata, num_elems_*sizeof(float),
    cudaMemcpyHostToDevice));

  CHECK_ERROR(FillSlices);

  if (alloc)
  {
    int* t1;
    int* t2;
    checkCudaErrors(cudaMalloc((void **)&t1, dev_v_slices[dev].size.z*sizeof(int)));
    dev_d_slice_sicesX[dev] = (t1);
    checkCudaErrors(cudaMalloc((void **)&t2, dev_v_slices[dev].size.z*sizeof(int)));
    dev_d_slice_sicesY[dev] = (t2);
  }
  checkCudaErrors(cudaMemcpy(dev_d_slice_sicesX[dev], &sizesX[start], (end - start)*sizeof(int), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dev_d_slice_sicesY[dev], &sizesY[start], (end - start)*sizeof(int), cudaMemcpyHostToDevice));

  //needed for stack-wise restore slice intensity
  checkCudaErrors(cudaMemcpy(d_ldata, dev_v_slices[dev].data, num_elems_*sizeof(float), cudaMemcpyDefault));
}

void Reconstruction::UpdateReconstructed(const uint3 vsize, float* data)
{
  cudaMemcpy(dev_reconstructed_[0].data, data, vsize.x*vsize.y*vsize.z*sizeof(float), cudaMemcpyHostToDevice);

  //update other GPUs
  for (int d = 1; d < devicesToUse.size(); d++)
  {
    int dev = devicesToUse[d];
    UpdateReconstructedOnX(vsize, data, dev);
  }
}
void Reconstruction::UpdateReconstructedOnX(const uint3 vsize, float* data, int dev)
{
  unsigned int N = dev_reconstructed_[0].size.x*dev_reconstructed_[0].size.y*dev_reconstructed_[0].size.z;
  checkCudaErrors(cudaSetDevice(dev));
  checkCudaErrors(cudaMemcpy(dev_reconstructed_[dev].data, dev_reconstructed_[0].data,
    N*sizeof(float), cudaMemcpyDefault));
}


struct not_equal_than
{
  float val;
  not_equal_than(float t) { val = t; }
  __host__ __device__
    bool operator()(float x) { return x != val; }
};


__global__ void calculateResidual3D_adv(int numSlices, float* slices, float* bias,
  float* weights, float* simweights,
  float* simslices, Volume<float> wb_, Volume<float> wr_, float* scales)
{
  const uint3 pos = make_uint3(__umul24(blockIdx.x, blockDim.x) + threadIdx.x,
    __umul24(blockIdx.y, blockDim.y) + threadIdx.y,
    __umul24(blockIdx.z, blockDim.z) + threadIdx.z);

  //z is slice index
  if (pos.z >= numSlices || pos.z < 0)
    return;

  unsigned int idx = pos.x + pos.y*wb_.size.x + pos.z*wb_.size.x*wb_.size.y;
  float s = slices[idx];
  if ((s == -1.0))
    return;

  float b = bias[idx];
  float w = weights[idx];
  float sw = simweights[idx];
  float ss = simslices[idx];
  float wb = wb_[pos];
  float wr = wr_[pos];
  float scale = scales[pos.z];
  float wbo = 0.0;
  float wro = 0.0;

  if (sw > 0.99)
  {
    float eb = exp(-b);
    float sliceVal = s*(eb * scale);
    wbo = w * sliceVal;

    if ((ss > 1.0) && (sliceVal > 1.0)) {
      wro = log(sliceVal / ss) * wbo;
    }

  }
  if (wbo > 0)
  {
    wb_.set(pos, wbo);
    wr_.set(pos, wro);
  }

}


__global__ void updateBiasField3D_adv(int numSlices,
  float* slices, Volume<float> bias_slices, float* wb_, float* wr_)
{
  const uint3 pos = make_uint3(__umul24(blockIdx.x, blockDim.x) + threadIdx.x,
    __umul24(blockIdx.y, blockDim.y) + threadIdx.y,
    __umul24(blockIdx.z, blockDim.z) + threadIdx.z);

  if (pos.z >= numSlices || pos.z < 0 || pos.x >= bias_slices.size.x || pos.y >= bias_slices.size.y
    || pos.x < 0 || pos.y < 0)
    return;

  unsigned int idx = pos.x + pos.y*bias_slices.size.x + pos.z*bias_slices.size.x*bias_slices.size.y;

  float s = slices[idx];
  if ((s == -1.0))
    return;

  float wb = wb_[idx];
  float wr = wr_[idx];

  if (wb > 0)
  {
    bias_slices.set(pos, bias_slices[pos] + wr / wb);
  }
}

__global__ void transformBiasMean(int numSlices,
  float* slices, Volume<float> bias_slices, float* d_means)
{
  const uint3 pos = make_uint3(__umul24(blockIdx.x, blockDim.x) + threadIdx.x,
    __umul24(blockIdx.y, blockDim.y) + threadIdx.y,
    __umul24(blockIdx.z, blockDim.z) + threadIdx.z);

  if (pos.z >= numSlices || pos.z < 0 || pos.x >= bias_slices.size.x || pos.y >= bias_slices.size.y
    || pos.x < 0 || pos.y < 0)
    return;

  unsigned int idx = pos.x + pos.y*bias_slices.size.x + pos.z*bias_slices.size.x*bias_slices.size.y;

  float s = slices[idx];
  if ((s == -1.0))
    return;
  float mean = d_means[pos.z];
  if ((mean == -1.0))
    return;
  if (mean != 0)
  {
    bias_slices.set(pos, bias_slices[pos] - mean);
  }
}


template< typename T >
class divS
{
public:
  T operator()(T a, T b)
  {
    return (b != 0) ? a / b : 0;
  }
};

template< typename T >
class divSame
{
public:
  T operator()(T a, T b)
  {
    return (b != 0) ? a / b : a;
  }
};


class divMin
{
public:
  float operator()(float a, float b)
  {
    return (b != 0) ? a / b : FLT_MIN;
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

struct is_greater_than
{
  float threshold;
  is_greater_than(float t) { threshold = t; }
  __host__ __device__
    bool operator()(float x) { return x > threshold; }
};


void Reconstruction::CorrectBias(float sigma_bias, bool _global_bias_correction)
{
  float* d_lbiasdata = 0;
  if (_debugGPU)
  {
    d_lbiasdata = v_bias.data;
  }
  if (multiThreadedGPU)
  {
    for (int d = 0; d < devicesToUse.size(); d++)
      GPU_workers[d]->prepareCorrectBias(sigma_bias, _global_bias_correction, d_lbiasdata);
    GPU_sync.runNextRound();
  }
  else
  {
    for (int d = 0; d < devicesToUse.size(); d++)
    {
      int dev = devicesToUse[d];
      CorrectBiasOnX(sigma_bias, _global_bias_correction, d_lbiasdata, dev);
    }
  }
  checkCudaErrors(cudaSetDevice(0));
}


void Reconstruction::CorrectBiasOnX(float sigma_bias, bool _global_bias_correction, float * d_lbiasdata, int dev)
{
  checkCudaErrors(cudaSetDevice(dev));
  //printf("CorrectBias device %d\n", dev);

  unsigned int start = dev_slice_range_offset[dev].x;
  unsigned int end = dev_slice_range_offset[dev].y;

  dim3 blockSize3 = dim3(8, 8, 8);
  dim3 gridSize3 = divup(dim3(dev_v_slices[dev].size.x, dev_v_slices[dev].size.y, dev_v_slices[dev].size.z), blockSize3);
  unsigned int N = dev_v_wb[dev].size.x*dev_v_wb[dev].size.y*dev_v_wb[dev].size.z;
  cudaMemsetAsync(dev_v_wb[dev].data, 0, N*sizeof(float));
  cudaMemsetAsync(dev_v_wresidual[dev].data, 0, N*sizeof(float));
  cudaMemsetAsync(dev_v_buffer[dev].data, 0, N*sizeof(float));

  calculateResidual3D_adv << <gridSize3, blockSize3, 0, streams[dev] >> >(dev_v_slices[dev].size.z, dev_v_slices[dev].data, dev_v_bias[dev].data, dev_v_weights[dev].data,
    dev_v_simulated_weights[dev].data, dev_v_simulated_slices[dev].data, dev_v_wb[dev], dev_v_wresidual[dev], dev_d_scales[dev]);

  uint2 vSize = make_uint2(dev_v_wb[dev].size.x, dev_v_wb[dev].size.y);

  blockSize3 = dim3(8, 8, 8);
  gridSize3 = divup(dim3(dev_v_slices[dev].size.x, dev_v_slices[dev].size.y, dev_v_slices[dev].size.z), blockSize3);

  //slice wise gauss for whole slice collection
  GaussianConvolutionKernel<float> << <gridSize3, blockSize3, 0, streams[dev] >> >(dev_v_slices[dev].size.z, dev_d_slice_sicesX[dev], dev_d_slice_sicesY[dev], dev_v_wb[dev].data, dev_v_buffer[dev], dev_d_sliceDims[dev], vSize, sigma_bias, true, 0);
  CHECK_ERROR(GaussianConvolutionKernel);
  GaussianConvolutionKernel<float> << <gridSize3, blockSize3, 0, streams[dev] >> >(dev_v_slices[dev].size.z, dev_d_slice_sicesX[dev], dev_d_slice_sicesY[dev], dev_v_buffer[dev].data, dev_v_wb[dev], dev_d_sliceDims[dev], vSize, sigma_bias, false, 0);
  GaussianConvolutionKernel<float> << <gridSize3, blockSize3, 0, streams[dev] >> >(dev_v_slices[dev].size.z, dev_d_slice_sicesX[dev], dev_d_slice_sicesY[dev], dev_v_wresidual[dev].data, dev_v_buffer[dev], dev_d_sliceDims[dev], vSize, sigma_bias, true, 0);
  CHECK_ERROR(GaussianConvolutionKernel);
  GaussianConvolutionKernel<float> << <gridSize3, blockSize3, 0, streams[dev] >> >(dev_v_slices[dev].size.z, dev_d_slice_sicesX[dev], dev_d_slice_sicesY[dev], dev_v_buffer[dev].data, dev_v_wresidual[dev], dev_d_sliceDims[dev], vSize, sigma_bias, false, 0);

  gridSize3 = divup(dim3(dev_v_bias[dev].size.x, dev_v_bias[dev].size.y, dev_v_bias[dev].size.z), blockSize3);
  updateBiasField3D_adv << <gridSize3, blockSize3, 0, streams[dev] >> >(dev_v_bias[dev].size.z, dev_v_slices[dev].data, dev_v_bias[dev], dev_v_wb[dev].data, dev_v_wresidual[dev].data);

  if (!_global_bias_correction)
  {
    //normalize bias field to have zero mean
    thrust::device_ptr<float> d_b(dev_v_bias[dev].data);
    thrust::device_ptr<float> d_s(dev_v_slices[dev].data);
    unsigned int N = dev_v_bias[dev].size.x*dev_v_bias[dev].size.y;
    //std::cout << " Bias num GPU: ";
    std::vector<float> h_means;
    for (unsigned int i = 0; i < dev_v_bias[dev].size.z; i++)
    {
      is_greater_than pred(-1);
      int num = count_if(d_s + i*N, d_s + ((i + 1)*N), pred);
      //std::cout << num << " ";
      double sum = reduce(d_b + i*N, d_b + ((i + 1)*N), 0.0, plus<float>());
      //std::cout << sum << " ";
      float mean = 0;
      if (num > 0)
      {
        mean = (float)(sum / (double)num);
        //std::cout << mean << " ";
        h_means.push_back(mean);
      }
      else
      {
        h_means.push_back(-1.0f); //indicator no mean
      }
    }

    thrust::device_vector<float> d_means(h_means.begin(), h_means.end());
    float* d_means_p = thrust::raw_pointer_cast(&d_means[0]);
    blockSize3 = dim3(8, 8, 8);
    gridSize3 = divup(dim3(dev_v_bias[dev].size.x, dev_v_bias[dev].size.y, dev_v_bias[dev].size.z), blockSize3);
    transformBiasMean << <gridSize3, blockSize3, 0, streams[dev] >> >(dev_v_bias[dev].size.z, dev_v_slices[dev].data, dev_v_bias[dev], d_means_p);
  }
  else
  {
    std::cout << "_global_bias_correction is not implemented in CUDA yet" << std::endl;
  }

  if (_debugGPU)
  {
    //TODO DEBUG copy back
    unsigned int num_elems = (end - start)*dev_v_bias[dev].size.x*dev_v_bias[dev].size.y;
    checkCudaErrors(cudaMemcpyAsync(d_lbiasdata, dev_v_bias[dev].data, num_elems*sizeof(float), cudaMemcpyDefault));
  }
  checkCudaErrors(cudaStreamSynchronize(streams[dev]));
}

__global__ void AdaptiveRegularizationPrep(bool _adaptive, float _alpha, Volume<float> reconstructed, Volume<float> addon, Volume<float> confidence_map,
  float _min_intensity, float _max_intensity, int zofs, int thisrun)
{
  const uint3 pos = make_uint3(blockIdx.x * blockDim.x + threadIdx.x,
    blockIdx.y * blockDim.y + threadIdx.y,
    blockIdx.z * blockDim.z + threadIdx.z + zofs);

  if (pos.z > thisrun) return;

  if (!_adaptive)
  {
    if (confidence_map[pos] != 0)
    {
      addon.set(pos, addon[pos] / confidence_map[pos]);
      confidence_map.set(pos, 1.0);
    }
  }

  reconstructed.set(pos, reconstructed[pos] + addon[pos] * _alpha);

  if (reconstructed[pos] < _min_intensity * 0.9)
    reconstructed.set(pos, _min_intensity * 0.9);
  if (reconstructed[pos] > _max_intensity * 1.1)
    reconstructed.set(pos, _max_intensity * 1.1);

}


__global__ void	weightedResidulaKernel(Volume<float> original, Volume<float> residual, Volume<float> weights, float _low_intensity_cutoff,
  float _max_intensity)
{

  const uint3 pos = make_uint3(blockIdx.x * blockDim.x + threadIdx.x,
    blockIdx.y * blockDim.y + threadIdx.y,
    blockIdx.z * blockDim.z + threadIdx.z);

  if (pos.x >= original.size.x || pos.y >= original.size.y || pos.z >= original.size.z
    || pos.x < 0 || pos.y < 0 || pos.z < 0)
    return;

  if ((weights[pos] == 1) && (original[pos] > _low_intensity_cutoff * _max_intensity)
    && (residual[pos] > _low_intensity_cutoff * _max_intensity))
  {
    if (original[pos] != 0)
    {
      float val = residual[pos] / original[pos];
      residual.set(pos, log(val));
    }
  }
  else
  {
    residual.set(pos, 0.0);
    weights.set(pos, 0.0);
  }

}

__global__ void	calcBiasFieldKernel(Volume<float> reconstructed, Volume<float> residual, Volume<float> weights, Volume<float> mask,
  float _min_intensity, float _max_intensity)
{
  const uint3 pos = make_uint3(blockIdx.x * blockDim.x + threadIdx.x,
    blockIdx.y * blockDim.y + threadIdx.y,
    blockIdx.z * blockDim.z + threadIdx.z);

  if (pos.x >= reconstructed.size.x || pos.y >= reconstructed.size.y || pos.z >= reconstructed.size.z
    || pos.x < 0 || pos.y < 0 || pos.z < 0)
    return;

  if (mask[pos] == 1.0 && weights[pos] != 0)
  {
    float val = residual[pos] / weights[pos];
    val = exp(val);
    residual.set(pos, val);
    float rval = reconstructed[pos] / val;
    if (rval  < _min_intensity * 0.9)
      rval = _min_intensity * 0.9;
    if (rval > _max_intensity * 1.1)
      rval = _max_intensity * 1.1;

    reconstructed.set(pos, rval);
  }
  else
  {
    residual.set(pos, 0.0);
  }
}

void Reconstruction::syncCPU(float* reconstructed)
{
  cudaDeviceSynchronize();
  checkCudaErrors(cudaSetDevice(0));
  cudaMemcpy(reconstructed, dev_reconstructed_[0].data, dev_reconstructed_[0].size.x*dev_reconstructed_[0].size.y*dev_reconstructed_[0].size.z*sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
}

void Reconstruction::SyncConfidenceMapAddon(float* cmdata, float* addondata)
{
  cudaMemcpy(cmdata, dev_confidence_map_[0].data, dev_confidence_map_[0].size.x*dev_confidence_map_[0].size.y*dev_confidence_map_[0].size.z*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(addondata, dev_addon_[0].data, dev_addon_[0].size.x*dev_addon_[0].size.y*dev_addon_[0].size.z*sizeof(float), cudaMemcpyDeviceToHost);
}

//original without prep
__device__ float AdaptiveRegularization1(int i, uint3 pos, uint3 pos2, Volume<float> original, Volume<float> confidence_map, float delta)
{
  if (pos.x >= original.size.x || pos.y >= original.size.y || pos.z >= original.size.z
    || pos.x < 0 || pos.y < 0 || pos.z < 0 || confidence_map[pos] <= 0 || confidence_map[pos2] <= 0 ||
    pos2.x >= original.size.x || pos2.y >= original.size.y || pos2.z >= original.size.z
    || pos2.x < 0 || pos2.y < 0 || pos2.z < 0)
    return 0.0;

  //central differences would be better... improve with texture linear interpolation
  float diff = (original[pos2] - original[pos]) * sqrt(d_factor[i]) / delta;
  return d_factor[i] / sqrt(1.0 + diff * diff);
}

// 50% occupancy, 3.5 ms -- improve
//original with prep
__global__ void AdaptiveRegularizationKernel(Volume<float> reconstructed, Volume<float> original,
  Volume<float> confidence_map, float delta, float alpha, float lambda, int zofs, int thisrun)
{
  uint3 pos = make_uint3(blockIdx.x * blockDim.x + threadIdx.x,
    blockIdx.y * blockDim.y + threadIdx.y,
    blockIdx.z * blockDim.z + threadIdx.z + zofs);

  if (pos.z > thisrun) return;

  float val = 0;
  float valW = 0;
  float sum = 0;

  for (int i = 0; i < 13; i++)
  {
    uint3 pos2 = make_uint3(pos.x + d_directions[i][0], pos.y + d_directions[i][1], pos.z + d_directions[i][2]);

    if ((pos2.x >= 0) && (pos2.x  < original.size.x) && (pos2.y >= 0) && (pos2.y < original.size.y)
      && (pos2.z >= 0) && (pos2.z < original.size.z))
    {
      float bi = AdaptiveRegularization1(i, pos, pos2, original, confidence_map, delta);
      val += bi * reconstructed[pos2] * confidence_map[pos2]; //reconstructed == original2
      valW += bi * confidence_map[pos2];
      sum += bi;
    }

    uint3 pos3 = make_uint3(pos.x - d_directions[i][0], pos.y - d_directions[i][1], pos.z - d_directions[i][2]); //recycle pos register

    if ((pos3.x >= 0) && (pos3.x < original.size.x) && (pos3.y >= 0) && (pos3.y < original.size.y)
      && (pos3.z >= 0) && (pos3.z < original.size.z) &&
      (pos2.x >= 0) && (pos2.x  < original.size.x) && (pos2.y >= 0) && (pos2.y < original.size.y)
      && (pos2.z >= 0) && (pos2.z < original.size.z)
      )
    {
      float bi = AdaptiveRegularization1(i, pos3, pos2, original, confidence_map, delta);
      val += bi * reconstructed[pos2] * confidence_map[pos2];
      valW += bi * confidence_map[pos2];
      sum += bi;
    }

  }

  val -= sum * reconstructed[pos] * confidence_map[pos];
  valW -= sum * confidence_map[pos];

  val = reconstructed[pos] * confidence_map[pos] + alpha * lambda / (delta * delta) * val;
  valW = confidence_map[pos] + alpha * lambda / (delta * delta) * valW;

  if (valW > 0.0) {
    reconstructed.set(pos, val / valW);
  }
  else
  {
    reconstructed.set(pos, 0.0);
  }

}

void Reconstruction::Superresolution(int iter, std::vector<float> _slice_weight, bool _adaptive, float alpha,
  float _min_intensity, float _max_intensity, float delta, float lambda, bool _global_bias_correction, float sigma_bias,
  float _low_intensity_cutoff)
{
  UpdateSliceWeights(_slice_weight);
  if (alpha * lambda / (delta * delta) > 0.068)
  {
    printf("Warning: regularization might not have smoothing effect! Ensure that alpha*lambda/delta^2 is below 0.068.");
  }

  checkCudaErrors(cudaSetDevice(0));

  unsigned int N = dev_addon_[0].size.x*dev_addon_[0].size.y*dev_addon_[0].size.z;

  Volume<float> dev_addon_accbuf_;
  dev_addon_accbuf_.init(dev_reconstructed_[0].size, dev_reconstructed_[0].dim);
  Volume<float> dev_cmap_accbuf_;
  dev_cmap_accbuf_.init(dev_reconstructed_[0].size, dev_reconstructed_[0].dim);

  Volume<float> original;
  original.init(dev_reconstructed_[0].size, dev_reconstructed_[0].dim);
  checkCudaErrors(cudaMemcpy(original.data, dev_reconstructed_[0].data,
    dev_reconstructed_[0].size.x*dev_reconstructed_[0].size.y*dev_reconstructed_[0].size.z*sizeof(float), cudaMemcpyDeviceToDevice));

  if (multiThreadedGPU)
  {
    for (int d = 0; d < devicesToUse.size(); d++)
      GPU_workers[d]->prepareSuperresolution(N, dev_addon_accbuf_, dev_cmap_accbuf_, original);
    GPU_sync.runNextRound();
  }
  else
  {
    for (int d = 0; d < devicesToUse.size(); d++)
    {
      int dev = devicesToUse[d];
      SuperresolutionOnX1(N, dev_addon_accbuf_, dev_cmap_accbuf_, original, dev);
    }
  }

  //TODO try to improve with MRF instead of anisotropic diffusion -- get enhanced segmentation

  //TODO multi-GPU regularization -- distribute addon and cmap to GPUs, integrate and regularize multiGPU?
  checkCudaErrors(cudaSetDevice(0));
  dim3 blockSize = dim3(8, 8, 10);
  dim3 gridSize = divup(dim3(dev_addon_[0].size.x, dev_addon_[0].size.y, dev_addon_[0].size.z), blockSize);

  AdaptiveRegularizationPrep << <gridSize, blockSize, 0, streams[0] >> >(_adaptive, alpha, dev_reconstructed_[0],
    dev_addon_[0], dev_confidence_map_[0], _min_intensity, _max_intensity, 0, dev_reconstructed_[0].size.z);

  AdaptiveRegularizationKernel << <gridSize, blockSize, 0, streams[0] >> >(dev_reconstructed_[0], original, dev_confidence_map_[0],
    delta, alpha, lambda, 0, dev_reconstructed_[0].size.z);

  for (int d = 1; d < devicesToUse.size(); d++)
  {
    int dev = devicesToUse[d];
    checkCudaErrors(cudaMemcpy(dev_reconstructed_[dev].data, dev_reconstructed_[0].data,
      N*sizeof(float), cudaMemcpyDefault));
  }

  if (_global_bias_correction)
  {
    printf("_global_bias_correction not implemented\n");
  }

  dev_addon_accbuf_.release();
  dev_cmap_accbuf_.release();
  original.release();
}


void Reconstruction::SuperresolutionOnX1(int N, Volume<float>& dev_addon_accbuf_, Volume<float>& dev_cmap_accbuf_, Volume<float>& original, int dev)
{
  checkCudaErrors(cudaSetDevice(dev));
  //printf("Superresolution device %d\n", dev);
  //checkGPUMemory();

  int start = dev_slice_range_offset[dev].x;
  int end = dev_slice_range_offset[dev].y;

  cudaMemsetAsync(dev_addon_[dev].data, 0, N*sizeof(float));
  cudaMemsetAsync(dev_confidence_map_[dev].data, 0, N*sizeof(float));

  float psfsize = dev_addon_[dev].dim.x / h_quality_factor;
  int rest = dev_v_slices[dev].size.z;
  for (int i = 0; i < ceil(dev_v_slices[dev].size.z / (float)MAX_SLICES_PER_RUN); i++)
  {
    int thisrun = (rest >= MAX_SLICES_PER_RUN) ? MAX_SLICES_PER_RUN : rest;
    dim3 blockSize3 = dim3(8, 8, 4);
    dim3 gridSize3 = divup(dim3(dev_v_slices[dev].size.x, dev_v_slices[dev].size.y, thisrun), blockSize3);

    SuperresolutionKernel3D_tex << <gridSize3, blockSize3, 0, streams[dev] >> >(dev_v_slices[dev].size.z, dev_v_slices[dev].data, dev_v_bias[dev].data, dev_v_weights[dev].data,
      dev_v_simulated_slices[dev].data, dev_d_slice_weights[dev], dev_d_scales[dev], dev_mask_[dev].data, dev_v_PSF_sums_[dev].data,
      dev_addon_[dev], dev_confidence_map_[dev].data,
      dev_d_slicesI2W[dev], dev_d_slicesW2I[dev], dev_d_slicesTransformation[dev], dev_d_slicesInvTransformation[dev], dev_d_sliceDims[dev],
      reconstructedVoxelSize, i*MAX_SLICES_PER_RUN, thisrun, make_ushort2(dev_v_slices[dev].size.x, dev_v_slices[dev].size.y), psfsize);

    rest -= MAX_SLICES_PER_RUN;
  }

  CHECK_ERROR(SuperresolutionKernel3D_tex);

  //copy addon back to gpu 0
  if (dev > 0)
  {
    checkCudaErrors(cudaSetDevice(0));
    checkCudaErrors(cudaMemcpyAsync(dev_addon_accbuf_.data, dev_addon_[dev].data,
      N*sizeof(float), cudaMemcpyDefault));
    thrust::device_ptr<float> ptr_addon(dev_addon_[0].data);
    thrust::device_ptr<float> ptr_dev_addon(dev_addon_accbuf_.data);
    thrust::transform(ptr_addon, ptr_addon + N, ptr_dev_addon, ptr_addon, plus<float>());

    checkCudaErrors(cudaMemcpyAsync(dev_cmap_accbuf_.data, dev_confidence_map_[dev].data,
      N*sizeof(float), cudaMemcpyDefault));
    thrust::device_ptr<float> ptr_cmap(dev_confidence_map_[0].data);
    thrust::device_ptr<float> ptr_dev_cmap(dev_cmap_accbuf_.data);
    thrust::transform(ptr_cmap, ptr_cmap + N, ptr_dev_cmap, ptr_cmap, plus<float>());
  }
  checkCudaErrors(cudaStreamSynchronize(streams[dev]));
}

struct transformRS
{
  transformRS(){}

  __host__ __device__
    thrust::tuple<float, unsigned int> operator()(const thrust::tuple<float, char, float, float>& v)
  {
    float s_ = thrust::get<0>(v);
    char si_ = thrust::get<1>(v);
    float ss_ = thrust::get<2>(v);
    float sw_ = thrust::get<3>(v);

    thrust::tuple<float, unsigned int> t = make_tuple(0.0, 0);

    if (s_ != -1 && si_ == 1 && sw_ > 0.99)
    {
      float sval = s_ - ss_;
      t = make_tuple(sval*sval, 1);
    }
    return t;
  }

};

struct reduceRS
{
  reduceRS(){}

  __host__ __device__
    thrust::tuple<float, unsigned int> operator()(const thrust::tuple<float, unsigned int>& a,
    const thrust::tuple<float, unsigned int>& b)
  {
    return make_tuple(thrust::get<0>(a)+thrust::get<0>(b), thrust::get<1>(a)+thrust::get<1>(b));
  }
};

void Reconstruction::InitializeRobustStatistics(float& _sigma)
{
  float sa = 0;
  float sb = 0;
  for (int d = 0; d < devicesToUse.size(); d++)
  {
    int dev = devicesToUse[d];
    checkCudaErrors(cudaSetDevice(dev));
    // printf("InitializeRobustStatistics device %d\n", dev);

    thrust::device_ptr<float> d_s(dev_v_slices[dev].data);
    thrust::device_ptr<char> d_si(dev_v_simulated_inside[dev].data);
    thrust::device_ptr<float> d_ss(dev_v_simulated_slices[dev].data);
    thrust::device_ptr<float> d_sw(dev_v_simulated_weights[dev].data);

    unsigned int N = dev_v_slices[dev].size.x*dev_v_slices[dev].size.y*dev_v_slices[dev].size.z;

    thrust::tuple<float, unsigned int> out = transform_reduce(make_zip_iterator(make_tuple(d_s, d_si, d_ss, d_sw)),
      make_zip_iterator(make_tuple(d_s + N, d_si + N, d_ss + N, d_sw + N)), transformRS(),
      make_tuple(0.0, 0), reduceRS()); //+

    sa += get<0>(out);
    sb += (float)get<1>(out);

  }
  _sigma = sa / sb;

  checkCudaErrors(cudaSetDevice(0));
}


void Reconstruction::GaussianReconstruction(std::vector<int>& voxel_num)
{
  voxel_num.resize(devicesToUse.size(), 0);

  unsigned int N = dev_reconstructed_[0].size.x*dev_reconstructed_[0].size.y*dev_reconstructed_[0].size.z;
  checkCudaErrors(cudaSetDevice(0));

  Volume<float> dev_reconstructed_accbuf_;
  dev_reconstructed_accbuf_.init(dev_reconstructed_[0].size, dev_reconstructed_[0].dim);

  if (multiThreadedGPU)
  {
    for (int d = 0; d < devicesToUse.size(); d++)
      GPU_workers[d]->prepareGaussianReconstruction1(voxel_num[d], dev_reconstructed_accbuf_);
    GPU_sync.runNextRound();
  }
  else
  {
    for (int d = 0; d < devicesToUse.size(); d++)
    {
      int dev = devicesToUse[d];
      GaussianReconstructionOnX1(voxel_num[d], dev_reconstructed_accbuf_, dev);
    }
  }

  checkCudaErrors(cudaSetDevice(0));

  thrust::device_ptr<float> ptr_recons(dev_reconstructed_[0].data);
  thrust::device_ptr<float> ptr_count(dev_reconstructed_volWeigths[0].data);
  thrust::transform(ptr_recons, ptr_recons + N, ptr_count, ptr_recons, divS<float>());

  //update other GPUs
  for (int d = 1; d < devicesToUse.size(); d++)
  {
    int dev = devicesToUse[d];
    checkCudaErrors(cudaMemcpy(dev_reconstructed_[dev].data, dev_reconstructed_[0].data,
      N*sizeof(float), cudaMemcpyDefault));
  }

  dev_reconstructed_accbuf_.release();

  if (multiThreadedGPU)
  {
    for (int d = 0; d < devicesToUse.size(); d++)
      GPU_workers[d]->prepareGaussianReconstruction2(voxel_num[d]);
    GPU_sync.runNextRound();
  }
  else
  {
    for (int d = 0; d < devicesToUse.size(); d++)
    {
      int dev = devicesToUse[d];
      GaussianReconstructionOnX2(voxel_num[d], dev);
    }
  }

  CHECK_ERROR(gaussianReconstructionKernel3Dcount);
}

void Reconstruction::GaussianReconstructionOnX1(int& voxel_num, Volume<float>& dev_reconstructed_accbuf_, int dev)
{
  checkCudaErrors(cudaSetDevice(dev));
  int start = dev_slice_range_offset[dev].x;
  int end = dev_slice_range_offset[dev].y;
  unsigned int N = dev_reconstructed_[0].size.x*dev_reconstructed_[0].size.y*dev_reconstructed_[0].size.z;

  unsigned int N2 = dev_v_weights[dev].size.x*dev_v_weights[dev].size.y*dev_v_weights[dev].size.z;
  cudaMemset(dev_v_weights[dev].data, 0, N2*sizeof(float));
  cudaMemset(dev_v_simulated_weights[dev].data, 0, N2*sizeof(float));
  cudaMemset(dev_v_simulated_slices[dev].data, 0, N2*sizeof(float));
  cudaMemset(dev_v_wresidual[dev].data, 0, N2*sizeof(float));
  cudaMemset(dev_v_wb[dev].data, 0, N2*sizeof(float));
  cudaMemset(dev_v_buffer[dev].data, 0, N2*sizeof(float));
  cudaMemset(dev_v_simulated_inside[dev].data, 0, N2*sizeof(char));
  cudaMemset(dev_sliceVoxel_count_[dev].data, 0, N2*sizeof(int));
  cudaMemset(dev_reconstructed_[dev].data, 0, N*sizeof(float));
  cudaMemset(dev_reconstructed_volWeigths[dev].data, 0, N*sizeof(float));

  short rest = dev_v_slices[dev].size.z;
  for (short i = 0; i < ceil(dev_v_slices[dev].size.z / (float)MAX_SLICES_PER_RUN_GAUSS); i++)
  {
    short thisrun = (rest >= MAX_SLICES_PER_RUN_GAUSS) ? MAX_SLICES_PER_RUN_GAUSS : rest;

    dim3 blockSize3 = dim3(8, 8, 10);
    dim3 gridSize3 = divup(dim3(dev_v_slices[dev].size.x, dev_v_slices[dev].size.y, thisrun), blockSize3);

    checkCudaErrors(cudaDeviceSynchronize());

    gaussianReconstructionKernel3D_tex << <gridSize3, blockSize3 >> >(dev_v_slices[dev].size.z,
      dev_v_slices[dev].data, dev_v_bias[dev].data, dev_d_scales[dev], dev_reconstructed_[dev], dev_reconstructed_volWeigths[dev],
      dev_sliceVoxel_count_[dev], dev_v_PSF_sums_[dev], dev_mask_[dev].data,
      make_uint2(dev_v_slices[dev].size.x, dev_v_slices[dev].size.y), dev_d_slicesI2W[dev], dev_d_slicesW2I[dev],
      dev_d_slicesTransformation[dev], dev_d_slicesInvTransformation[dev], dev_d_sliceDims[dev], dev_d_sliceDim[dev],
      reconstructedVoxelSize, i*(MAX_SLICES_PER_RUN_GAUSS), thisrun);

    checkCudaErrors(cudaDeviceSynchronize());

    rest -= MAX_SLICES_PER_RUN_GAUSS;
  }
  if (_debugGPU)
  {
    checkCudaErrors(cudaDeviceSynchronize());

    unsigned int ofs = start*dev_sliceVoxel_count_[dev].size.x*dev_sliceVoxel_count_[dev].size.y;
    checkCudaErrors(cudaMemcpyAsync(sliceVoxel_count_.data + ofs, dev_sliceVoxel_count_[dev].data,
      dev_sliceVoxel_count_[dev].size.x*dev_sliceVoxel_count_[dev].size.y*dev_sliceVoxel_count_[dev].size.z*sizeof(int), cudaMemcpyDefault));

    checkCudaErrors(cudaMemcpyAsync(v_PSF_sums_.data + ofs, dev_v_PSF_sums_[dev].data,
      dev_v_PSF_sums_[dev].size.x*dev_v_PSF_sums_[dev].size.y*dev_v_PSF_sums_[dev].size.z*sizeof(float), cudaMemcpyDefault));
  }
  if (dev > 0)
  {
    //collect data from other GPUs -- needs to be done that way because of atomics in kernels
    checkCudaErrors(cudaSetDevice(0));
    checkCudaErrors(cudaMemcpyAsync(dev_reconstructed_accbuf_.data, dev_reconstructed_[dev].data,
      N*sizeof(float), cudaMemcpyDefault));
    thrust::device_ptr<float> ptr_recons(dev_reconstructed_[0].data);
    thrust::device_ptr<float> ptr_dev_recons(dev_reconstructed_accbuf_.data);
    thrust::transform(ptr_recons, ptr_recons + N, ptr_dev_recons, ptr_recons, plus<float>());

    checkCudaErrors(cudaMemcpyAsync(dev_reconstructed_accbuf_.data, dev_reconstructed_volWeigths[dev].data,
      N*sizeof(float), cudaMemcpyDefault));
    thrust::device_ptr<float> ptr_recons_vW(dev_reconstructed_volWeigths[0].data);
    thrust::device_ptr<float> ptr_dev_recons_vW(dev_reconstructed_accbuf_.data);
    thrust::transform(ptr_recons_vW, ptr_recons_vW + N, ptr_dev_recons_vW, ptr_recons_vW, plus<float>());
  }
  checkCudaErrors(cudaDeviceSynchronize());
}

void Reconstruction::GaussianReconstructionOnX2(int& voxel_num, int dev)
{
  checkCudaErrors(cudaSetDevice(dev));
  //printf("slice voxel count %d\n", dev);
  for (unsigned int i = 0; i < dev_sliceVoxel_count_[dev].size.z; i++)
  {
    checkCudaErrors(cudaDeviceSynchronize());
    unsigned int N2 = dev_sliceVoxel_count_[dev].size.x*dev_sliceVoxel_count_[dev].size.y;
    thrust::device_ptr<int> ptr_count(dev_sliceVoxel_count_[dev].data + (i*dev_sliceVoxel_count_[dev].size.x*dev_sliceVoxel_count_[dev].size.y));
    //only ++ if n > 0 == set pixel
    int vnum = thrust::count_if(ptr_count, ptr_count + N2, is_larger_zero());
    voxel_num = vnum;
  }
}

template< typename T >
class maskS
{
public:
  T operator()(T a, T b)
  {
    return (b != 0) ? a : 0;
  }
};

template< typename T >
class divexp
{
public:
  T operator()(T a, T b)
  {
    return  (a != -1.0) ? (a / exp(-b)) : a;
  }
};


void Reconstruction::NormaliseBias(int iter, float sigma_bias)
{
  checkCudaErrors(cudaSetDevice(0));

  unsigned int N = dev_reconstructed_[0].size.x*dev_reconstructed_[0].size.y*dev_reconstructed_[0].size.z;
  Volume<float> dev_bias_accbuf_;
  dev_bias_accbuf_.init(dev_reconstructed_[0].size, dev_reconstructed_[0].dim);

  Volume<float> dev_volume_weights_accbuf_;
  dev_volume_weights_accbuf_.init(dev_reconstructed_[0].size, dev_reconstructed_[0].dim);

  if (multiThreadedGPU)
  {
    for (int d = 0; d < devicesToUse.size(); d++)
      GPU_workers[d]->prepareNormaliseBias(iter, dev_bias_accbuf_, sigma_bias);
    GPU_sync.runNextRound();
  }
  else
  {
    for (int d = 0; d < devicesToUse.size(); d++)
    {
      int dev = devicesToUse[d];
      NormaliseBiasOnX(iter, dev_bias_accbuf_, dev_volume_weights_accbuf_, sigma_bias, dev);
    }
  }

  checkCudaErrors(cudaSetDevice(0));

  dev_bias_accbuf_.release();
  dev_volume_weights_accbuf_.release();

  //single GPU finalization
  thrust::device_ptr<float> ptr_bias(dev_bias_[0].data);

  thrust::device_ptr<float> ptr_volWeights(dev_reconstructed_volWeigths[0].data);
  thrust::transform(ptr_bias, ptr_bias + N, ptr_volWeights, ptr_bias, divS<float>()); //TODO check if divS or divSame

  /* thrust::device_ptr<float> ptr_weights(dev_volume_weights_[0].data);
  thrust::transform(ptr_weights, ptr_weights + N, ptr_volWeights, ptr_weights, divS<float>());*/

  dim3 blockSize3k = dim3(8, 8, 8);
  dim3 gridSize3k = divup(dim3(dev_bias_[0].size.x, dev_bias_[0].size.y, dev_bias_[0].size.z), blockSize3k);
  Volume<float> mbuf;
  mbuf.init(dev_bias_[0].size, dev_bias_[0].dim);
  GaussianConvolutionKernel3D << <gridSize3k, blockSize3k, 0, streams[0] >> >(dev_bias_[0].data, mbuf.data, sigma_bias, 0, dev_bias_[0].dim, dev_bias_[0].size);
  GaussianConvolutionKernel3D << <gridSize3k, blockSize3k, 0, streams[0] >> >(mbuf.data, dev_bias_[0].data, sigma_bias, 1, dev_bias_[0].dim, dev_bias_[0].size);
  GaussianConvolutionKernel3D << <gridSize3k, blockSize3k, 0, streams[0] >> >(dev_bias_[0].data, mbuf.data, sigma_bias, 2, dev_bias_[0].dim, dev_bias_[0].size);
  checkCudaErrors(cudaMemcpy(dev_bias_[0].data, mbuf.data, dev_bias_[0].size.x*dev_bias_[0].size.y*dev_bias_[0].size.z*sizeof(float), cudaMemcpyDeviceToDevice));
  cudaDeviceSynchronize();
  CHECK_ERROR(GaussianConvolutionKernel3D);
  //bias/=m;
  thrust::device_ptr<float> ptr_mbuf(maskC_.data);
  thrust::transform(ptr_bias, ptr_bias + N, ptr_mbuf, ptr_bias, divS<float>());

  // *pi /=exp(-(*pb));
  thrust::device_ptr<float> ptr_reconstructed(dev_reconstructed_[0].data);
  thrust::transform(ptr_reconstructed, ptr_reconstructed + N, ptr_bias, ptr_reconstructed, divexp<float>());

  /* thrust::device_ptr<float> ptr_combinedVolWeights(dev_volume_weights_[0].data);
  thrust::device_ptr<float> ptr_count(dev_reconstructed_volWeigths[0].data);
  thrust::transform(ptr_combinedVolWeights, ptr_combinedVolWeights + N, ptr_count, ptr_combinedVolWeights, divS<float>());*/

  //updat eother GPUs
  for (int d = 1; d < devicesToUse.size(); d++)
  {
    int dev = devicesToUse[d];
    checkCudaErrors(cudaMemcpy(dev_reconstructed_[dev].data, dev_reconstructed_[0].data,
      N*sizeof(float), cudaMemcpyDefault));
    checkCudaErrors(cudaMemcpy(dev_bias_[dev].data, dev_bias_[0].data,
      N*sizeof(float), cudaMemcpyDefault)); //might be not neccessary
    checkCudaErrors(cudaMemcpy(dev_volume_weights_[dev].data, dev_volume_weights_[0].data,
      N*sizeof(float), cudaMemcpyDefault)); //might be not neccessary
  }



  CHECK_ERROR(NormaliseBias);
  mbuf.release();

}

void Reconstruction::NormaliseBiasOnX(int iter, Volume<float>& dev_bias_accbuf_, Volume<float>& dev_volume_weights_accbuf_, float sigma_bias, int dev)
{
  unsigned int N = dev_reconstructed_[0].size.x*dev_reconstructed_[0].size.y*dev_reconstructed_[0].size.z;
  checkCudaErrors(cudaSetDevice(dev));
  // printf("NormaliseBias device %d\n", dev);

  int start = dev_slice_range_offset[dev].x;
  int end = dev_slice_range_offset[dev].y;

  checkCudaErrors(cudaMemsetAsync(dev_bias_[dev].data, 0, dev_bias_[dev].size.x*dev_bias_[dev].size.y*dev_bias_[dev].size.z*sizeof(float)));

  int rest = dev_v_slices[dev].size.z;
  for (int i = 0; i < ceil(dev_v_slices[dev].size.z / (float)MAX_SLICES_PER_RUN); i++)
  {
    int thisrun = (rest >= MAX_SLICES_PER_RUN) ? MAX_SLICES_PER_RUN : rest;

    dim3 blockSize3 = dim3(8, 8, 10);
    dim3 gridSize3 = divup(dim3(dev_v_slices[dev].size.x, dev_v_slices[dev].size.y, thisrun), blockSize3);

    normalizeBiasKernel3D_tex << <gridSize3, blockSize3, 0, streams[dev] >> >(dev_v_slices[dev].size.z, dev_v_slices[dev], dev_v_bias[dev],
      dev_d_scales[dev], dev_mask_[dev].data, dev_v_PSF_sums_[dev].data,
      dev_d_slicesI2W[dev], dev_d_slicesW2I[dev], dev_d_slicesTransformation[dev], dev_d_slicesInvTransformation[dev],
      dev_d_sliceDims[dev], reconstructedVoxelSize, dev_bias_[dev], dev_reconstructed_volWeigths[dev], dev_volume_weights_[dev], i*(MAX_SLICES_PER_RUN), thisrun);
    cudaDeviceSynchronize();
    rest -= MAX_SLICES_PER_RUN;
  }

  //copy back to GPU 0 if dev > 0
  if (dev > 0)
  {
    checkCudaErrors(cudaSetDevice(0));
    checkCudaErrors(cudaMemcpyAsync(dev_bias_accbuf_.data, dev_bias_[dev].data,
      N*sizeof(float), cudaMemcpyDefault));
    thrust::device_ptr<float> ptr_bias(dev_bias_[0].data);
    thrust::device_ptr<float> ptr_dev_bias(dev_bias_accbuf_.data);
    thrust::transform(ptr_bias, ptr_bias + N, ptr_dev_bias, ptr_bias, plus<float>());

    checkCudaErrors(cudaSetDevice(0));
    checkCudaErrors(cudaMemcpyAsync(dev_volume_weights_accbuf_.data, dev_volume_weights_[dev].data,
      N*sizeof(float), cudaMemcpyDefault));
    thrust::device_ptr<float> ptr_volume_weights_(dev_volume_weights_[0].data);
    thrust::device_ptr<float> ptr_dev_volume_weights_(dev_volume_weights_accbuf_.data);
    thrust::transform(ptr_volume_weights_, ptr_volume_weights_ + N, ptr_dev_volume_weights_, ptr_volume_weights_, plus<float>());
  }
}

void Reconstruction::SimulateSlices(std::vector<bool>& slice_inside)
{
  unsigned int coffset = 0;

  if (multiThreadedGPU)
  {
    for (int d = 0; d < devicesToUse.size(); d++)
    {
      int dev = devicesToUse[d];
      GPU_workers[d]->prepareSimulateSlices(slice_inside.begin() + coffset);
      coffset += dev_v_simulated_inside[dev].size.z;
    }
    GPU_sync.runNextRound();
  }
  else
  {
    for (int d = 0; d < devicesToUse.size(); d++)
    {
      int dev = devicesToUse[d];
      SimulateSlicesOnX(slice_inside.begin() + coffset, dev);
      coffset += dev_v_simulated_inside[dev].size.z;
    }
  }
  checkCudaErrors(cudaSetDevice(0));

  //check if necessary here
  unsigned int N = dev_bias_[0].size.x*dev_bias_[0].size.y*dev_bias_[0].size.z;
  thrust::device_ptr<float> ptr_mask(dev_mask_[0].data);
  thrust::device_ptr<float> ptr_reconstructed(dev_reconstructed_[0].data);
  thrust::transform(ptr_reconstructed, ptr_reconstructed + N, ptr_mask, ptr_reconstructed, maskS<float>());
  CHECK_ERROR(simulated_inside masking);
}

void Reconstruction::SimulateSlicesOnX(std::vector<bool>::iterator slice_inside, int dev)
{
  checkCudaErrors(cudaSetDevice(dev));
  //printf("SimulateSlices device %d\n", dev);

  int start = dev_slice_range_offset[dev].x;
  int end = dev_slice_range_offset[dev].y;

  int rest = dev_v_slices[dev].size.z;
  for (int i = 0; i < ceil(dev_v_slices[dev].size.z / (float)MAX_SLICES_PER_RUN); i++)
  {
    int thisrun = (rest >= MAX_SLICES_PER_RUN) ? MAX_SLICES_PER_RUN : rest;
    dim3 blockSize3 = dim3(8, 8, 9);
    dim3 gridSize3 = divup(dim3(dev_v_slices[dev].size.x, dev_v_slices[dev].size.y, thisrun), blockSize3);

    simulateSlicesKernel3D_tex << <gridSize3, blockSize3, 0, streams[dev] >> >(dev_v_slices[dev].size.z, dev_v_slices[dev].data,
      dev_v_simulated_slices[dev], dev_v_simulated_weights[dev], dev_v_simulated_inside[dev],
      dev_reconstructed_[dev], dev_mask_[dev].data, dev_v_PSF_sums_[dev].data,
      make_uint2(dev_v_slices[dev].size.x, dev_v_slices[dev].size.y),
      dev_d_slicesI2W[dev], dev_d_slicesW2I[dev], dev_d_slicesTransformation[dev], dev_d_slicesInvTransformation[dev], dev_d_sliceDims[dev],
      reconstructedVoxelSize, i*MAX_SLICES_PER_RUN, thisrun);

    rest -= MAX_SLICES_PER_RUN;
  }
  CHECK_ERROR(simulateSlicesKernel3D());

  if (_debugGPU)
  {

    float* d_lssdata = v_simulated_slices.data + start*dev_v_simulated_slices[dev].size.x*dev_v_simulated_slices[dev].size.y;;
    float* d_lswdata = v_simulated_weights.data + start*dev_v_simulated_slices[dev].size.x*dev_v_simulated_slices[dev].size.y;;
    char* d_lsidata = v_simulated_inside.data + start*dev_v_simulated_slices[dev].size.x*dev_v_simulated_slices[dev].size.y;;

    //TODO temporary output back on device 0
    unsigned int num_elems = (end - start)*dev_v_simulated_slices[dev].size.x*dev_v_simulated_slices[dev].size.y;
    checkCudaErrors(cudaMemcpyAsync(d_lssdata, dev_v_simulated_slices[dev].data,
      dev_v_simulated_slices[dev].size.x*dev_v_simulated_slices[dev].size.y*dev_v_simulated_slices[dev].size.z*sizeof(float), cudaMemcpyDefault));
    d_lssdata += num_elems;

    checkCudaErrors(cudaMemcpyAsync(d_lswdata, dev_v_simulated_weights[dev].data,
      dev_v_simulated_weights[dev].size.x*dev_v_simulated_weights[dev].size.y*dev_v_simulated_weights[dev].size.z*sizeof(float), cudaMemcpyDefault));
    d_lswdata += num_elems;

    checkCudaErrors(cudaMemcpyAsync(d_lsidata, dev_v_simulated_inside[dev].data,
      dev_v_simulated_inside[dev].size.x*dev_v_simulated_inside[dev].size.y*dev_v_simulated_inside[dev].size.z*sizeof(char), cudaMemcpyDefault));
    d_lsidata += num_elems;
    //TODO temporary output back on device 0 end
  }

  for (unsigned int i = 0; i < dev_v_simulated_inside[dev].size.z; i++)
  {
    unsigned int N = dev_v_simulated_inside[dev].size.x*dev_v_simulated_inside[dev].size.y;
    thrust::device_ptr<char> d_si(dev_v_simulated_inside[dev].data + (i*dev_v_simulated_inside[dev].size.x*dev_v_simulated_inside[dev].size.y));
    int h_sliceInside = thrust::count(d_si, d_si + N, 1); //need to... is synchronous

    if (h_sliceInside > 0)
      slice_inside[i] = true;
    else
      slice_inside[i] = false;
  }
  CHECK_ERROR(simulated_inside count);

  //check if necessary here
  unsigned int N = dev_bias_[0].size.x*dev_bias_[0].size.y*dev_bias_[0].size.z;
  thrust::device_ptr<float> ptr_mask(dev_mask_[dev].data);
  thrust::device_ptr<float> ptr_reconstructed(dev_reconstructed_[dev].data);
  thrust::transform(ptr_reconstructed, ptr_reconstructed + N, ptr_mask, ptr_reconstructed, maskS<float>());
  CHECK_ERROR(simulated_inside masking);
}


__global__ void EStepKernel3D_tex(int numSlices, float* __restrict slices, float* __restrict bias, float* weights,
  float* __restrict simslices, float* __restrict simweights, float* __restrict scales,
  float _m, float _sigma, float _mix, uint2 vSize)
{

  const uint3 pos = make_uint3(blockIdx.x * blockDim.x + threadIdx.x,
    blockIdx.y * blockDim.y + threadIdx.y,
    blockIdx.z * blockDim.z + threadIdx.z);

  //z is slice index
  if (pos.z >= numSlices || pos.z < 0)
    return;

  unsigned int idx = pos.x + pos.y*vSize.x + pos.z*vSize.x*vSize.y;

  float s = slices[idx];
  float sw = simweights[idx];

  if ((s == -1) || sw <= 0)
    return;

  float b = bias[idx];
  float ss = simslices[idx];
  float scale = scales[pos.z];

  float sliceVal = s * exp(-b) * scale;

  sliceVal -= ss;

  //Gaussian distribution for inliers (likelihood)
  float g = G_(sliceVal, _sigma);
  //Uniform distribution for outliers (likelihood)
  float m = M_(_m);

  float weight = (g * _mix) / (g *_mix + m * (1.0 - _mix));
  if (sw > 0)
  {
    weights[idx] = weight;
  }
  else
  {
    weights[idx] = 0.0;
  }
}


struct transformSlicePotential
{
  __host__ __device__
    tuple<float, float> operator()(const tuple<float, float>& a)
  {

    if (thrust::get<1>(a) > 0.99)
    {
      return thrust::make_tuple(((1.0 - thrust::get<0>(a)) * (1.0 - thrust::get<0>(a))), 1.0);
    }
    else
    {
      return thrust::make_tuple(0.0, 0.0);
    }

  }
};

struct reduceSlicePotential
{
  __host__ __device__
    tuple<float, float> operator()(const tuple<float, float>& a, const tuple<float, float>& b)
  {
    return thrust::make_tuple(thrust::get<0>(a) +thrust::get<0>(b), thrust::get<1>(a) +thrust::get<1>(b));
  }
};


void Reconstruction::EStep(float _m, float _sigma, float _mix, std::vector<float>& slice_potential)
{

  float* d_lweigthsdata = 0;
  if (/*_debugGPU*/true)
  {
    d_lweigthsdata = v_weights.data;
  }


  /*if (multiThreadedGPU)
  {
  for (int d = 0; d < devicesToUse.size(); d++)
  GPU_workers[d]->prepareEStep(_m, _sigma, _mix, slice_potential, d_lweigthsdata);
  GPU_sync.runNextRound();
  }
  else
  {*/
  for (int d = 0; d < devicesToUse.size(); d++)
  {
    int dev = devicesToUse[d];
    //printf("EStep %d\n", devicesToUse.size());
    EStepOnX(_m, _sigma, _mix, slice_potential, d_lweigthsdata, dev);
  }
  //}
  checkCudaErrors(cudaSetDevice(0));

}

void Reconstruction::EStepOnX(float _m, float _sigma, float _mix, std::vector<float>& slice_potential, float* d_lweigthsdata, int dev)
{
  checkCudaErrors(cudaSetDevice(dev));
  //printf("Estep %d\n", dev);
  int start = dev_slice_range_offset[dev].x;
  int end = dev_slice_range_offset[dev].y;

  unsigned int N = dev_v_weights[dev].size.x*dev_v_weights[dev].size.y*dev_v_weights[dev].size.z;
  cudaMemsetAsync(dev_v_weights[dev].data, 0, N*sizeof(float));

  dim3 blockSize3 = dim3(8, 8, 8);
  dim3 gridSize3 = divup(dim3(dev_v_slices[dev].size.x, dev_v_slices[dev].size.y, dev_v_slices[dev].size.z), blockSize3);

  EStepKernel3D_tex << <gridSize3, blockSize3, 0, streams[dev] >> >(dev_v_slices[dev].size.z, dev_v_slices[dev].data, dev_v_bias[dev].data, dev_v_weights[dev].data,
    dev_v_simulated_slices[dev].data, dev_v_simulated_weights[dev].data, dev_d_scales[dev], _m, _sigma, _mix,
    make_uint2(dev_v_slices[dev].size.x, dev_v_slices[dev].size.y));

  CHECK_ERROR(EStepKernel3D_tex);

  for (unsigned int i = 0; i < dev_v_slices[dev].size.z; i++)
  {

    unsigned int N = dev_v_weights[dev].size.x*dev_v_weights[dev].size.y;

    thrust::device_ptr<float> d_w(dev_v_weights[dev].data + (i*dev_v_weights[dev].size.x*dev_v_weights[dev].size.y));//w->data());
    thrust::device_ptr<float> d_sw(dev_v_simulated_weights[dev].data + (i*dev_v_simulated_weights[dev].size.x*dev_v_simulated_weights[dev].size.y));//sw->data());
    tuple<float, float> out = transform_reduce(make_zip_iterator(make_tuple(d_w, d_sw)), make_zip_iterator(make_tuple(d_w + N, d_sw + N)),
      transformSlicePotential(), make_tuple(0.0, 0.0), reduceSlicePotential());

    if (thrust::get<1>(out) > 0)
    {
      slice_potential[start + i] = sqrt(thrust::get<0>(out) / thrust::get<1>(out));
    }
    else
    {
      slice_potential[start + i] = -1; // slice has no unpadded voxels
    }
  }


  if (/*_debugGPU*/true)
  {
    //DEBUG copy back
    checkCudaErrors(cudaDeviceSynchronize());

    unsigned int ofs = start*dev_v_weights[dev].size.x*dev_v_weights[dev].size.y;

    checkCudaErrors(cudaMemcpyAsync(v_weights.data + ofs, dev_v_weights[dev].data,
      dev_v_weights[dev].size.x*dev_v_weights[dev].size.y*dev_v_weights[dev].size.z*sizeof(float), cudaMemcpyDefault));
  }
  checkCudaErrors(cudaStreamSynchronize(streams[dev]));
}

struct transformMStep3D
{
  __host__ __device__
    thrust::tuple<float, float, float, float, float> operator()(const thrust::tuple<float, float, float, float, float, float>& v)
    //thrust::tuple<sigma_, mix_, count, e, e> //this order is very important for the thrust optization
  {
    const float s_ = thrust::get<0>(v);
    const float b_ = thrust::get<1>(v);
    const float w_ = thrust::get<2>(v);
    const float ss_ = thrust::get<3>(v);
    const float sw_ = thrust::get<4>(v);
    const float scale = thrust::get<5>(v);

    float sigma_ = 0.0;
    float mix_ = 0.0;
    float count = 0.0;
    float e = 0.0;

    thrust::tuple<float, float, float, float, float> t;

    if (s_ != -1.0  && sw_ > 0.99)
    {
      e = (s_*exp(-b_) * scale) - ss_;
      sigma_ = e * e * w_;
      mix_ = w_;
      count = 1.0;
      float e1 = e;
      t = thrust::make_tuple(sigma_, mix_, count, e, e1);
    }
    else
    {
      t = thrust::make_tuple(0.0, 0.0, 0.0, DBL_MAX, DBL_MIN);
    }
    return t;
  }
};


struct reduceMStep
{
  __host__ __device__
    thrust::tuple<float, float, float, float, float> operator()(const thrust::tuple<float, float, float, float, float>& a,
    const thrust::tuple<float, float, float, float, float>& b)
  {
    return thrust::make_tuple(thrust::get<0>(a)+thrust::get<0>(b), thrust::get<1>(a)+thrust::get<1>(b),
      thrust::get<2>(a)+thrust::get<2>(b), min(thrust::get<3>(a), thrust::get<3>(b)),
      max(thrust::get<4>(a), thrust::get<4>(b)));
  }
};

void Reconstruction::MStep(int iter, float _step, float& _sigma, float& _mix, float& _m)
{
  float num = 0;
  float min_ = FLT_MAX;
  float max_ = FLT_MIN;
  float sigma = 0;
  float mix = 0;

  /* if (multiThreadedGPU)
  {
  std::vector<thrust::tuple<float, float, float, float, float> > results(devicesToUse.size());
  for (int d = 0; d < devicesToUse.size(); d++)
  GPU_workers[d]->prepareMStep(results[d]);
  GPU_sync.runNextRound();
  for (int d = 0; d < devicesToUse.size(); d++)
  {
  sigma += get<0>(results[d]);
  mix += get<1>(results[d]);
  num += get<2>(results[d]);
  min_ = min(min_, get<3>(results[d]));
  max_ = max(max_, get<4>(results[d]));
  }
  }
  else
  {*/
  for (int d = 0; d < devicesToUse.size(); d++)
  {
    int dev = devicesToUse[d];
    checkCudaErrors(cudaSetDevice(dev));
    //printf("MStep device %d\n", dev);

    thrust::tuple<float, float, float, float, float> results;
    MStepOnX(results, dev);

    sigma += get<0>(results);
    mix += get<1>(results);
    num += get<2>(results);
    min_ = min(min_, get<3>(results));
    max_ = max(max_, get<4>(results));
  }
  //}
  checkCudaErrors(cudaSetDevice(0));
  //printf("GPU sigma %f, mix %f, num %f, min_ %f, max_ %f\n", sigma, mix, num, min_, max_);

  if (mix > 0) {
    _sigma = sigma / mix;
  }
  else {
    printf("Something went wrong: sigma= %f mix= %f\n", sigma, mix);
    //exit(1);
  }
  if (_sigma < _step * _step / 6.28f)
    _sigma = _step * _step / 6.28f;
  if (iter > 1)
    _mix = mix / num;

  //Calculate m
  _m = 1.0f / (max_ - min_);
}

void Reconstruction::MStepOnX(thrust::tuple<float, float, float, float, float> &results, int dev)
{
  checkCudaErrors(cudaSetDevice(dev));
  //printf("MStep device %d\n", dev);

  int start = dev_slice_range_offset[dev].x;
  int end = dev_slice_range_offset[dev].y;

  thrust::device_ptr<float> d_s(dev_v_slices[dev].data);
  thrust::device_ptr<float> d_b(dev_v_bias[dev].data);
  thrust::device_ptr<float> d_w(dev_v_weights[dev].data);
  thrust::device_ptr<float> d_ss(dev_v_simulated_slices[dev].data);
  thrust::device_ptr<float> d_sw(dev_v_simulated_weights[dev].data);
  thrust::device_ptr<float> d_buf(dev_v_buffer[dev].data);

  unsigned int N1 = dev_v_buffer[dev].size.x*dev_v_buffer[dev].size.y;

  for (unsigned int i = 0; i < dev_v_buffer[dev].size.z; i++)
  {
    initMem<float>(dev_v_buffer[dev].data + i*N1, N1, h_scales[start + i]);
  }

  unsigned int N3 = dev_v_buffer[dev].size.x*dev_v_buffer[dev].size.y*dev_v_buffer[dev].size.z;

  results = transform_reduce(make_zip_iterator(make_tuple(d_s, d_b, d_w, d_ss, d_sw, d_buf)),
    make_zip_iterator(make_tuple(d_s + N3, d_b + N3, d_w + N3, d_ss + N3, d_sw + N3, d_buf + N3)), transformMStep3D(),
    make_tuple<float, float, float, float, float>(0.0, 0.0, 0.0, 0.0, 0.0), reduceMStep());
}

struct transformScale
{
  transformScale(){}

  __host__ __device__
    thrust::tuple<float, float> operator()(const thrust::tuple<float, float, float, float, float>& v)
  {
    float s_ = thrust::get<0>(v);
    const float b_ = thrust::get<1>(v);
    const float w_ = thrust::get<2>(v);
    const float ss_ = thrust::get<3>(v);
    const float sw_ = thrust::get<4>(v);

    if ((s_ == -1) || sw_ <= 0.99)
    {
      return make_tuple(0.0, 0.0);
    }
    else
    {
      float eb = exp(-(b_));
      float scalenum = w_ * s_ * eb * ss_;
      float scaleden = w_ * s_ * eb * s_ * eb;
      return make_tuple(scalenum, scaleden);
    }
  }
};


struct reduceScale
{
  reduceScale(){}

  __host__ __device__
    thrust::tuple<float, float> operator()(const thrust::tuple<float, float>& a, const thrust::tuple<float, float>& b)
  {
    return make_tuple(thrust::get<0>(a)+thrust::get<0>(b), thrust::get<1>(a)+thrust::get<1>(b));
  }
};

void Reconstruction::CalculateScaleVector(std::vector<float>& scale_vec)
{
  if (multiThreadedGPU)
  {
    for (int d = 0; d < devicesToUse.size(); d++)
      GPU_workers[d]->prepareCalculateScaleVector(scale_vec);
    GPU_sync.runNextRound();
  }
  else
  {
    for (int d = 0; d < devicesToUse.size(); d++)
    {
      int dev = devicesToUse[d];
      CalculateScaleVectorOnX(scale_vec, dev);
    }
  }

  h_scales = scale_vec;
}

void Reconstruction::CalculateScaleVectorOnX(std::vector<float>& scale_vec, int dev)
{
  checkCudaErrors(cudaSetDevice(dev));

  int start = dev_slice_range_offset[dev].x;
  int end = dev_slice_range_offset[dev].y;

  unsigned int N = dev_v_slices[dev].size.x*dev_v_slices[dev].size.y;
  for (unsigned int i = 0; i < dev_v_slices[dev].size.z; i++)
  {
    thrust::device_ptr<float> d_s(dev_v_slices[dev].data + (i*N));
    thrust::device_ptr<float> d_b(dev_v_bias[dev].data + (i*N));
    thrust::device_ptr<float> d_w(dev_v_weights[dev].data + (i*N));
    thrust::device_ptr<float> d_ss(dev_v_simulated_slices[dev].data + (i*N));
    thrust::device_ptr<float> d_sw(dev_v_simulated_weights[dev].data + (i*N));

    thrust::tuple<float, float> out = transform_reduce(make_zip_iterator(make_tuple(d_s, d_b, d_w, d_ss, d_sw)),
      make_zip_iterator(make_tuple(d_s + N, d_b + N, d_w + N, d_ss + N, d_sw + N)), transformScale(),
      make_tuple(0.0, 0.0), reduceScale());

    if (thrust::get<1>(out) != 0.0)
    {
      scale_vec[start + i] = thrust::get<0>(out) / thrust::get<1>(out);
    }
    else
    {
      scale_vec[start + i] = 1.0;
    }
  }
  checkCudaErrors(cudaMemcpyAsync(dev_d_scales[dev], &h_scales[start], dev_v_slices[dev].size.z*sizeof(float), cudaMemcpyHostToDevice));
}

__global__ void InitializeEMValuesKernel(int numSlices, int* __restrict d_sliceSicesX, int* __restrict d_sliceSicesY,
  float* __restrict slices, float* bias, float* weights, uint2 vSize)
{
  const int3 pos = make_int3((blockIdx.x* blockDim.x) + threadIdx.x,
    (blockIdx.y* blockDim.y) + threadIdx.y,
    (blockIdx.z* blockDim.z) + threadIdx.z);

  //z is slice index
  if (pos.z >= numSlices || pos.z < 0)
    return;

  //int ssizeX = d_sliceSicesX[pos.z];
  //int ssizeY = d_sliceSicesY[pos.z];
  unsigned int idx = pos.x + pos.y*vSize.x + pos.z*vSize.x*vSize.y;

  float s = slices[idx];

  if (s != -1)
  {
    weights[idx] = 1;
  }
  else
  {
    weights[idx] = 0;
  }
  bias[idx] = 0;
}

void Reconstruction::InitializeEMValues()
{
  if (multiThreadedGPU)
  {
    for (int d = 0; d < devicesToUse.size(); d++)
      GPU_workers[d]->prepareInitializeEMValues();
    GPU_sync.runNextRound();
  }
  else
  {
    for (int d = 0; d < devicesToUse.size(); d++)
    {
      int dev = devicesToUse[d];
      InitializeEMValuesOnX(dev);
    }
  }
  CHECK_ERROR(InitializeEMValuesKernel);
  checkCudaErrors(cudaSetDevice(0));
}


void Reconstruction::InitializeEMValuesOnX(int dev)
{
  int start = dev_slice_range_offset[dev].x;
  int end = dev_slice_range_offset[dev].y;

  checkCudaErrors(cudaSetDevice(dev));
  // printf("InitializeEMValues device %d\n", dev);

  dim3 blockSize3 = dim3(8, 8, 8);
  dim3 gridSize3 = divup(dim3(dev_v_slices[dev].size.x, dev_v_slices[dev].size.y, dev_v_slices[dev].size.z), blockSize3);

  //printf("%llx %llx %llx %llx %llx\n",dev_d_slice_sicesX[dev], dev_d_slice_sicesY[dev], dev_v_slices[dev].data, dev_v_bias[dev].data, dev_v_weights[dev].data); 
  InitializeEMValuesKernel << <gridSize3, blockSize3 >> >(dev_v_slices[dev].size.z, dev_d_slice_sicesX[dev], dev_d_slice_sicesY[dev],
    dev_v_slices[dev].data, dev_v_bias[dev].data, dev_v_weights[dev].data, make_uint2(dev_v_slices[dev].size.x, dev_v_slices[dev].size.y));
}


__global__ void maskVolumeKernel(Volume<float> reconstructed, Volume<float> mask)
{
  const uint3 pos = make_uint3(blockIdx.x * blockDim.x + threadIdx.x,
    blockIdx.y * blockDim.y + threadIdx.y,
    blockIdx.z * blockDim.z + threadIdx.z);

  if (pos.x >= reconstructed.size.x || pos.y >= reconstructed.size.y || pos.z >= reconstructed.size.z
    || pos.x < 0 || pos.y < 0 || pos.z < 0)
    return;

  if (mask[pos] == 0)
    reconstructed.set(pos, -1);

}

void Reconstruction::maskVolume()
{
  printf("masking volume \n");
  checkCudaErrors(cudaSetDevice(0));
  dim3 blockSize = dim3(8, 8, 8);
  dim3 gridSize = divup(dim3(dev_reconstructed_[0].size.x, dev_reconstructed_[0].size.y, dev_reconstructed_[0].size.z), blockSize);

  maskVolumeKernel << <gridSize, blockSize, 0, streams[0] >> >(dev_reconstructed_[0], dev_mask_[0]);

  //update other GPUs
  unsigned int N = dev_reconstructed_[0].size.x*dev_reconstructed_[0].size.y*dev_reconstructed_[0].size.z;
  for (int d = 1; d < devicesToUse.size(); d++)
  {
    int dev = devicesToUse[d];
    checkCudaErrors(cudaMemcpy(dev_reconstructed_[dev].data, dev_reconstructed_[0].data,
      N*sizeof(float), cudaMemcpyDefault));
  }

  CHECK_ERROR(maskVolume);
}

__global__ void RestoreSliceIntensitiesKernel(float* slices, float* stack_factors, int* stack_index_, uint2 vSize, int num_slices)
{
  const uint3 pos = make_uint3(__umul24(blockIdx.x, blockDim.x) + threadIdx.x,
    __umul24(blockIdx.y, blockDim.y) + threadIdx.y,
    __umul24(blockIdx.z, blockDim.z) + threadIdx.z);

  if (pos.z < 0 || pos.z >= num_slices)
    return;

  float factor = stack_factors[stack_index_[pos.z]];

  unsigned int idx = pos.x + pos.y*vSize.x + pos.z*vSize.x*vSize.y;
  float s = slices[idx];
  if (s > 0)
  {
    slices[idx] = s / factor;
  }

}

void Reconstruction::RestoreSliceIntensities(std::vector<float> stack_factors_, std::vector<int> stack_index_)
{
  //single GPU task
  dim3 blockSize = dim3(8, 8, 8);
  dim3 gridSize = divup(dim3(v_slices.size.x, v_slices.size.y, v_slices.size.z), blockSize);

  thrust::device_vector<float> d_stack_factors_(stack_factors_.begin(), stack_factors_.end());
  float* d_stack_factors_p = thrust::raw_pointer_cast(&d_stack_factors_[0]);
  thrust::device_vector<int> d_stack_index_(stack_index_.begin(), stack_index_.end());
  int* d_stack_index_p = thrust::raw_pointer_cast(&d_stack_index_[0]);
  uint2 vSize = make_uint2(v_slices.size.x, v_slices.size.y);

  RestoreSliceIntensitiesKernel << <gridSize, blockSize, 0, streams[0] >> >(v_slices.data, d_stack_factors_p, d_stack_index_p, vSize, v_slices.size.z);
  CHECK_ERROR(RestoreSliceIntensitiesKernel);
}


__global__ void ScaleVolumeKernel(float* slices, float* scalenum, float* scaleden,
  float* simweights, float* simslices, float* weights, float* slice_weight, uint2 vSize, int num_slices)
{
  const uint3 pos = make_uint3(__umul24(blockIdx.x, blockDim.x) + threadIdx.x,
    __umul24(blockIdx.y, blockDim.y) + threadIdx.y,
    __umul24(blockIdx.z, blockDim.z) + threadIdx.z);

  if (pos.z < 0 || pos.z >= num_slices)
    return;

  unsigned int idx = pos.x + pos.y*vSize.x + pos.z*vSize.x*vSize.y;
  float s = slices[idx];
  if (s == -1)
  {
    return;
  }
  float sw = simweights[idx];
  if (sw <= 0.99)
  {
    return;
  }
  float ss = simslices[idx];
  float w = weights[idx];
  float slicew = slice_weight[pos.z];

  scalenum[idx] = w*slicew*s*ss;
  scaleden[idx] = w*slicew*ss*ss;
}

__global__ void scaleVolumeKernel(Volume<float> vol, float scale)
{
  const uint3 pos = make_uint3(__umul24(blockIdx.x, blockDim.x) + threadIdx.x,
    __umul24(blockIdx.y, blockDim.y) + threadIdx.y,
    __umul24(blockIdx.z, blockDim.z) + threadIdx.z);

  if (vol[pos] > 0)
    vol.set(pos, vol[pos] * scale);
}

void Reconstruction::ScaleVolume()
{
  double scalenum = 0, scaleden = 0;

  for (int d = 0; d < devicesToUse.size(); d++)
  {
    int dev = devicesToUse[d];
    checkCudaErrors(cudaSetDevice(dev));
    printf("ScaleVolume device %d\n", dev);

    unsigned int start = dev_slice_range_offset[dev].x;
    unsigned int end = dev_slice_range_offset[dev].y;

    unsigned int N = dev_v_buffer[dev].size.x*dev_v_buffer[dev].size.y*dev_v_buffer[dev].size.z;
    cudaMemsetAsync(dev_v_buffer[dev].data, 0, N*sizeof(float));
    cudaMemsetAsync(dev_v_wresidual[dev].data, 0, N*sizeof(float));

    dim3 blockSize = dim3(8, 8, 8);
    dim3 gridSize = divup(dim3(dev_v_slices[dev].size.x, dev_v_slices[dev].size.y, dev_v_slices[dev].size.z), blockSize);
    uint2 vSize = make_uint2(dev_v_slices[dev].size.x, dev_v_slices[dev].size.y);

    ScaleVolumeKernel << <gridSize, blockSize, 0, streams[dev] >> >(dev_v_slices[dev].data, dev_v_buffer[dev].data, dev_v_wresidual[dev].data,
      dev_v_simulated_weights[dev].data, dev_v_simulated_slices[dev].data,
      dev_v_weights[dev].data, dev_d_slice_weights[dev], vSize, dev_v_slices[dev].size.z);
    CHECK_ERROR(ScaleVolumeKernel);

    thrust::device_ptr<float> d_n(dev_v_buffer[dev].data);
    thrust::device_ptr<float> d_d(dev_v_wresidual[dev].data);
    scalenum += reduce(d_n, d_n + N, 0.0, plus<float>());
    scaleden += reduce(d_d, d_d + N, 0.0, plus<float>());

  }
  checkCudaErrors(cudaSetDevice(0));

  float scale = scalenum / scaleden;
  printf("Volume scale GPU: %f\n", scale);

  dim3 blockSize = dim3(8, 8, 8);
  dim3 gridSize = divup(dim3(dev_reconstructed_[0].size.x, dev_reconstructed_[0].size.y, dev_reconstructed_[0].size.z), blockSize);

  scaleVolumeKernel << <gridSize, blockSize, 0, streams[0] >> >(dev_reconstructed_[0], scale);

  checkCudaErrors(cudaDeviceSynchronize());

  //no update of other GPUs necessary
}

/////////////////////////////////////////////////////////////////////////////////////////////
//Registration test MULTI GPU

__global__ void genenerateRegistrationSlicesVolume(Volume<float> regSlices, Matrix4* d_slicesI2Winit,
  Matrix4* d_slicesTransformation, int numSlices, uint3 reconsize)
{
  //z indicates slice number
  const uint3 pos = make_uint3(__umul24(blockIdx.x, blockDim.x) + threadIdx.x,
    __umul24(blockIdx.y, blockDim.y) + threadIdx.y,
    __umul24(blockIdx.z, blockDim.z) + threadIdx.z);

  if (pos.z >= numSlices || pos.z < 0)
    return;

  Matrix4 sliceM = d_slicesI2Winit[pos.z];
  Matrix4 T = d_slicesTransformation[pos.z];

  float3 slicePos = make_float3((float)pos.x, (float)pos.y, 0);
  float3 wpos = sliceM*slicePos;
  wpos = T*wpos;
  float3 volumePos = d_reconstructedW2I * wpos;

  if (pos.x < regSlices.size.x && pos.y < regSlices.size.y && pos.z < regSlices.size.z &&
    pos.x >= 0 && pos.y >= 0 && pos.z >= 0)
  {
    float val = tex3D(reconstructedTex_, volumePos.x / (float)reconsize.x, volumePos.y / (float)reconsize.y, volumePos.z / (float)reconsize.z);
    if (val > 0)
      regSlices.set(make_uint3(pos.x, pos.y, pos.z), val);
  }

}

__global__ void genenerateRegistrationSlices(cudaSurfaceObject_t out, int* activelayers, Matrix4* d_slicesI2Winit,
  Matrix4* trans, Matrix4* ofsSlice, float3 reconsize, uint2 sliceSize, int insliceofs)
{
  //z indicates slice number
  const uint3 pos = make_uint3(blockIdx.x * blockDim.x + threadIdx.x,
    blockIdx.y * blockDim.y + threadIdx.y,
    activelayers[blockIdx.z]);

  if (pos.x >= sliceSize.x || pos.y >= sliceSize.y)
    return;

  //  Matrix4 sliceM = d_slicesI2Winit[pos.z];
  Matrix4 T = trans[pos.z]; //T is optimized
  Matrix4 sliceOfs = ofsSlice[pos.z];

  float3 slicePos = make_float3((float)pos.x, (float)pos.y, insliceofs * 2);
  float3 wpos = /*sliceM **/ sliceOfs * slicePos;
  wpos = T*wpos;
  float3 volumePos = d_reconstructedW2I * wpos;


  float val = tex3D(reconstructedTex_, volumePos.x / reconsize.x, volumePos.y / reconsize.y, volumePos.z / reconsize.z);
  if (val < 0)
    val = -1.0f;
  surf2DLayeredwrite(val, out, pos.x * 4, pos.y, blockIdx.z, cudaBoundaryModeZero);
}




__global__ void genenerateRegistrationSliceNoOfs(float* sampledSlice, int sliceNum, Matrix4* d_slicesI2Winit,
  Matrix4 trans, uint3 reconsize, uint2 sliceSize)
{
  //z indicates slice number
  const uint3 pos = make_uint3(__umul24(blockIdx.x, blockDim.x) + threadIdx.x,
    __umul24(blockIdx.y, blockDim.y) + threadIdx.y,
    sliceNum);

  if (pos.x < 0 || pos.y < 0 || pos.z < 0 || pos.x >= sliceSize.x || pos.y >= sliceSize.y)
    return;

  Matrix4 sliceM = d_slicesI2Winit[sliceNum];
  Matrix4 T = trans; //T is optimized
  //Matrix4 sliceI2W = T*sliceM;

  //d_reconstructedW2I d_reconstructedRegOffset d_reconstructedRegOffsetInv
  float3 slicePos = make_float3((float)pos.x, (float)pos.y, 0);
  float3 wpos = sliceM * slicePos;
  wpos = T*wpos;
  float3 volumePos = d_reconstructedW2I * wpos;

  if (pos.x < sliceSize.x && pos.y < sliceSize.y  &&
    pos.x >= 0 && pos.y >= 0)
  {
    float val = tex3D(reconstructedTex_, volumePos.x / (float)reconsize.x, volumePos.y / (float)reconsize.y, volumePos.z / (float)reconsize.z);
    if (val > 0)
      sampledSlice[pos.x + pos.y*sliceSize.x] = val;
  }

}

__global__ void genenerateRegistrationSlice(float* sampledSlice, int sliceNum, Matrix4* d_slicesI2Winit,
  Matrix4 trans, Matrix4* ofsSlice, uint3 reconsize, uint2 sliceSize, int insliceofs)
{
  //z indicates slice number
  const uint3 pos = make_uint3(__umul24(blockIdx.x, blockDim.x) + threadIdx.x,
    __umul24(blockIdx.y, blockDim.y) + threadIdx.y,
    sliceNum);

  if (pos.x < 0 || pos.y < 0 || pos.z < 0 || pos.x >= sliceSize.x || pos.y >= sliceSize.y)
    return;

  //  Matrix4 sliceM = d_slicesI2Winit[sliceNum];
  Matrix4 T = trans; //T is optimized
  Matrix4 sliceOfs = ofsSlice[sliceNum];

  float3 slicePos = make_float3((float)pos.x, (float)pos.y, insliceofs * 2);
  float3 wpos = /*sliceM **/ sliceOfs * slicePos;
  wpos = T*wpos;
  float3 volumePos = d_reconstructedW2I * wpos;

  if (pos.x < sliceSize.x && pos.y < sliceSize.y  &&
    pos.x >= 0 && pos.y >= 0)
  {
    float val = tex3D(reconstructedTex_, volumePos.x / (float)reconsize.x, volumePos.y / (float)reconsize.y, volumePos.z / (float)reconsize.z);
    if (val > 0)
      sampledSlice[pos.x + pos.y*sliceSize.x] = val;
  }
}



//((N*s*st) - (s-st))/sqrt((N*s*s - s*s)*(N*st*st - st*st))
struct transformNCCopt
{
  unsigned int N;
  transformNCCopt(unsigned int N_){ N = N_; }

  __host__ __device__
    float operator()(const thrust::tuple<float, float>& v)
  {
    float s_ = thrust::get<0>(v);
    float stex_ = thrust::get<1>(v);

    float n = N*s_*stex_ - s_*stex_;
    float d = (N*s_*s_ - s_*s_)*(N*stex_*stex_ - stex_*stex_);

    return n * rsqrtf(d);
  }

};


template<typename T>
struct sumPad //: public binary_function<T,T,T>
{
  __host__ __device__ T operator()(const T &a, const T &b) const
  {
    if (a == -1) return b;
    else if (b == -1) return a;
    else return a + b;
  }
};

void Reconstruction::Matrix2Parameters(Matrix4 m, float* params)
{
  float tmp;
  float TOL = 0.000001f;

  params[TX] = m.data[0].w;
  params[TY] = m.data[1].w;
  params[TZ] = m.data[2].w;

  tmp = asin(-1.0f * m.data[0].z);

  // asin returns values for tmp in range -pi/2 to +pi/2, i.e. cos(tmp) >=
  // 0 so the division by cos(tmp) in the first part of the if clause was
  // not needed.
  if (fabs(cos(tmp)) > TOL) {
    params[RX] = atan2(m.data[1].z, m.data[2].z);
    params[RY] = tmp;
    params[RZ] = atan2(m.data[0].y, m.data[0].x);
  }
  else {
    //m(0,2) is close to +1 or -1
    params[RX] = atan2(-1.0f*m.data[0].z*m.data[1].x, -1.0f*m.data[0].z*m.data[2].x);
    params[RY] = tmp;
    params[RZ] = 0;
  }

  // Convert to degrees.
  params[RX] *= 180.0f / M_PI;
  params[RY] *= 180.0f / M_PI;
  params[RZ] *= 180.0f / M_PI;

}

Matrix4 Reconstruction::Parameters2Matrix(float *params)
{
  float tx = params[TX];
  float ty = params[TY];
  float tz = params[TZ];

  float rx = params[RX];
  float ry = params[RY];
  float rz = params[RZ];

  float cosrx = cos(rx*(M_PI / 180.0f));
  float cosry = cos(ry*(M_PI / 180.0f));
  float cosrz = cos(rz*(M_PI / 180.0f));
  float sinrx = sin(rx*(M_PI / 180.0f));
  float sinry = sin(ry*(M_PI / 180.0f));
  float sinrz = sin(rz*(M_PI / 180.0f));

  // Create a transformation whose transformation matrix is an identity matrix
  Matrix4 mat;
  identityM(mat);

  // Add other transformation parameters to transformation matrix
  mat.data[0].x = cosry*cosrz;
  mat.data[0].y = cosry*sinrz;
  mat.data[0].z = -sinry;
  mat.data[0].w = tx;

  mat.data[1].x = (sinrx*sinry*cosrz - cosrx*sinrz);
  mat.data[1].y = (sinrx*sinry*sinrz + cosrx*cosrz);
  mat.data[1].z = sinrx*cosry;
  mat.data[1].w = ty;

  mat.data[2].x = (cosrx*sinry*cosrz + sinrx*sinrz);
  mat.data[2].y = (cosrx*sinry*sinrz - sinrx*cosrz);
  mat.data[2].z = cosrx*cosry;
  mat.data[2].w = tz;
  mat.data[3].w = 1.0f;

  return mat;
}

//return (_xy - (_x * _y) / _n) / (sqrt(_x2 - _x * _x / _n) * sqrt(_y2 - _y *_y / _n));
/*struct transformCCITPadIRTK2
{
int level;
transformCCITPadIRTK2(int level_){level = level_;}

__host__ __device__
thrust::tuple<float,float,float> operator()(const thrust::tuple<float,float,int>& a)
{
if(thrust::get<0>(a) >= 0 && thrust::get<1>(a) >= 0  && thrust::get<2>(a)%level == 0)
{
float x = (float)thrust::get<0>(a);
float y = (float)thrust::get<1>(a);
return thrust::make_tuple((x*y) -
//return thrust::make_tuple(x*y, x, x*x, y, y*y, 1.0);
}
else
{
return thrust::make_tuple(0,0,0,0,0,0);
}

}
};*/

struct transformCCITPadIRTK
{
  uint64_t level;
  transformCCITPadIRTK(uint64_t level_){ level = level_; }

  __host__ __device__
    thrust::tuple<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t> operator()
    (const thrust::tuple<float, float, uint64_t>& a)
  {
    if (thrust::get<0>(a) >= 0 && thrust::get<1>(a) >= 0 && thrust::get<2>(a) % level == 0)
    {
      uint64_t x = (uint64_t)thrust::get<0>(a);
      uint64_t y = (uint64_t)thrust::get<1>(a);
      return thrust::make_tuple(x*y, x, x*x, y, y*y, 1.0);
    }
    else
    {
      return thrust::make_tuple(0, 0, 0, 0, 0, 0);
    }

  }
};

struct reduceCCITPadIRTK
{
  reduceCCITPadIRTK(){}

  __host__ __device__
    thrust::tuple<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t> operator()(
    const thrust::tuple<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t>& a,
    const thrust::tuple<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t>& b)
  {
    return thrust::make_tuple(thrust::get<0>(a)+thrust::get<0>(b), thrust::get<1>(a)+thrust::get<1>(b),
      thrust::get<2>(a)+thrust::get<2>(b), thrust::get<3>(a)+thrust::get<3>(b), thrust::get<4>(a)+thrust::get<4>(b), thrust::get<5>(a)+thrust::get<5>(b));
  }
};

struct transformNCCITPad
{
  int level;
  float av1;
  float av2;
  transformNCCITPad(float av1_, float av2_, int level_){ av1 = av1_; av2 = av2_; level = level_; }

  __host__ __device__
    thrust::tuple<float, float, float> operator()(const thrust::tuple<float, float, int>& a)
  {

    if (thrust::get<0>(a) >= 0 && thrust::get<1>(a) >= 0 && thrust::get<2>(a) % level == 0)
    {
      float s = (float)(thrust::get<0>(a)-av1);
      float stex = (float)(thrust::get<1>(a)-av2);
      return thrust::make_tuple((s)*(stex), s*s, stex*stex);
    }
    else
    {
      return thrust::make_tuple(0, 0, 0);
    }
  }
};

struct reduceNCCIT
{
  reduceNCCIT(){}

  __host__ __device__
    thrust::tuple<float, float, float> operator()(const thrust::tuple<float, float, float>& a,
    const thrust::tuple<float, float, float>& b)
  {
    return thrust::make_tuple(thrust::get<0>(a)+thrust::get<0>(b), thrust::get<1>(a)+thrust::get<1>(b),
      thrust::get<2>(a)+thrust::get<2>(b));
  }
};

void Reconstruction::prepareSliceToVolumeReg()
{

  //update other GPUs -- not done in final recon step
  unsigned int N = dev_reconstructed_[0].size.x*dev_reconstructed_[0].size.y*dev_reconstructed_[0].size.z;
  for (int d = 1; d < devicesToUse.size(); d++)
  {
    int dev = devicesToUse[d];
    checkCudaErrors(cudaMemcpy(dev_reconstructed_[dev].data, dev_reconstructed_[0].data,
      N*sizeof(float), cudaMemcpyDefault));
  }

  bool allocate = !reconstructed_arrays_init;
  if (allocate)
  {
    int storagesize = 1 + *std::max_element(devicesToUse.begin(), devicesToUse.end());
    dev_reconstructed_array.resize(storagesize);
    reconstructed_arrays_init = true;
  }

  if (multiThreadedGPU)
  {
    for (int d = 0; d < devicesToUse.size(); d++)
      GPU_workers[d]->preparePrepareSliceToVolumeReg(allocate);
    GPU_sync.runNextRound();
  }
  else
  {
    for (int d = 0; d < devicesToUse.size(); d++)
    {
      int dev = devicesToUse[d];
      prepareSliceToVolumeRegOnX(allocate, dev);
    }
  }


  checkCudaErrors(cudaSetDevice(0));

#if 0
  ////////////////////////singe GPU
  Volume<float> recon_float;
  recon_float.init(dev_reconstructed_[0].size, dev_reconstructed_[0].dim);
  //printf("initVolume/n");
  dim3 blockSize3 = dim3(8, 8, 8);
  dim3 gridSize3 = divup(dim3(dev_reconstructed_[0].size.x, dev_reconstructed_[0].size.y, dev_reconstructed_[0].size.z), blockSize3);
  castToFloat << <gridSize3, blockSize3, 0, streams[0] >> >(dev_reconstructed_[0], recon_float);
  CHECK_ERROR(castToFloat);
  //printf("castToFloat/n");
  regSlices.init(v_slices_resampled_float.size, v_slices_resampled_float.dim);
  unsigned int N = regSlices.size.x*regSlices.size.y*regSlices.size.z;
  initMem<float>(regSlices.data, N, -1.0);
  CHECK_ERROR(initMem);
  //printf("initMem/n");

  //only float is possible for interpolated texture access
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

  cudaExtent asize;
  asize.width = recon_float.size.x;
  asize.height = recon_float.size.y;
  asize.depth = recon_float.size.z;
  //if not already alloced
  if (reconstructed_array == NULL) checkCudaErrors(cudaMalloc3DArray(&reconstructed_array, &channelDesc, asize));

  cudaMemcpy3DParms copyParams = { 0 };
  copyParams.srcPtr = make_cudaPitchedPtr((void*)recon_float.data, recon_float.size.x*sizeof(float),
    recon_float.size.x, recon_float.size.y);
  copyParams.dstArray = reconstructed_array;
  copyParams.extent = asize;
  copyParams.kind = cudaMemcpyDeviceToDevice;
  checkCudaErrors(cudaMemcpy3D(&copyParams));

  reconstructedTex_.addressMode[0] = cudaAddressModeBorder;
  reconstructedTex_.addressMode[1] = cudaAddressModeBorder;
  reconstructedTex_.addressMode[2] = cudaAddressModeBorder;
  reconstructedTex_.filterMode = cudaFilterModeLinear;
  reconstructedTex_.normalized = true;

  checkCudaErrors(cudaBindTextureToArray(reconstructedTex_, reconstructed_array, channelDesc));
  CHECK_ERROR(cudaBindTextureToArray);

  recon_float.release();
#endif

  _NumberOfLevels = 2;
  _NumberOfSteps = 4;
  _NumberOfIterations = 20;
  _Epsilon = 0.0001f;

  checkGPUMemory();

  //TODO should get deleted somewhere
  _Blurring = new float[_NumberOfLevels];
  _LengthOfSteps = new float[_NumberOfLevels];
  _Blurring[0] = (dev_reconstructed_[0].dim.x) / 2.0f;
  for (int i = 0; i < _NumberOfLevels; i++) {
    _LengthOfSteps[i] = 0.1 * pow(2.0f, i);
  }
  for (int i = 1; i < _NumberOfLevels; i++) {
    _Blurring[i] = _Blurring[i - 1] * 2;
  }

  /*for(int s = 0; s < 8; s++)
  {
  cudaStreamCreate ( &streams[s]) ;
  }*/
}


void Reconstruction::prepareSliceToVolumeRegOnX(bool alloc, int dev)
{
  checkCudaErrors(cudaSetDevice(dev));
  printf("prepareSliceToVolumeReg %d\n", dev);

  int start = dev_slice_range_offset[dev].x;
  int end = dev_slice_range_offset[dev].y;

  /*Volume<float> recon_float;
  recon_float.init(dev_reconstructed_[dev].size, dev_reconstructed_[dev].dim);
  dim3 blockSize3 = dim3(8,8,8);
  dim3 gridSize3 = divup(dim3(dev_reconstructed_[dev].size.x, dev_reconstructed_[dev].size.y, dev_reconstructed_[dev].size.z), blockSize3);
  castToFloat<<<gridSize3,blockSize3, 0, streams[dev]>>>(dev_reconstructed_[dev], recon_float);
  CHECK_ERROR(castToFloat);*/

  unsigned int N1 = dev_regSlices[dev].size.x*dev_regSlices[dev].size.y*dev_regSlices[dev].size.z;
  //initMem<float>(dev_regSlices[dev].data,N1,-1.0);
  //CHECK_ERROR(initMem);

  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

  cudaExtent asize;
  asize.width = dev_reconstructed_[dev].size.x;
  asize.height = dev_reconstructed_[dev].size.y;
  asize.depth = dev_reconstructed_[dev].size.z;

  if (alloc)
  {
    cudaArray* a1;
    checkCudaErrors(cudaMalloc3DArray(&a1, &channelDesc, asize));
    dev_reconstructed_array[dev] = a1;
  }

  cudaMemcpy3DParms copyParams = { 0 };
  copyParams.srcPtr = make_cudaPitchedPtr((void*)dev_reconstructed_[dev].data, dev_reconstructed_[dev].size.x*sizeof(float),
    dev_reconstructed_[dev].size.x, dev_reconstructed_[dev].size.y);
  copyParams.dstArray = dev_reconstructed_array[dev];
  copyParams.extent = asize;
  copyParams.kind = cudaMemcpyDeviceToDevice;
  checkCudaErrors(cudaMemcpy3D(&copyParams));

  reconstructedTex_.addressMode[0] = cudaAddressModeBorder;
  reconstructedTex_.addressMode[1] = cudaAddressModeBorder;
  reconstructedTex_.addressMode[2] = cudaAddressModeBorder;
  reconstructedTex_.filterMode = cudaFilterModeLinear;
  reconstructedTex_.normalized = true;

  checkCudaErrors(cudaBindTextureToArray(reconstructedTex_, dev_reconstructed_array[dev], channelDesc));
  CHECK_ERROR(cudaBindTextureToArray);
  //	recon_float.release();
}




__global__ void initActiveSlices(int* buffer, int num);
__global__ void adjustSamplingMatrixForCentralDifferences(const Matrix4* in, Matrix4* __restrict out, int* activeMask, int activeSlices, int slices, int part, float step);
__global__ void computeGradientCentralDiff(const float* similarities, float* gradient, int* activeMask, int activeSlices, int slices, int p);
__global__ void normalizeGradient(float* gradient, int* activeMask, int activeSlices, int slices);
__global__ void copySimilarity(float* similarities, int active_slices, int slices, int* activeMask, int target, int source);
__global__ void gradientStep(Matrix4* matrices, const float* gradient, int activeSlices, int slices, int* activeMask, float step);
__global__ void checkImprovement(int* newActiveMask, int activeSlices, int slices, const int* activeMask, float* similarities, int cursim, int prev, float eps);

__device__ int dev_active_slice_count = 0;


template<typename T>
void printBuffer(T* buffer, int size, const char* printer = "%f ")
{
  std::vector<T> t(size);
  cudaMemcpy(&t[0], buffer, sizeof(float)*size, cudaMemcpyDeviceToHost);
  for (int i = 0; i < size; ++i)
    printf(printer, t[i]);
  printf("\n");
}


void printSim(float* sim, int* active, int dev_slices, int active_slices)
{
  std::vector<float> hsim(3 * dev_slices);
  std::vector<int> hactive(active_slices);
  cudaMemcpy(&hsim[0], sim, sizeof(float) * 3 * dev_slices, cudaMemcpyDeviceToHost);
  cudaMemcpy(&hactive[0], active, sizeof(int)*active_slices, cudaMemcpyDeviceToHost);

  for (int i = 0; i < dev_slices; ++i)
  {
    int a = std::find(hactive.begin(), hactive.end(), i) != hactive.end();
    printf("slice %d(%d): %f %f %f\n", i, a, hsim[i], hsim[dev_slices + i], hsim[2 * dev_slices + i]);
  }

}

void Reconstruction::registerMultipleSlicesToVolume(std::vector<Matrix4>& transf_, int global_slice, int dev_slices, int dev)
{
  cudaSetDevice(dev);

  // cpy transform matrices
  checkCudaErrors(cudaMemcpy(dev_recon_matrices[dev], &transf_[global_slice], sizeof(Matrix4)*dev_slices, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dev_recon_matrices_orig[dev], &transf_[global_slice], sizeof(Matrix4)*dev_slices, cudaMemcpyHostToDevice));

  // all levels .. whatever they are :P
  for (int level = _NumberOfLevels - 1; level >= 0; --level)
  {
    // settings for level
    float _TargetBlurring = _Blurring[level];
    float _StepSize = _LengthOfSteps[level];

    // copy all slices
    unsigned int Ns = dev_regSlices[dev].size.x*dev_regSlices[dev].size.y;

    //thrust::device_ptr<float> d_sd(dev_v_slices_resampled[dev].data);
    //thrust::device_ptr<float> d_sf(dev_v_slices_resampled_float[dev].data);
    //thrust::copy(d_sd,d_sd+Ns*dev_slices,d_sf);
    dev_v_slices_resampled_float[dev].copyFromOther(dev_v_slices_resampled[dev]);

    FilterGaussStack(dev_v_slices_resampled_float[dev].surface, dev_v_slices_resampled_float[dev].surface,
      dev_temp_slices[dev].surface, dev_regSlices[dev].size.x,
      dev_regSlices[dev].size.y, dev_slices, _TargetBlurring);

    //dev_v_slices_resampled_float[dev].print(make_uint3(0,0,0), make_uint3(0,0,1), -1);

    //dev_v_slices_resampled[dev].print(make_uint3(0,0,0), make_uint3(0,0,1), -1);

    //checkCudaErrors(cudaDeviceSynchronize());

    //dev_v_slices_resampled_float[dev].print(make_uint3(69,74,0), make_uint3(10,10,5));

    for (int st = 0; st < _NumberOfSteps; st++)
    {
      //active all slices
      initActiveSlices << <divup(dev_slices, 512), 512 >> >(dev_active_slices[dev], dev_slices);
      int activeSlices = dev_slices;
      //checkCudaErrors(cudaDeviceSynchronize());

      //printf("%d: ", dev_slices);
      //printBuffer(dev_active_slices[dev], dev_slices, "%d "); 
      //printf("\n");

      for (int iter = 0; iter < _NumberOfIterations; iter++)
      {
        evaluateCostsMultipleSlices(activeSlices, dev_slices, level, _TargetBlurring, 0, 1, 3, dev);
        //checkCudaErrors(cudaDeviceSynchronize());

        //printf("%d/%d %d/%d + %d/%d (%d active)\n",_NumberOfLevels-level, _NumberOfLevels, st, _NumberOfSteps, iter, _NumberOfIterations, activeSlices);
        for (int p = 0; p < 6; ++p)
        {
          // adjust sampling matrix +stepsize @ p
          adjustSamplingMatrixForCentralDifferences << <divup(activeSlices, 512), 512 >> >(dev_recon_matrices_orig[dev], dev_recon_matrices[dev], dev_active_slices[dev], activeSlices, dev_slices, p, _StepSize);
          //checkCudaErrors(cudaDeviceSynchronize());
          // run evaluate costs
          evaluateCostsMultipleSlices(activeSlices, dev_slices, level, _TargetBlurring, 3, 0, 1, dev);

          // adjust sampling matrix -stepsize @ p
          adjustSamplingMatrixForCentralDifferences << <divup(activeSlices, 512), 512 >> >(dev_recon_matrices_orig[dev], dev_recon_matrices[dev], dev_active_slices[dev], activeSlices, dev_slices, p, -_StepSize);
          //checkCudaErrors(cudaDeviceSynchronize());
          // run evaluate costs
          evaluateCostsMultipleSlices(activeSlices, dev_slices, level, _TargetBlurring, 4, 0, 1, dev);

          // write gradient part from central diff and add to norm
          computeGradientCentralDiff << <divup(activeSlices, 512), 512 >> >(dev_recon_similarities[dev] + 3 * dev_slices, dev_recon_gradient[dev], dev_active_slices[dev], activeSlices, dev_slices, p);
          //checkCudaErrors(cudaDeviceSynchronize());
        }

        //checkCudaErrors(cudaDeviceSynchronize());

        // divide gradient by norm
        //printf("%llx %llx %d %d\n", dev_recon_gradient[dev], dev_active_slices[dev], activeSlices, dev_slices);
        normalizeGradient << <divup(activeSlices, 512), 512 >> >(dev_recon_gradient[dev], dev_active_slices[dev], activeSlices, dev_slices);
        //checkCudaErrors(cudaDeviceSynchronize());
        //printBuffer(dev_recon_gradient[dev], activeSlices*7, "%f \n");
        //for(int i = 0; i < 6; ++i)
        //  printBuffer(dev_recon_gradient[dev]+i*dev_slices, 1, "%f ");
        //printf("\n");

        int prevActiveSlices = activeSlices;
        checkCudaErrors(cudaMemcpy(dev_active_slices_prev[dev], dev_active_slices[dev], activeSlices*sizeof(int), cudaMemcpyDeviceToDevice));

        //simple gradient decent
        do {
          //checkCudaErrors(cudaDeviceSynchronize());

          //new similarity = similarity;
          copySimilarity << <divup(activeSlices, 512), 512 >> >(dev_recon_similarities[dev], activeSlices, dev_slices, dev_active_slices[dev], 2, 0);
          //checkCudaErrors(cudaDeviceSynchronize());
          // step along gradient and generate matrix
          gradientStep << <divup(activeSlices, 512), 512 >> >(dev_recon_matrices[dev], dev_recon_gradient[dev], activeSlices, dev_slices, dev_active_slices[dev], _StepSize);
          //checkCudaErrors(cudaDeviceSynchronize());
          // run evaluate cost for similarity
          evaluateCostsMultipleSlices(activeSlices, dev_slices, level, _TargetBlurring, 0, 1, 1, dev);

          // check how many are still improving and compact (similarity > new_similarity + _Epsilon)
          int h_active_slice_count = 0;
          checkCudaErrors(cudaMemcpyToSymbol(dev_active_slice_count, &h_active_slice_count, sizeof(int)));

          checkImprovement << <divup(activeSlices, 512), 512, 512 * sizeof(int) >> >(dev_active_slices2[dev], activeSlices, dev_slices, dev_active_slices[dev], dev_recon_similarities[dev], 0, 2, _Epsilon);
          std::swap(dev_active_slices[dev], dev_active_slices2[dev]);
          checkCudaErrors(cudaDeviceSynchronize());
          //printSim(dev_recon_similarities[dev], dev_active_slices[dev], dev_slices, activeSlices);

          // copy number of improving
          checkCudaErrors(cudaMemcpyFromSymbol(&activeSlices, dev_active_slice_count, sizeof(int)));

          //printf("improving: %d/%d\n", activeSlices, dev_slices);
          //printSim(dev_recon_similarities[dev], dev_active_slices[dev], dev_slices, activeSlices);
          //printf("\n");
          // end if all done
        } while (activeSlices > 0);


        // last step was no improvement, so back track
        gradientStep << <divup(prevActiveSlices, 512), 512 >> >(dev_recon_matrices[dev], dev_recon_gradient[dev], prevActiveSlices, dev_slices, dev_active_slices_prev[dev], -_StepSize);
        //checkCudaErrors(cudaDeviceSynchronize());
        // copy matrices
        checkCudaErrors(cudaMemcpy(dev_recon_matrices_orig[dev], dev_recon_matrices[dev], sizeof(Matrix4)*dev_slices, cudaMemcpyDeviceToDevice));

        // check for overall improvement and compact
        int h_active_slice_count = 0;
        checkCudaErrors(cudaMemcpyToSymbol(dev_active_slice_count, &h_active_slice_count, sizeof(int)));
        checkImprovement << <divup(prevActiveSlices, 512), 512, 512 * sizeof(int) >> >(dev_active_slices[dev], prevActiveSlices, dev_slices, dev_active_slices_prev[dev], dev_recon_similarities[dev], 2, 1, _Epsilon);
        checkCudaErrors(cudaMemcpyFromSymbol(&activeSlices, dev_active_slice_count, sizeof(int)));

        //printf("still active: %d/%d\n", activeSlices, dev_slices);
        //printSim(dev_recon_similarities[dev], dev_active_slices[dev], dev_slices, activeSlices);
        if (activeSlices == 0)
          break;
      }
      _StepSize /= 2.0f;
    }
  }
  //copy matrices to host
  checkCudaErrors(cudaMemcpy(&transf_[global_slice], dev_recon_matrices[dev], sizeof(Matrix4)*dev_slices, cudaMemcpyDeviceToHost));
  CHECK_ERROR(registerMultipleSlicesToVolume);
}


__global__ void averageIf(cudaSurfaceObject_t layers, int* activelayers, float* sum, int* count, int width, int height);
__global__ void computeNCCAndReduce(cudaSurfaceObject_t layersA, int* activelayersA, cudaSurfaceObject_t layersB, const float* sums, const int *counts,
  float* results, int width, int height, int slices, int level);
__global__ void addNccValues(const float* prevData, float* result, int slices);
__global__ void writeSimilarities(const float* nvccResults, int* activelayers, int writestep, int writenum, float* similarities, int active_slices, int slices);

void Reconstruction::evaluateCostsMultipleSlices(int active_slices, int slices, int level, float targetBlurring, int writeoffset, int writestep, int writenum, int dev)
{
  if (active_slices == 0)
    return;
  dim3 redblock(32, 32);
  dim3 slicesdim(dev_v_slices_resampled_float[dev].size.x, dev_v_slices_resampled_float[dev].size.y, active_slices);
  dim3 redgrid = divup(slicesdim, dim3(redblock.x * 2, redblock.y * 2));

  // init temp vars to zero
  checkCudaErrors(cudaMemset(dev_temp_float[dev], 0, 6 * active_slices*sizeof(float)));
  checkCudaErrors(cudaMemset(dev_temp_int[dev], 0, 2 * active_slices*sizeof(int)));

  // compute av_s = dev_temp_float[dev] / dev_temp_int[dev]
  // compute average of each slice
  averageIf << <redgrid, redblock, redblock.x*redblock.y*(sizeof(float) + sizeof(int)) >> >(dev_v_slices_resampled_float[dev].surface, dev_active_slices[dev], dev_temp_float[dev], dev_temp_int[dev], dev_v_slices_resampled_float[dev].size.x, dev_v_slices_resampled_float[dev].size.y);

  //checkCudaErrors(cudaDeviceSynchronize());
  //float ff[128];
  //int tf[128];
  //cudaMemcpy(ff, dev_temp_float[dev], min(128,slices)*sizeof(float), cudaMemcpyDeviceToHost);
  //cudaMemcpy(tf, dev_temp_int[dev], min(128,slices)*sizeof(int), cudaMemcpyDeviceToHost);
  //for(int i = 0; i < min(128,slices); ++i)
  //  printf("%f \n",ff[i]/tf[i]);
  //printf("\n\n");


  for (int insofs = -1; insofs <= 1; insofs++)
  {
    dim3 blockSize3 = dim3(16, 16, 1);
    dim3 gridSize3 = divup(dim3(dev_regSlices[dev].size.x, dev_regSlices[dev].size.y, active_slices), blockSize3);
    float3 fsize = make_float3(dev_reconstructed_[dev].size.x, dev_reconstructed_[dev].size.y, dev_reconstructed_[dev].size.z);

    //checkCudaErrors(cudaDeviceSynchronize());

    // generate slices
    //printf("%d %d: ",gridSize3.z, active_slices);
    //printBuffer(dev_active_slices[dev], gridSize3.z, "%d "); 
    //printf("\n");
    genenerateRegistrationSlices << <gridSize3, blockSize3 >> >(dev_regSlices[dev].surface, dev_active_slices[dev], dev_d_slicesResampledI2W[dev], dev_recon_matrices[dev], dev_d_slicesOfs[dev],
      fsize, make_uint2(regSlices.size.x, regSlices.size.y), insofs);
    //checkCudaErrors(cudaDeviceSynchronize());
    // filter gauss
    FilterGaussStack(dev_regSlices[dev].surface, dev_regSlices[dev].surface, dev_temp_slices[dev].surface, dev_regSlices[dev].size.x, dev_regSlices[dev].size.y, active_slices, targetBlurring);

    // compute av_stex  = dev_temp_float[dev] +slices  / dev_temp_int[dev] + slices
    averageIf << <redgrid, redblock, redblock.x*redblock.y*(sizeof(float) + sizeof(int)) >> >(dev_regSlices[dev].surface, 0, dev_temp_float[dev] + active_slices, dev_temp_int[dev] + active_slices, dev_regSlices[dev].size.x, dev_regSlices[dev].size.y);
    //checkCudaErrors(cudaDeviceSynchronize());


    // set ncc temp values to zero
    checkCudaErrors(cudaMemset(dev_temp_float[dev] + 2 * slices, 0, 3 * slices*sizeof(float)));

    // compute ncc 
    computeNCCAndReduce << <redgrid, redblock, redblock.x*redblock.y*sizeof(float) * 3 >> >(dev_v_slices_resampled_float[dev].surface, dev_active_slices[dev], dev_regSlices[dev].surface, dev_temp_float[dev], dev_temp_int[dev],
      dev_temp_float[dev] + 3 * active_slices, dev_regSlices[dev].size.x, dev_regSlices[dev].size.y, active_slices, level + 1);
    //checkCudaErrors(cudaDeviceSynchronize());
    // sum up result

    addNccValues << <divup(active_slices, 512), 512 >> >(dev_temp_float[dev] + 3 * active_slices, dev_temp_float[dev] + 2 * active_slices, active_slices);
    //checkCudaErrors(cudaDeviceSynchronize());

  }

  writeSimilarities << <divup(active_slices, 512), 512 >> >(dev_temp_float[dev] + 2 * active_slices, dev_active_slices[dev], writestep, writenum, dev_recon_similarities[dev] + writeoffset*slices, active_slices, slices);
  //checkCudaErrors(cudaDeviceSynchronize());
  //std::vector<float> sim(3*slices);
  //cudaMemcpy(&sim[0],  dev_recon_similarities[dev], sizeof(float)*3*slices, cudaMemcpyDeviceToHost);
  //for(int i = 0; i < 3*slices; ++i)
  //  printf("%f ", sim[i]);
  //printf("\n");

}


__global__ void initActiveSlices(int* buffer, int num)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < num)
    buffer[i] = i;
}

__global__ void adjustSamplingMatrixForCentralDifferences(const Matrix4* inmatrices, Matrix4* __restrict outmatrices, int* activeMask, int activeSlices, int slices, int part, float step)
{
  const float pi = 3.14159265358979323846f;
  //  const float One80dvPi =  57.2957795131f;

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= activeSlices)
    return;

  int slice = activeMask[i];

  Matrix4& out = outmatrices[slice];
  const Matrix4& in = inmatrices[slice];
  out = in;

  if (part < 3)
  {
    // translation is easy
    out.data[part].w = in.data[part].w + step;
  }
  else
  {
    // rotation is more complex
    float TOL = 0.000001f;
    float tmp = asinf(-1.0f * in.data[0].z);

    float p_rot[3];

    if (fabsf(cos(tmp)) > TOL) {
      p_rot[0] = atan2f(in.data[1].z, in.data[2].z);
      p_rot[1] = tmp;
      p_rot[2] = atan2f(in.data[0].y, in.data[0].x);
    }
    else {
      //m(0,2) is close to +1 or -1
      p_rot[0] = atan2f(-in.data[0].z*in.data[1].x, -in.data[0].z*in.data[2].x);
      p_rot[1] = tmp;
      p_rot[2] = 0;
    }

    // parameter space is in deg so convert to rad
    p_rot[part - 3] += step*pi / 180.0f;

    float cosrx = cos(p_rot[0]);
    float cosry = cos(p_rot[1]);
    float cosrz = cos(p_rot[2]);
    float sinrx = sin(p_rot[0]);
    float sinry = sin(p_rot[1]);
    float sinrz = sin(p_rot[2]);

    out.data[0].x = cosry*cosrz;
    out.data[0].y = cosry*sinrz;
    out.data[0].z = -sinry;

    out.data[1].x = (sinrx*sinry*cosrz - cosrx*sinrz);
    out.data[1].y = (sinrx*sinry*sinrz + cosrx*cosrz);
    out.data[1].z = sinrx*cosry;

    out.data[2].x = (cosrx*sinry*cosrz + sinrx*sinrz);
    out.data[2].y = (cosrx*sinry*sinrz - sinrx*cosrz);
    out.data[2].z = cosrx*cosry;
  }
}

__global__ void computeGradientCentralDiff(const float* similarities, float* gradient, int* activeMask, int activeSlices, int slices, int p)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= activeSlices)
    return;
  int slice = activeMask[i];

  float dx = similarities[slice] - similarities[slices + slice];
  gradient[p*slices + slice] = dx;
  if (p == 0)
    gradient[6 * slices + slice] = dx*dx;
  else
    gradient[6 * slices + slice] += dx*dx;
}
__global__ void normalizeGradient(float* gradient, int* activeMask, int activeSlices, int slices)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= activeSlices)
    return;

  int slice = activeMask[i];

  float norm = gradient[6 * slices + slice];
  if (norm > 0)
    norm = 1.0f / sqrtf(norm);

  for (int j = 0; j < 6; ++j)
    gradient[j*slices + slice] *= norm;
}


__global__ void copySimilarity(float* similarities, int active_slices, int slices, int* activeMask, int target, int source)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= active_slices)
    return;
  int slice = activeMask[i];
  similarities[target*slices + slice] = similarities[source*slices + slice];
}

__global__ void gradientStep(Matrix4* matrices, const float* gradient, int active_slices, int slices, int* activeMask, float step)
{
  const float pi = 3.14159265358979323846f;

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= active_slices)
    return;
  int slice = activeMask[i];

  Matrix4& matrix = matrices[slice];

  // translation is easy
  for (int p = 0; p < 3; ++p)
    matrix.data[p].w = matrix.data[p].w + step*gradient[p*slices + slice];


  // rotation is more complex
  float TOL = 0.000001f;
  float tmp = asinf(-1.0f * matrix.data[0].z);

  float p_rot[3];

  if (fabsf(cos(tmp)) > TOL) {
    p_rot[0] = atan2f(matrix.data[1].z, matrix.data[2].z);
    p_rot[1] = tmp;
    p_rot[2] = atan2f(matrix.data[0].y, matrix.data[0].x);
  }
  else {
    //m(0,2) is close to +1 or -1
    p_rot[0] = atan2f(-matrix.data[0].z*matrix.data[1].x, -matrix.data[0].z*matrix.data[2].x);
    p_rot[1] = tmp;
    p_rot[2] = 0;
  }

  // parameter space is in deg so convert to rad
  for (int p = 0; p < 3; ++p)
    p_rot[p] += gradient[(p + 3)*slices + slice] * step*pi / 180.0f;

  float cosrx = cos(p_rot[0]);
  float cosry = cos(p_rot[1]);
  float cosrz = cos(p_rot[2]);
  float sinrx = sin(p_rot[0]);
  float sinry = sin(p_rot[1]);
  float sinrz = sin(p_rot[2]);

  matrix.data[0].x = cosry*cosrz;
  matrix.data[0].y = cosry*sinrz;
  matrix.data[0].z = -sinry;

  matrix.data[1].x = (sinrx*sinry*cosrz - cosrx*sinrz);
  matrix.data[1].y = (sinrx*sinry*sinrz + cosrx*cosrz);
  matrix.data[1].z = sinrx*cosry;

  matrix.data[2].x = (cosrx*sinry*cosrz + sinrx*sinrz);
  matrix.data[2].y = (cosrx*sinry*sinrz - sinrx*cosrz);
  matrix.data[2].z = cosrx*cosry;
}

__global__ void checkImprovement(int* newActiveMask, int activeSlices, int slices, const int* activeMask, float* similarities, int cursim, int prev, float eps)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int active = 0;
  int slice = -1;
  if (tid < activeSlices)
  {
    slice = activeMask[tid];
    // check for improvement > eps
    active = similarities[cursim*slices + slice] > similarities[prev*slices + slice] + eps ? 1 : 0;
  }

  // run prefix sum
  extern __shared__ int prefixSumSpace[];
  prefixSumSpace[tid] = active;
  int offset = 1;
  int n = blockDim.x;
  int overallcount = 0;

  for (int d = n >> 1; d > 0; d /= 2)
  {
    __syncthreads();
    if (tid < d)
    {
      int ai = offset*(2 * tid + 1) - 1;
      int bi = offset*(2 * tid + 2) - 1;
      prefixSumSpace[bi] += prefixSumSpace[ai];
    }
    offset *= 2;
  }

  if (tid == 0)
  {
    overallcount = prefixSumSpace[n - 1];
    prefixSumSpace[n - 1] = 0;
  }

  for (int d = 1; d < n; d *= 2)
  {
    offset /= 2;
    __syncthreads();
    if (tid < d)
    {
      int ai = offset*(2 * tid + 1) - 1;
      int bi = offset*(2 * tid + 2) - 1;
      float t = prefixSumSpace[ai];
      prefixSumSpace[ai] = prefixSumSpace[bi];
      prefixSumSpace[bi] += t;
    }
  }

  // get global array offset
  __shared__ int globalOffset;
  if (tid == 0)
  {
    globalOffset = atomicAdd(&dev_active_slice_count, overallcount);
  }
  __syncthreads();

  if (active)
  {
    newActiveMask[globalOffset + prefixSumSpace[tid]] = slice;
  }
}

__global__ void averageIf(cudaSurfaceObject_t layers, int* activelayers, float* sum, int* count, int width, int height)
{
  extern __shared__ int reductionSpace[];
  int localid = threadIdx.x + blockDim.x*threadIdx.y;
  int slice = blockIdx.z;
  if (activelayers != 0)
    slice = activelayers[blockIdx.z];
  int threads = blockDim.x*blockDim.y;

  int myActive = 0;
  float myCount = 0;
  for (int y = blockIdx.y*blockDim.y + threadIdx.y; y < height; y += blockDim.y*gridDim.y)
    for (int x = blockIdx.x*blockDim.x + threadIdx.x; x < width; x += blockDim.x*gridDim.x)
    {
    float val = surf2DLayeredread<float>(layers, x * 4, y, slice, cudaBoundaryModeClamp);
    if (val > -1.0f)
      ++myActive, myCount += val;
    }

  float* f_reduction = reinterpret_cast<float*>(reductionSpace + threads);
  reductionSpace[localid] = myActive;
  f_reduction[localid] = myCount;
  __syncthreads();

  for (int n = threads / 2; n > 1; n /= 2)
  {
    if (localid < n)
      reductionSpace[localid] += reductionSpace[localid + n],
      f_reduction[localid] += f_reduction[localid + n];
    __syncthreads();
  }

  if (threadIdx.x == 0)
  {
    atomicAdd(sum + blockIdx.z, f_reduction[0] + f_reduction[1]),
      atomicAdd(count + blockIdx.z, reductionSpace[0] + reductionSpace[1]);
  }
}

__global__ void computeNCCAndReduce(cudaSurfaceObject_t layersA, int* activelayersA, cudaSurfaceObject_t layersB, const float* sums, const int *counts,
  float* results, int width, int height, int slices, int level)
{
  int layerA = activelayersA[blockIdx.z];
  int layerB = blockIdx.z;
  int threads = blockDim.x*blockDim.y;

  float avg_a = sums[layerB];
  float avg_b = sums[slices + layerB];
  if (avg_a != 0)   avg_a /= counts[layerB];
  if (avg_b != 0)   avg_b /= counts[slices + layerB];

  float3 values = make_float3(0, 0, 0);

  for (int y = blockIdx.y*blockDim.y + threadIdx.y; y < height; y += blockDim.y*gridDim.y)
    for (int x = blockIdx.x*blockDim.x + threadIdx.x; x < width; x += blockDim.x*gridDim.x)
    {
    float a = surf2DLayeredread<float>(layersA, x * 4, y, layerA, cudaBoundaryModeClamp);
    float b = surf2DLayeredread<float>(layersB, x * 4, y, layerB, cudaBoundaryModeClamp);
    int lin = y * width + x;
    if (a >= 0.0f && b >= 0.0f && lin % level == 0)
    {
      float s = a - avg_a;
      float stex = b - avg_b;
      values = values + make_float3(s*stex, s*s, stex*stex);
    }
    }

  extern __shared__ float reduction[];
  int localid = threadIdx.x + blockDim.x*threadIdx.y;
  reduction[localid] = values.x;
  reduction[localid + threads] = values.y;
  reduction[localid + 2 * threads] = values.z;

  __syncthreads();

  for (int n = threads / 2; n > 1; n /= 2)
  {
    if (localid < n)
      reduction[localid] = reduction[localid] + reduction[localid + n],
      reduction[localid + threads] = reduction[localid + threads] + reduction[localid + threads + n],
      reduction[localid + 2 * threads] = reduction[localid + 2 * threads] + reduction[localid + 2 * threads + n];
    __syncthreads();
  }

  //write results
  if (threadIdx.x == 0)
  {
    atomicAdd(results + blockIdx.z * 3 + 0, reduction[0] + reduction[1]);
    atomicAdd(results + blockIdx.z * 3 + 1, reduction[threads] + reduction[threads + 1]);
    atomicAdd(results + blockIdx.z * 3 + 2, reduction[2 * threads] + reduction[2 * threads + 1]);
  }
}

__global__ void addNccValues(const float* prevData, float* result, int slices)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < slices)
  {
    float norm = prevData[3 * tid + 1] * prevData[3 * tid + 2];
    float res = 0;
    if (norm > 0)
      res = prevData[3 * tid] / sqrtf(norm);
    result[tid] += res;
  }
}

__global__ void writeSimilarities(const float* nvccResults, int* activelayers, int writestep, int writenum, float* similarities, int active_slices, int slices)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < active_slices)
  {
    float res = nvccResults[tid];
    int slice = activelayers[tid];
    for (int i = 0; i < writenum; ++i)
      similarities[slices*writestep*i + slice] = res;
  }
}

//void Reconstruction::registerSliceToVolume(Matrix4& transf_, int global_slice, int dev_slice, short dev)
//{
//  checkCudaErrors(cudaSetDevice(dev));
//  Matrix4 h_trans;
//  h_trans = transf_;
//  uint2 vSize = make_uint2(dev_v_slices_resampled_float[dev].size.x, dev_v_slices_resampled_float[dev].size.y);
//
//  for (int level = _NumberOfLevels-1; level >= 0; level--) 
//  {
//
//    float _TargetBlurring = _Blurring[level];
//   
//    unsigned int Ns = dev_regSlices[dev].size.x*dev_regSlices[dev].size.y;
//
//    //reset from blurring
//    thrust::device_ptr<float> d_sd(dev_v_slices_resampled[dev].data + dev_slice*Ns);
//    thrust::device_ptr<float> d_sf(dev_v_slices_resampled_float[dev].data + dev_slice*Ns);
//    thrust::copy(d_sd,d_sd+Ns,d_sf);
//
//    float _StepSize = _LengthOfSteps[level];//2 * pow(2.0, level);
//    //std::cout << "Resolution level no. " << level+1 << " (step sizes ";
//    //std::cout << _StepSize << " to " << _StepSize / pow(2.0, static_cast<float>(_NumberOfSteps-1)) << ")\n";
//
//    FilterGauss(dev_v_slices_resampled_float[dev].data+dev_slice*Ns, dev_v_slices_resampled_float[dev].data+dev_slice*Ns, dev_regSlices[dev].size.x, 
//      dev_regSlices[dev].size.y, dev_regSlices[dev].size.x, 1, _TargetBlurring );
//
//    //printf("_NumberOfLevels %d, targetBlurring: %f, _NumberOfSteps %d, _NumberOfIterations %d, \n", _NumberOfLevels, _TargetBlurring, _NumberOfSteps, _NumberOfIterations);
//
//    for (int st = 0; st < _NumberOfSteps; st++) 
//    {
//      for(int iter = 0; iter < _NumberOfIterations; iter++)
//      {
//        float similarity, new_similarity, old_similarity;
//        old_similarity = new_similarity = similarity = evaluateCosts(dev_slice, h_trans, level, _TargetBlurring, dev);
//        //printf("evaluateCosts %d %d/n", iter, st);
//        float params[6];
//        float dx[6];
//        Matrix2Parameters(h_trans, params);
//        //printf("Matrix2Parameters %d %d/n", iter, st);
//        //printf("old vs. new: %f  ", old_similarity);
//        //printf("iter: %d slice: %d ncc %f  size: %d %d %d\n", iter, slice, similarity, regSlices.size.x, regSlices.size.y, regSlices.size.z);
//
//        //printf("tx %f ty %f tz %f rx %f ry %f rz %f \n", params[TX], params[TY], params[TZ], params[RX], params[RY], params[RZ]);
//        //for (int j = 0; j < 6; j++) {  x[j] = (float)params[j]; }
//
//        for(int p = 0; p < 6; p++)
//        {
//          float pValue = params[p];
//          params[p] = pValue+_StepSize;
//          Matrix4 p1 = Parameters2Matrix(params);
//          //float s1 = evaluateCostsWithPadding(slice, p1);
//          float s1 = evaluateCosts(dev_slice, p1, level, _TargetBlurring, dev);
//          params[p] = pValue-_StepSize;
//          Matrix4 p2 = Parameters2Matrix(params);
//          //float s2 = evaluateCostsWithPadding(slice, p2);
//          float s2 = evaluateCosts(dev_slice, p2, level, _TargetBlurring, dev);
//          params[p] = pValue;
//          dx[p] = s1-s2;//(s1 > s2) ? s2-s1 : s1-s2;//s1-s2;//
//        }
//        //printf("Gradient %d %d/n", iter, st);
//
//        float norm = 0;
//        for (int j = 0; j < 6; j++) { norm += dx[j] * dx[j]; }
//
//        norm = sqrt(norm);
//        if (norm > 0) {
//          for (int j = 0; j < 6; j++) { dx[j] /= norm;}
//        } else {
//          for (int j = 0; j < 6; j++) {dx[j] = 0;}
//        }
//        //gradient dx ready
//        /*std::cout <<"gradient: ";
//        for(int p = 0; p < 6; p++)
//        {
//        std::cout << dx[p] << " ";
//        }
//        std::cout << std::endl;*/
//        //simple gradient decent
//        do {
//          new_similarity = similarity;
//          for (int p = 0; p < 6; p++) {
//            params[p] = params[p]+_StepSize*dx[p];
//            //	_Transformation->Put(i, _Transformation->Get(i) + _StepSize * dx[i]);
//          }
//          Matrix4 p1 = Parameters2Matrix(params);
//          //similarity = evaluateCostsWithPadding(slice, p1);
//          similarity = evaluateCosts(dev_slice, p1, level, _TargetBlurring, dev);
//          //similarity = _Registration->Evaluate();
//          /*if (similarity > new_similarity + _Epsilon) 
//          {
//          std::cout << slice << " : " << similarity << " tx = " << params[0] <<  " ty = " << params[1] << 
//          " tz = " << params[2] << " rx = " << params[3] << " ry = " << params[4] <<
//          " rz = " << params[5] << std::endl;
//          //	printf("similarity: %f old_similarity: %f \n", similarity, old_similarity);
//          //printf("%f\n", similarity);
//          }*/
//          //printf("%f\n", similarity);
//        } while (similarity > new_similarity + _Epsilon);
//        //printf("\n ");
//        //printf("Gradient Decent %d %d/n", iter, st);
//
//        // Last step was no improvement, so back track
//        for (int p = 0; p < 6; p++) {
//          params[p] = params[p]-_StepSize*dx[p];
//        }
//
//        h_trans = Parameters2Matrix(params);
//        //regUncertainty_[slice] = similarity;
//        //printf("params back %d %d/n", iter, st);
//
//        float eps = 0.0;
//        if (new_similarity > old_similarity) {
//          eps = new_similarity - old_similarity;
//        } 
//        else
//        {
//          eps = 0;
//        }
//
//        if(eps <= _Epsilon)
//          break;
//      }
//      _StepSize /= 2.0;
//    }
//  }
//
//  //cudaDeviceSynchronize();
//  transf_ = h_trans;
//}

void Reconstruction::updateResampledSlicesI2W(std::vector<Matrix4> ofsSlice)
{
  bool allocate = !dev_d_slicesOfs_allocated;

  if (allocate)
  {
    int storagesize = 1 + *std::max_element(devicesToUse.begin(), devicesToUse.end());
    dev_d_slicesOfs.resize(storagesize);
    dev_d_slicesOfs_allocated = true;
  }


  if (multiThreadedGPU)
  {
    for (int d = 0; d < devicesToUse.size(); d++)
      GPU_workers[d]->prepareUpdateResampledSlicesI2W(ofsSlice, allocate);
    GPU_sync.runNextRound();
  }
  else
  {
    for (int d = 0; d < devicesToUse.size(); d++)
    {
      int dev = devicesToUse[d];
      updateResampledSlicesI2WOnX(ofsSlice, allocate, dev);
    }
  }
  checkCudaErrors(cudaSetDevice(0));

  if (!dev_d_slicesOfs_allocated)
  {
    dev_d_slicesOfs_allocated = true;
  }

}

void Reconstruction::updateResampledSlicesI2WOnX(std::vector<Matrix4>& ofsSlice, bool allocate, int dev)
{
  checkCudaErrors(cudaSetDevice(dev));
  //printf("updateResampledSlicesI2W %d\n", dev);

  int start = dev_slice_range_offset[dev].x;
  int end = dev_slice_range_offset[dev].y;

  if (allocate)
  {
    Matrix4* t1;
    checkCudaErrors(cudaMalloc((void**)&t1, (end - start)*sizeof(Matrix4)));
    dev_d_slicesOfs[dev] = t1;
  }
  checkCudaErrors(cudaMemcpy(dev_d_slicesOfs[dev], &ofsSlice[start], (end - start)*sizeof(Matrix4), cudaMemcpyHostToDevice));
}

//better but still not perfect. TODO
void Reconstruction::registerSlicesToVolume(std::vector<Matrix4>& transf_)
{
  std::vector<Matrix4> h_trans(transf_.size());
  h_trans = transf_;

#if 0
  unsigned int dev_ = 0;
  unsigned int* dev_slice = new unsigned int[devicesToUse.size()];

  for (int d = 0; d < devicesToUse.size(); d++)
  {
    int dev = devicesToUse[d];
    dev_slice[dev] = 0;
  }

  //alternating between GPUs. this part needs also some CPU power
  //How to get the CPU computation into a second CPU thread? OMP and TBB don't work for this here
  for (int global_slice = 0; global_slice < num_slices_; global_slice++)
  {
    //alternating slice/device -- would need to alterate a CPU thread as well...
    checkCudaErrors(cudaSetDevice(dev_));
    if (dev_slice[dev_] >= dev_v_slices_resampled_float[dev_].size.z) continue;
    registerSliceToVolume(h_trans[global_slice], global_slice, dev_slice[dev_], dev_);
    dev_++;
    dev_slice[dev_]++;
    if (dev_ >= devicesToUse.size()) dev_ = 0;
  }

  float* lsrdata = v_slices_resampled.data;
  float* lrdata = regSlices.data;
  for (int d = 0; d < devicesToUse.size(); d++)
  {
    int dev = devicesToUse[d];
    checkCudaErrors(cudaSetDevice(dev));
    printf("registerSlicesToVolume %d\n", dev);

    unsigned int start = dev_slice_range_offset[dev].x;
    unsigned int end = dev_slice_range_offset[dev].y;

    unsigned int num_elems_ = (end - start)*dev_v_slices_resampled[dev].size.x*dev_v_slices_resampled[dev].size.y;
    checkCudaErrors(cudaMemcpy(lsrdata, dev_v_slices_resampled[dev].data, num_elems_*sizeof(float), cudaMemcpyDefault));
    lsrdata += num_elems_;

    checkCudaErrors(cudaMemcpy(lrdata, dev_regSlices[dev].data, num_elems_*sizeof(float), cudaMemcpyDefault));
    lrdata += num_elems_;
  }
  checkCudaErrors(cudaSetDevice(0));

  delete[] dev_slice;

#else
  ////////////////////////////////
  unsigned int global_slice = 0;

  //alternating between GPUs. this part needs also some CPU power
  //How to get the CPU computation into a second CPU thread? OMP and TBB don't work for this here

  //TODO
  // need separate threads for different GPUs otherwise memcpy and stuff will always serialize!!!!!
  if (multiThreadedGPU)
  {
    for (int d = 0; d < devicesToUse.size(); d++)
    {
      int dev = devicesToUse[d];
      int slices = dev_v_slices_resampled_float[dev].size.z;
      GPU_workers[d]->prepareRegisterSlicesToVolume1(global_slice, h_trans);
      global_slice += slices;
    }
    GPU_sync.runNextRound();
  }
  else
  {
    for (int d = 0; d < devicesToUse.size(); d++)
    {
      int dev = devicesToUse[d];
      int slices = dev_v_slices_resampled_float[dev].size.z;
      registerSlicesToVolumeOnX1(global_slice, h_trans, dev);
      global_slice += slices;

      //for(int dev_slice = 0; dev_slice < dev_v_slices_resampled_float[dev].size.z; dev_slice++)
      //{
      //  registerSliceToVolume(h_trans[global_slice], global_slice, dev_slice, dev);
      //  global_slice++;
      //}
    }
  }

  //TODO if debug GPU
  if (multiThreadedGPU)
  {
    for (int d = 0; d < devicesToUse.size(); d++)
      GPU_workers[d]->prepareRegisterSlicesToVolume2();
    GPU_sync.runNextRound();
  }
  else
  {
    for (int d = 0; d < devicesToUse.size(); d++)
    {
      int dev = devicesToUse[d];
      registerSlicesToVolumeOnX2(dev);
    }
  }
  checkCudaErrors(cudaSetDevice(0));
#endif

  transf_ = h_trans;

}

void Reconstruction::registerSlicesToVolumeOnX1(int global_slice, std::vector<Matrix4>& h_trans, int dev)
{
  checkCudaErrors(cudaSetDevice(dev));
  printf("registerSlicesToVolume %d\n", dev);

  //  uint2 vSize = make_uint2(dev_v_slices_resampled_float[dev].size.x, dev_v_slices_resampled_float[dev].size.y);

  int slices = dev_v_slices_resampled_float[dev].size.z;
  registerMultipleSlicesToVolume(h_trans, global_slice, slices, dev);
}
void Reconstruction::registerSlicesToVolumeOnX2(int dev)
{
  checkCudaErrors(cudaSetDevice(dev));
  int start = dev_slice_range_offset[dev].x;
  int end = dev_slice_range_offset[dev].y;

  float* lsrdata = v_slices_resampled.data + start * dev_v_slices_resampled[dev].size.x*dev_v_slices_resampled[dev].size.y;;
  float* lrdata = regSlices.data + start * dev_v_slices_resampled[dev].size.x*dev_v_slices_resampled[dev].size.y;;

  unsigned int num_elems_ = (end - start)*dev_v_slices_resampled[dev].size.x*dev_v_slices_resampled[dev].size.y;
  dev_v_slices_resampled[dev].copyToHost(lsrdata);
  dev_regSlices[dev].copyToHost(lrdata);
}

void Reconstruction::initRegStorageVolumes(uint3 size, float3 dim)
{
  int storagesize = 1 + *std::max_element(devicesToUse.begin(), devicesToUse.end());

  bool init = !regStorageInit;
  if (init)
  {
    dev_v_slices_resampled.resize(storagesize);
    dev_v_slices_resampled_float.resize(storagesize);
    dev_regSlices.resize(storagesize);
    dev_temp_slices.resize(storagesize);
    dev_recon_matrices.resize(storagesize);
    dev_recon_matrices_orig.resize(storagesize);
    dev_recon_params.resize(storagesize);
    dev_recon_similarities.resize(storagesize);
    dev_recon_gradient.resize(storagesize);
    dev_active_slices.resize(storagesize);
    dev_active_slices2.resize(storagesize);
    dev_active_slices_prev.resize(storagesize);
    dev_temp_float.resize(storagesize);
    dev_temp_int.resize(storagesize);
    regStorageInit = true;
  }
  if (multiThreadedGPU)
  {
    for (int d = 0; d < devicesToUse.size(); d++)
      GPU_workers[d]->prepareInitRegStorageVolumes(size, dim, init);
    GPU_sync.runNextRound();
  }
  else
  {
    for (int d = 0; d < devicesToUse.size(); d++)
    {
      int dev = devicesToUse[d];
      initRegStorageVolumesOnX(size, dim, init, dev);

      //unsigned int N1 = dsize.x*dsize.y*dsize.z;
      //checkCudaErrors(cudaMemset(dev_v_slices_resampled[dev].data, 0, N1*sizeof(float)));
    }
  }

  if (!regStorageInit)
  {
    regStorageInit = true;
  }

  regSlices.init(size, dim);
  v_slices_resampled.init(size, dim);

  unsigned int N = size.x*size.y*size.z;
  checkCudaErrors(cudaMemset(v_slices_resampled.data, 0, N*sizeof(float)));

  CHECK_ERROR(initRegStorageVolumes);
}

void Reconstruction::initRegStorageVolumesOnX(uint3 size, float3 dim, bool init, int dev)
{
  checkCudaErrors(cudaSetDevice(dev));
  printf("initRegStorageVolumes %d\n", dev);

  int start = dev_slice_range_offset[dev].x;
  int end = dev_slice_range_offset[dev].y;

  uint3 dsize = size;
  dsize.z = end - start;

  if (init)
  {
    LayeredSurface3D<float> l_slices;
    l_slices.init(dsize, dim);
    dev_v_slices_resampled[dev] = (l_slices);

    LayeredSurface3D<float> l_rfslices;
    l_rfslices.init(dsize, dim);
    dev_v_slices_resampled_float[dev] = (l_rfslices);

    LayeredSurface3D<float>l_regSlices;
    l_regSlices.init(dsize, dim);
    dev_regSlices[dev] = (l_regSlices);

    // FIXXME: need cleanup for everythign that is following
    LayeredSurface3D<float> l_tempSlices;
    l_tempSlices.init(dsize, dim);
    dev_temp_slices[dev] = (l_tempSlices);

    Matrix4* l_Matrices;
    checkCudaErrors(cudaMalloc(&l_Matrices, sizeof(Matrix4)*dsize.z));
    dev_recon_matrices[dev] = (l_Matrices);

    checkCudaErrors(cudaMalloc(&l_Matrices, sizeof(Matrix4)*dsize.z));
    dev_recon_matrices_orig[dev] = (l_Matrices);

    float* l_params;
    checkCudaErrors(cudaMalloc(&l_params, sizeof(float) * 6 * dsize.z));
    dev_recon_params[dev] = (l_params);

    float* l_similaritites;
    checkCudaErrors(cudaMalloc(&l_similaritites, sizeof(float) * 5 * dsize.z));
    dev_recon_similarities[dev] = (l_similaritites);

    float* l_gradient;
    checkCudaErrors(cudaMalloc(&l_gradient, sizeof(float) * 7 * dsize.z));
    dev_recon_gradient[dev] = (l_gradient);

    int* l_active_slices;
    checkCudaErrors(cudaMalloc(&l_active_slices, sizeof(int)*dsize.z));
    dev_active_slices[dev] = (l_active_slices);

    checkCudaErrors(cudaMalloc(&l_active_slices, sizeof(int)*dsize.z));
    dev_active_slices2[dev] = (l_active_slices);

    checkCudaErrors(cudaMalloc(&l_active_slices, sizeof(int)*dsize.z));
    dev_active_slices_prev[dev] = (l_active_slices);



    float* l_temp_float_slices;
    checkCudaErrors(cudaMalloc(&l_temp_float_slices, sizeof(float) * 6 * dsize.z));
    dev_temp_float[dev] = (l_temp_float_slices);
    int* l_temp_int_slices;
    checkCudaErrors(cudaMalloc(&l_temp_int_slices, sizeof(int) * 2 * dsize.z));
    dev_temp_int[dev] = (l_temp_int_slices);

  }
}

void Reconstruction::FillRegSlices(float* sdata, std::vector<Matrix4> slices_resampledI2W)
{
  bool allocate = !d_sliceResMatrices_allocated;

  if (allocate)
  {
    int storagesize = 1 + *std::max_element(devicesToUse.begin(), devicesToUse.end());
    dev_d_slicesResampledI2W.resize(storagesize);
    d_sliceResMatrices_allocated = true;
  }
  if (multiThreadedGPU)
  {
    for (int d = 0; d < devicesToUse.size(); d++)
      GPU_workers[d]->prepareFillRegSlices(sdata, slices_resampledI2W, allocate);
    GPU_sync.runNextRound();
  }
  else
  {
    for (int d = 0; d < devicesToUse.size(); d++)
    {
      int dev = devicesToUse[d];
      FillRegSlicesOnX(sdata, slices_resampledI2W, allocate, dev);
    }
  }

  checkCudaErrors(cudaSetDevice(0));
}


void Reconstruction::FillRegSlicesOnX(float* sdata, std::vector<Matrix4>& slices_resampledI2W, bool allocate, int dev)
{
  checkCudaErrors(cudaSetDevice(dev));

  int start = dev_slice_range_offset[dev].x;
  int end = dev_slice_range_offset[dev].y;
  printf("FillRegSlices %d %d %d\n", dev, start, end);

  float* ldata = sdata + start*dev_v_slices_resampled[dev].size.x*dev_v_slices_resampled[dev].size.y;
  float* lsrdata = v_slices_resampled.data + start*dev_v_slices_resampled[dev].size.x*dev_v_slices_resampled[dev].size.y;


  unsigned int num_elems_ = (end - start)*dev_v_slices_resampled[dev].size.x*dev_v_slices_resampled[dev].size.y;
  dev_v_slices_resampled[dev].copyFromHost(ldata);
  //for(int i = 0; i < num_elems_; ++i)
  //  if(sdata[i] != -1)
  //    printf("yes: %d %d %d: %f\n",i%dev_v_slices_resampled[dev].size.x, (i%(dev_v_slices_resampled[dev].size.x*dev_v_slices_resampled[dev].size.y))/dev_v_slices_resampled[dev].size.x, i/(dev_v_slices_resampled[dev].size.x*dev_v_slices_resampled[dev].size.y),sdata[i]);
  //checkCudaErrors(cudaMemcpy(dev_v_slices_resampled[dev].data, ldata, num_elems_*sizeof(float), cudaMemcpyHostToDevice));
  //printf("FillRegSlices %d \n", num_elems_ );
  dev_v_slices_resampled_float[dev].copyFromHost(ldata);
  //checkCudaErrors(cudaMemcpy(dev_v_slices_resampled_float[dev].data, ldata, num_elems_*sizeof(float), cudaMemcpyHostToDevice));
  //printf("FillRegSlices %d \n", num_elems_ );
  //		CHECK_ERROR(FillRegSlices);


  if (allocate)
  {
    Matrix4* t1;
    checkCudaErrors(cudaMalloc((void**)&t1, (end - start)*sizeof(Matrix4)));
    dev_d_slicesResampledI2W[dev] = t1;
  }
  checkCudaErrors(cudaMemcpy(dev_d_slicesResampledI2W[dev], &slices_resampledI2W[start], (end - start)*sizeof(Matrix4), cudaMemcpyHostToDevice));

  //TODO for debug
  dev_v_slices_resampled_float[dev].copyToHost(lsrdata);
  //checkCudaErrors(cudaMemcpy(lsrdata, dev_v_slices_resampled_float[dev].data, num_elems_*sizeof(float), cudaMemcpyDefault));
}


void Reconstruction::combineWeights(float* weights)
{
  unsigned int dev = 0;
  checkCudaErrors(cudaSetDevice(dev));
  cudaDeviceSynchronize();
  checkCudaErrors(cudaMemcpy(weights, dev_volume_weights_[0].data, dev_volume_weights_[0].size.x*dev_volume_weights_[0].size.y*dev_volume_weights_[0].size.z*sizeof(float), cudaMemcpyDeviceToHost));
}


void Reconstruction::debugConfidenceMap(float* cmap)
{
  if (_debugGPU)
  {
    cudaDeviceSynchronize();
    checkCudaErrors(cudaMemcpy(cmap, dev_confidence_map_[0].data,
      dev_confidence_map_[0].size.x*dev_confidence_map_[0].size.y*dev_confidence_map_[0].size.z*sizeof(float), cudaMemcpyDeviceToHost));
  }
}

void Reconstruction::debugAddon(float* addon)
{
  if (_debugGPU)
  {
    cudaDeviceSynchronize();
    checkCudaErrors(cudaMemcpy(addon, dev_addon_[0].data,
      dev_addon_[0].size.x*dev_addon_[0].size.y*dev_addon_[0].size.z*sizeof(float), cudaMemcpyDeviceToHost));
  }

}

void Reconstruction::debugRegSlicesVolume(float* regSlicesHost)
{
  if (_debugGPU)
  {
    cudaDeviceSynchronize();
    checkCudaErrors(cudaMemcpy(regSlicesHost, regSlices.data, regSlices.size.x*regSlices.size.y*regSlices.size.z*sizeof(float), cudaMemcpyDeviceToHost));
  }
}

void Reconstruction::debugWeights(float* weights)
{
  if (/*_debugGPU*/ true)
  {
    cudaDeviceSynchronize();
    checkCudaErrors(cudaMemcpy(weights, v_weights.data, v_weights.size.x*v_weights.size.y*v_weights.size.z*sizeof(float), cudaMemcpyDeviceToHost));
  }
}

void Reconstruction::debugBias(float* bias)
{
  if (_debugGPU)
  {
    cudaDeviceSynchronize();
    checkCudaErrors(cudaMemcpy(bias, v_bias.data, v_bias.size.x*v_bias.size.y*v_bias.size.z*sizeof(float), cudaMemcpyDeviceToHost));
  }
}

void Reconstruction::debugSimslices(float* simslices)
{
  if (_debugGPU)
  {
    cudaDeviceSynchronize();
    checkCudaErrors(cudaMemcpy(simslices, v_simulated_slices.data, v_simulated_slices.size.x*v_simulated_slices.size.y*v_simulated_slices.size.z*sizeof(float), cudaMemcpyDeviceToHost));
  }
}

void Reconstruction::debugSimweights(float* simweights)
{
  if (_debugGPU)
  {
    cudaDeviceSynchronize();
    checkCudaErrors(cudaMemcpy(simweights, v_simulated_weights.data, v_simulated_weights.size.x*v_simulated_weights.size.y*v_simulated_weights.size.z*sizeof(float), cudaMemcpyDeviceToHost));
  }
}

void Reconstruction::debugSiminside(char* siminside)
{
  if (_debugGPU)
  {
    cudaDeviceSynchronize();
    checkCudaErrors(cudaMemcpy(siminside, v_simulated_inside.data, v_simulated_inside.size.x*v_simulated_inside.size.y*v_simulated_inside.size.z*sizeof(char), cudaMemcpyDeviceToHost));
  }
}

void Reconstruction::debugv_PSF_sums(float* v_PSF_sums)
{
  if (_debugGPU)
  {
    cudaDeviceSynchronize();
    checkCudaErrors(cudaMemcpy(v_PSF_sums, v_PSF_sums_.data, v_PSF_sums_.size.x*v_PSF_sums_.size.y*v_PSF_sums_.size.z*sizeof(float), cudaMemcpyDeviceToHost));
  }
}


void Reconstruction::debugNormalizeBias(float* nbias)
{
  if (_debugGPU)
  {
    unsigned int dev = 0;
    checkCudaErrors(cudaSetDevice(dev));
    cudaDeviceSynchronize();
    checkCudaErrors(cudaMemcpy(nbias, dev_bias_[0].data, dev_bias_[0].size.x*dev_bias_[0].size.y*dev_bias_[0].size.z*sizeof(float), cudaMemcpyDeviceToHost));
  }
}

void Reconstruction::debugSmoothMask(float* smoothMask)
{
  if (_debugGPU)
  {
    unsigned int dev = 0;
    checkCudaErrors(cudaSetDevice(dev));
    cudaDeviceSynchronize();
    checkCudaErrors(cudaMemcpy(smoothMask, maskC_.data, maskC_.size.x*maskC_.size.y*maskC_.size.z*sizeof(float), cudaMemcpyDeviceToHost));
  }
}


void Reconstruction::testCPUReg(std::vector<Matrix4>& transf_)
{
#if 0
  Volume<float> recon_float;
  recon_float.init(dev_reconstructed_[0].size, dev_reconstructed_[0].dim);
  //printf("initVolume/n");
  dim3 blockSize3 = dim3(8, 8, 8);
  dim3 gridSize3 = divup(dim3(dev_reconstructed_[0].size.x, dev_reconstructed_[0].size.y, dev_reconstructed_[0].size.z), blockSize3);
  castToFloat << <gridSize3, blockSize3 >> >(dev_reconstructed_[0], recon_float);
  CHECK_ERROR(castToFloat);
  //printf("castToFloat/n");
  regSlices.init(v_slices_resampled_float.size, v_slices_resampled_float.dim);
  unsigned int N = regSlices.size.x*regSlices.size.y*regSlices.size.z;
  initMem<float>(regSlices.data, N, -1.0);
  CHECK_ERROR(initMem);

  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

  cudaExtent asize;
  asize.width = recon_float.size.x;
  asize.height = recon_float.size.y;
  asize.depth = recon_float.size.z;
  //if not already alloced
  if (reconstructed_array == NULL) checkCudaErrors(cudaMalloc3DArray(&reconstructed_array, &channelDesc, asize));

  cudaMemcpy3DParms copyParams = { 0 };
  copyParams.srcPtr = make_cudaPitchedPtr((void*)recon_float.data, recon_float.size.x*sizeof(float),
    recon_float.size.x, recon_float.size.y);
  copyParams.dstArray = reconstructed_array;
  copyParams.extent = asize;
  copyParams.kind = cudaMemcpyDeviceToDevice;
  checkCudaErrors(cudaMemcpy3D(&copyParams));

  reconstructedTex_.addressMode[0] = cudaAddressModeBorder;
  reconstructedTex_.addressMode[1] = cudaAddressModeBorder;
  reconstructedTex_.addressMode[2] = cudaAddressModeBorder;
  reconstructedTex_.filterMode = cudaFilterModeLinear;
  reconstructedTex_.normalized = true;

  checkCudaErrors(cudaBindTextureToArray(reconstructedTex_, reconstructed_array, channelDesc));
  CHECK_ERROR(cudaBindTextureToArray);

  for (int slice = 0; slice < transf_.size(); slice++)
  {

    unsigned int Ns = regSlices.size.x*regSlices.size.y;

    float* sp = regSlices.data + (regSlices.size.x*regSlices.size.y*slice);
    initMem<float>(sp, Ns, -1.0);
    CHECK_ERROR(initMem);

    dim3 blockSize3 = dim3(16, 16, 1);
    dim3 gridSize3 = divup(dim3(regSlices.size.x, regSlices.size.y, 1), blockSize3);
    genenerateRegistrationSliceNoOfs << <gridSize3, blockSize3 >> >(sp, slice, d_slicesResampledI2W, transf_[slice],
      dev_reconstructed_[0].size, make_uint2(regSlices.size.x, regSlices.size.y));
    CHECK_ERROR(genenerateRegistrationSliceNoOfs);
  }
#endif
}

void Reconstruction::getSlicesVol_debug(float* h_imdata)
{
  if (_debugGPU)
  {
    checkCudaErrors(cudaMemcpy(h_imdata, v_slices.data, v_slices.size.x*v_slices.size.y*v_slices.size.z*sizeof(float), cudaMemcpyDeviceToHost));
  }
}

void Reconstruction::getRegSlicesVol_debug(float* h_imdata)
{
  if (_debugGPU)
  {
    checkCudaErrors(cudaMemcpy(h_imdata, v_slices_resampled.data, v_slices_resampled.size.x*v_slices_resampled.size.y*v_slices_resampled.size.z*sizeof(float), cudaMemcpyDeviceToHost));
  }
}

void Reconstruction::syncGPUrecon(float* reconstructed)
{
  if (_debugGPU)
  {
    cudaMemcpy(dev_reconstructed_[0].data, reconstructed, dev_reconstructed_[0].size.x*dev_reconstructed_[0].size.y*dev_reconstructed_[0].size.z*sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
  }
}

void Reconstruction::getVolWeights(float* weights)
{
  cudaMemcpy(weights, dev_reconstructed_volWeigths[0].data,
    dev_reconstructed_volWeigths[0].size.x*dev_reconstructed_volWeigths[0].size.y*dev_reconstructed_volWeigths[0].size.z*sizeof(float),
    cudaMemcpyDeviceToHost);
}

