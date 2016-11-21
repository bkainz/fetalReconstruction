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
#include "volume.cuh"
#include "patchBasedVolume.cuh"
#include "patchBasedLayeredSurface3D.cuh"
#include "reconVolume.cuh"
#include "reconConfig.cuh"
#include "pointSpreadFunction.cuh"
#include <irtkImage.h>
#include <thrust/inner_product.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/count.h>

//The globally constant point spread function
extern __constant__ PointSpreadFunction<float> _PSF;

template <typename T>
__global__ void patchBasedPSFReconstructionKernel(PatchBasedVolume<T> inputStack, ReconVolume<T> reconstruction, bool useSpx)
{
  //patch based coordinates
  const uint3 pos = make_uint3(blockIdx.x* blockDim.x + threadIdx.x,
    blockIdx.y* blockDim.y + threadIdx.y,
    blockIdx.z* blockDim.z + threadIdx.z);

  uint3 vSize = inputStack.getXYZPatchGridSize();

  if (/*pos.x >= vSize.x || pos.y >= vSize.y ||*/  pos.z >= vSize.z)
    return;

  //from input data with patch calculation
  //float s = inputStack.getValueFromPatchCoords(pos);

  //from patch buffer
  float s = inputStack.getPatchValue(pos);

  if ((s == -1.0f))
    return;

  ImagePatch2D<T> patch = inputStack.getImagePatch2D(pos.z);
  s = s *patch.scale;

  float3 patchPos = make_float3(pos.x, pos.y, 0);
  float3 patchDim = inputStack.getDim();

  float size_inv = 2.0f * _PSF.m_quality_factor / reconstruction.m_dim.x;
  int xDim = round_((patchDim.x * size_inv));
  int yDim = round_((patchDim.y * size_inv));
  int zDim = round_((patchDim.z * size_inv));

  //truncate if value gets close to epsilon
  int dim = MAX_PSF_SUPPORT;
  int centre = (MAX_PSF_SUPPORT - 1) / 2;

  Matrix4<T> combInvTrans = patch.W2I * (patch.InvTransformation * reconstruction.reconstructedI2W);
  float3 psfxyz;
  float3 _psfxyz = reconstruction.reconstructedW2I*(patch.Transformation*  (patch.I2W * patchPos));
  psfxyz = make_float3(round_(_psfxyz.x), round_(_psfxyz.y), round_(_psfxyz.z));

  float sume = 0;
  for (unsigned int z = 0; z < dim; z++)
  {
    for (unsigned int y = 0; y < dim; y++)
    {
      float oldPSF = FLT_MAX;
      for (unsigned int x = 0; x < dim; x++)
      {
        float3 ofsPos;
        float psfval = _PSF.getPSFParamsPrecomp(ofsPos, psfxyz, make_int3(x - centre, y - centre, z - centre), combInvTrans, patchPos, patchDim);
        if (abs(oldPSF - psfval) < PSF_EPSILON) continue;
        oldPSF = psfval;

        uint3 apos = make_uint3(ofsPos.x, ofsPos.y, ofsPos.z);
        if (useSpx) {
          if (apos.x < reconstruction.m_size.x && apos.y < reconstruction.m_size.y && apos.z < reconstruction.m_size.z && patch.spxMask[pos.x+64*pos.y]=='1') { sume += psfval; }
        }else{
          if (apos.x < reconstruction.m_size.x && apos.y < reconstruction.m_size.y && apos.z < reconstruction.m_size.z) { sume += psfval; }
        }

      }
    }
  }

  // printf("patchDim.x = %f , patchDim.y = %f , patchDim.z = %f,sume %f  \n", patchDim.x, patchDim.y , patchDim.z, sume);

  //fix for crazy values at the border -> too accurate ;)
  if ((sume > PSF_EPSILON) || isnan(sume) )
  {
    inputStack.setPSFsumsValue(pos, sume);
  }
  else
  {
    return;
  }

  for (unsigned int z = 0; z < dim; z++)
  {
    for (unsigned int y = 0; y < dim; y++)
    {
      float oldPSF = FLT_MAX;
      for (unsigned int x = 0; x < dim; x++)
      {
        float3 ofsPos;
        float psfval = _PSF.getPSFParamsPrecomp(ofsPos, psfxyz, make_int3(x - centre, y - centre, z - centre), combInvTrans, patchPos, patchDim);
        if (abs(oldPSF - psfval) < PSF_EPSILON) continue;
        oldPSF = psfval;

        uint3 apos = make_uint3(round_(ofsPos.x), round_(ofsPos.y), round_(ofsPos.z)); //NN
        if (apos.x < reconstruction.m_size.x && apos.y < reconstruction.m_size.y && apos.z < reconstruction.m_size.z
          && reconstruction.m_d_mask[apos.x + apos.y*reconstruction.m_size.x + apos.z*reconstruction.m_size.x*reconstruction.m_size.y] != 0)
        {
          psfval /= sume;
          reconstruction.addReconVolWeightValue(apos, psfval);
          reconstruction.addReconValue(apos, s*psfval);//*psfval
        }

      }
    }
  }

}

template <typename T>
void patchBasedPSFReconstruction_gpu(int cuda_device, PatchBasedVolume<T> & inputStack, ReconVolume<T> & reconstruction, bool useSpx = false)
{
  printf("patchBasedPSFReconstruction_gpu\n");

  //TODO patch batch wise for kernel 2s watchdogs necesary?
  checkCudaErrors(cudaSetDevice(cuda_device));

  dim3 blockSize3 = dim3(8, 8, 8); //max 1024 threads
  dim3 gridSize3  = divup(dim3(inputStack.getXYZPatchGridSize().x, inputStack.getXYZPatchGridSize().y,
    inputStack.getXYZPatchGridSize().z), blockSize3);
  patchBasedPSFReconstructionKernel<T> << <gridSize3, blockSize3 >> >(inputStack, reconstruction, useSpx);
  CHECK_ERROR(patchBasedPSFReconstructionKernel);
  checkCudaErrors(cudaDeviceSynchronize());
}

template void patchBasedPSFReconstruction_gpu<float>(int cuda_device, PatchBasedVolume<float> & inputStack, ReconVolume<float> & reconstruction, bool useSpx);
template void patchBasedPSFReconstruction_gpu<double>(int cuda_device, PatchBasedVolume<double> & inputStack, ReconVolume<double> & reconstruction, bool useSpx);
