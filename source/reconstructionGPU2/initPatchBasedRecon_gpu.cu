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
#include "patchBasedVolume.cuh"
#include "patchBasedLayeredSurface3D.cuh"
#include <irtkImage.h>
#include "matrix4.cuh"
#include "reconVolume.cuh"
#include "pointSpreadFunction.cuh"


//__constant__ int d_directions[13][3];
//__constant__ float d_factor[13];
//TODO needs some thinking
//__constant__ Matrix4 d_reconstructedW2I;
//__constant__ Matrix4 d_reconstructedI2W;

//The globally constant point spread function
//@template -- well well needs to stay one type for now
__constant__ PointSpreadFunction<float> _PSF;

template <typename T>
__global__ void patchBasedPatchInitKernel(PatchBasedVolume<T> inputStack, ReconVolume<T> reconstruction, bool useSpx)
{
  //patch based coordinates
  const uint3 pos = make_uint3(blockIdx.x* blockDim.x + threadIdx.x,
    blockIdx.y* blockDim.y + threadIdx.y,
    blockIdx.z* blockDim.z + threadIdx.z);

  uint3 vSize = inputStack.getXYZPatchGridSize();

  if (/*pos.x >= vSize.x || pos.y >= vSize.y ||*/  pos.z >= vSize.z)
    return;

  //from input data with patch calculation
  float s = inputStack.getValueFromPatchCoords(pos);

  if ((s == -1.0f))
    return;

  //mask patches
  ImagePatch2D<T> patch = inputStack.getImagePatch2D(pos.z);

  float3 patchPos = make_float3((float)pos.x, (float)pos.y, 0);
  float3 wpos = patch.Transformation*patch.I2W*patchPos;
  float3 volumePos = reconstruction.reconstructedW2I * wpos;
  uint3 apos = make_uint3(volumePos.x, volumePos.y, volumePos.z);

  if (useSpx)
  {
    if (reconstruction.isMasked(apos) && (patch.spxMask[pos.x+64*pos.y]=='1'))
    {
      inputStack.setPatchValue(pos, s);
    }else if (reconstruction.isMasked(apos) && (patch.spxMask[pos.x+64*pos.y] !='1'))
    {
      inputStack.setPatchValue(pos, -1.0f);
    }

  }else{
    if (reconstruction.isMasked(apos)) { inputStack.setPatchValue(pos, s); }
  }

}


template <typename T>
void initPatchBasedRecon_gpu(int cuda_device, PatchBasedVolume<T> & inputStack, ReconVolume<T> & reconstruction, PointSpreadFunction<float> & h_PSF_, bool useSpx = false)
{
  checkCudaErrors(cudaSetDevice(cuda_device));

  checkCudaErrors(cudaMemcpyToSymbol(_PSF, &(h_PSF_), sizeof(PointSpreadFunction<T>)));

 /* int h_directions[][3] = {
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
  */

  //I actually wanted to avoid this redundancy but the registration required this at the moment
  dim3 blockSize3 = dim3(8, 8, 8); //max 1024 threads
  dim3 gridSize3 = divup(dim3(inputStack.getXYZPatchGridSize().x, inputStack.getXYZPatchGridSize().y,
    inputStack.getXYZPatchGridSize().z), blockSize3);
  patchBasedPatchInitKernel<T> << <gridSize3, blockSize3 >> >(inputStack, reconstruction, useSpx);
  CHECK_ERROR(patchBasedPSFReconstructionKernel);
  checkCudaErrors(cudaDeviceSynchronize());

}

template void initPatchBasedRecon_gpu<float>(int cuda_device, PatchBasedVolume<float> & inputStack, ReconVolume<float> & reconstruction, PointSpreadFunction<float> & _PSF, bool useSpx);
template void initPatchBasedRecon_gpu<double>(int cuda_device, PatchBasedVolume<double> & inputStack, ReconVolume<double> & reconstruction, PointSpreadFunction<float> & _PSF, bool useSpx);
