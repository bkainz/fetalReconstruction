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
#include "patchBasedSuperresolution_gpu.cuh"

//The globally constant point spread function
extern __constant__ PointSpreadFunction<float> _PSF;
__constant__ int d_directions[13][3];
__constant__ float d_factor[13];


template <typename T>
__global__ void patchBasedSuperresolution_gpuKernel(PatchBasedVolume<T> inputStack, ReconVolume<T> reconstruction)
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
  float patchVal = inputStack.getPatchValue(pos);
  if ((patchVal == -1.0f))
    return;

  ImagePatch2D<T> patch = inputStack.getImagePatch2D(pos.z);
  float scale = patch.scale;
  patchVal = patchVal * scale;

  float sume = inputStack.getPSFsumsValue(pos); //v_PSF_sums[idx];
  if (sume == 0.0f)
    return;

  float w = inputStack.getWeightValue(pos); 
  float ss = inputStack.getSimulatedPatchValue(pos);
  float patch_weight = patch.patchWeight;   

  if (ss > 0.0f)
    patchVal = (patchVal - ss);
  else
    patchVal = 0.0f;

  float3 patchPos = make_float3(pos.x, pos.y, 0);
  float3 patchDim = inputStack.getDim();

  float size_inv = 2.0f * _PSF.m_quality_factor / reconstruction.m_dim.x;
  int xDim = round_((patchDim.x * size_inv));
  int yDim = round_((patchDim.y * size_inv));
  int zDim = round_((patchDim.z * size_inv));

  //truncate if value gets close to epsilon
  int dim = MAX_PSF_SUPPORT;
  int centre = (MAX_PSF_SUPPORT - 1) / 2;

  Matrix4<float> combInvTrans = patch.W2I * (patch.InvTransformation * reconstruction.reconstructedI2W);
  float3 psfxyz;
  float3 _psfxyz = reconstruction.reconstructedW2I*(patch.Transformation*  (patch.I2W * patchPos));
  psfxyz = make_float3(round_(_psfxyz.x), round_(_psfxyz.y), round_(_psfxyz.z));

  for (int z = 0; z < dim; z++) {
    for (int y = 0; y < dim; y++) {
      float oldPSF = FLT_MAX;
      for (int x = 0; x < dim; x++)
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
          reconstruction.addAddonValue(apos, psfval * w * patch_weight * patchVal);
          reconstruction.addCMapValue(apos, psfval * w * patch_weight);
        }
      }
    }
  }
}


template <typename T>
void patchBasedSuperresolution_gpu<T>::run(int _cuda_device,
  PatchBasedVolume<T>* _inputStack, ReconVolume<T>* _reconstruction)
{
  printf("patchBasedSuperresolution_gpu\n");

  m_inputStack = _inputStack;
  m_reconstruction = _reconstruction;
  m_cuda_device = _cuda_device;

  //TODO patch batch wise for kernel 2s watchdogs necesary?
  checkCudaErrors(cudaSetDevice(m_cuda_device));
  //TODO addon and consider multi-GPU...
  //m_d_buffer as original
  //
  //TODO
  //updatePatchWeights(); --> these are done by Estep and kept with the patches, no explicit update!
  if (m_alpha * m_lambda / (m_delta * m_delta) > 0.068)
  {
    printf("Warning: regularization might not have smoothing effect! Ensure that alpha*lambda/delta^2 is below 0.068.");
  }

  dim3 blockSize3 = dim3(8, 8, 8); //max 1024 threads
  dim3 gridSize3 = divup(dim3(m_inputStack->getXYZPatchGridSize().x, m_inputStack->getXYZPatchGridSize().y,
    m_inputStack->getXYZPatchGridSize().z), blockSize3);
  patchBasedSuperresolution_gpuKernel<T> << <gridSize3, blockSize3 >> >(*m_inputStack, *m_reconstruction);
  CHECK_ERROR(patchBasedPSFReconstructionKernel);
  checkCudaErrors(cudaDeviceSynchronize());

}

template <typename T>
void patchBasedSuperresolution_gpu<T>::updatePatchWeights()
{

}



template <typename T>
__global__ void AdaptiveRegularizationPrepKernel(ReconVolume<T> reconstruction, bool _adaptive, T _alpha, T _min_intensity, T _max_intensity)
{
  const uint3 pos = make_uint3(blockIdx.x * blockDim.x + threadIdx.x,
    blockIdx.y * blockDim.y + threadIdx.y,
    blockIdx.z * blockDim.z + threadIdx.z);

  if (pos.x >= reconstruction.m_size.x || pos.y >= reconstruction.m_size.y || pos.z >= reconstruction.m_size.z)
    return;

  T addon = reconstruction.getAddonValue(pos);
  T cmap = reconstruction.getCMapValue(pos);
  T recon = reconstruction.getReconValue(pos);

  if (!_adaptive)
  {
    if (cmap != 0)
    {
      addon = addon / cmap;
      cmap = 1.0f;
    }
  }

  recon = recon + addon*_alpha;

  if (recon < _min_intensity * 0.9f)
    recon = _min_intensity * 0.9f;
  if (recon > _max_intensity * 1.1f)
    recon = _max_intensity * 1.1f;

  reconstruction.setReconValue(pos, recon);
  reconstruction.setAddonValue(pos, addon);
  reconstruction.setCMapValue(pos, cmap);

}

template <typename T>
__device__ T AdaptiveRegularization1(int i, uint3 pos, uint3 pos2, ReconVolume<T> reconstruction, T* original, float delta)
{
  
  if (pos.x >= reconstruction.m_size.x || pos.y >= reconstruction.m_size.y || pos.z >= reconstruction.m_size.z
    || pos.x < 0 || pos.y < 0 || pos.z < 0 || reconstruction.getCMapValue(pos) <= 0 || reconstruction.getCMapValue(pos2) <= 0 ||
    pos2.x >= reconstruction.m_size.x || pos2.y >= reconstruction.m_size.y || pos2.z >= reconstruction.m_size.z
    || pos2.x < 0 || pos2.y < 0 || pos2.z < 0)
    return 0.0;

  //central differences would be better... improve with texture linear interpolation
  unsigned int idx1 = pos.x + pos.y * reconstruction.m_size.x + pos.z * reconstruction.m_size.x * reconstruction.m_size.y;
  unsigned int idx2 = pos2.x + pos2.y * reconstruction.m_size.x + pos2.z * reconstruction.m_size.x * reconstruction.m_size.y;
  float diff = (original[idx2] - original[idx1]) * sqrt(d_factor[i]) / delta;
  return d_factor[i] / sqrt(1.0 + diff * diff);
}

template <typename T>
__global__ void AdaptiveRegularizationKernel(ReconVolume<T> reconstruction, T* original, T _delta, T _alpha, T _lambda)
{
  uint3 pos = make_uint3(blockIdx.x * blockDim.x + threadIdx.x,
    blockIdx.y * blockDim.y + threadIdx.y,
    blockIdx.z * blockDim.z + threadIdx.z);

  if (pos.x >= reconstruction.m_size.x || pos.y >= reconstruction.m_size.y || pos.z >= reconstruction.m_size.z)
    return;

  T val = 0;
  T valW = 0;
  T sum = 0;

  for (int i = 0; i < 13; i++)
  {
    uint3 pos2 = make_uint3(pos.x + d_directions[i][0], pos.y + d_directions[i][1], pos.z + d_directions[i][2]);

    if ((pos2.x >= 0) && (pos2.x  < reconstruction.m_size.x) && (pos2.y >= 0) && (pos2.y < reconstruction.m_size.y) && (pos2.z >= 0) && (pos2.z < reconstruction.m_size.z))
    {
      T bi = AdaptiveRegularization1(i, pos, pos2, reconstruction, original, _delta);
      T cmapval = reconstruction.getCMapValue(pos2);
      val += bi * reconstruction.getReconValue(pos2) * cmapval; //reconstructed == original2
      valW += bi * cmapval;
      sum += bi;
    }

    uint3 pos3 = make_uint3(pos.x - d_directions[i][0], pos.y - d_directions[i][1], pos.z - d_directions[i][2]); //recycle pos register

    if ((pos3.x >= 0) && (pos3.x < reconstruction.m_size.x) && (pos3.y >= 0) && (pos3.y < reconstruction.m_size.y)
      && (pos3.z >= 0) && (pos3.z < reconstruction.m_size.z) &&
      (pos2.x >= 0) && (pos2.x  < reconstruction.m_size.x) && (pos2.y >= 0) && (pos2.y < reconstruction.m_size.y)
      && (pos2.z >= 0) && (pos2.z < reconstruction.m_size.z)
      )
    {
      T bi = AdaptiveRegularization1(i, pos3, pos2, reconstruction, original, _delta);
      T cmapval = reconstruction.getCMapValue(pos3);
      val += bi * reconstruction.getReconValue(pos3) * cmapval; //reconstructed == original2
      valW += bi * cmapval;
      sum += bi;
    }

  }
 
  T reconval = reconstruction.getReconValue(pos);
  T cmapval = reconstruction.getCMapValue(pos);
  val -= sum * reconval * cmapval;
  valW -= sum * cmapval;

  val = reconval * cmapval + _alpha * _lambda / (_delta * _delta) * val;
  valW = cmapval + _alpha * _lambda / (_delta * _delta) * valW;

  if (valW > 0.0) {
    reconstruction.setReconValue(pos, val / valW);
  }
  else
  {
    reconstruction.setReconValue(pos, 0.0);
  }
  
}

template <typename T>
void patchBasedSuperresolution_gpu<T>::regularize(int rdevice, ReconVolume<T>* _reconstruction)
{
  Volume<T> original;
  original.init(_reconstruction->m_size, _reconstruction->m_dim);
  checkCudaErrors(cudaMemcpy((original.m_d_data), _reconstruction->m_d_data, original.m_size.x*original.m_size.y*original.m_size.z*sizeof(T), cudaMemcpyDeviceToDevice));

  checkCudaErrors(cudaSetDevice(rdevice));
  dim3 blockSize = dim3(8, 8, 8);
  dim3 gridSize = divup(dim3(_reconstruction->m_size.x, _reconstruction->m_size.y, _reconstruction->m_size.z), blockSize);
  
  AdaptiveRegularizationPrepKernel<T> << <gridSize, blockSize >> >(*_reconstruction, m_adaptive, m_alpha, m_min_intensity, m_max_intensity);
   CHECK_ERROR(AdaptiveRegularizationPrep);
  checkCudaErrors(cudaDeviceSynchronize());

  AdaptiveRegularizationKernel<T> << <gridSize, blockSize >> >(*_reconstruction, original.m_d_data, m_delta, m_alpha, m_lambda);
  CHECK_ERROR(AdaptiveRegularizationKernel);
  checkCudaErrors(cudaDeviceSynchronize());

  original.release();
}

template <typename T>
patchBasedSuperresolution_gpu<T>::patchBasedSuperresolution_gpu(T _min_intensity, T _max_intensity, bool _adaptive) : 
m_min_intensity(_min_intensity), m_max_intensity(_max_intensity), m_adaptive(_adaptive)
{
  m_delta = 1;
  m_lambda = 0.1f;
  m_alpha = (0.05f / m_lambda) * m_delta * m_delta;

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

  checkCudaErrors(cudaMemcpyToSymbol(d_factor, factor, 13 * sizeof(float)));
  checkCudaErrors(cudaMemcpyToSymbol(d_directions, h_directions, 3 * 13 * sizeof(int)));

}

template <typename T>
patchBasedSuperresolution_gpu<T>::~patchBasedSuperresolution_gpu()
{

}

template class patchBasedSuperresolution_gpu < float >;
template class patchBasedSuperresolution_gpu < double >;