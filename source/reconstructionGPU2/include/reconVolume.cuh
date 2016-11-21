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

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "volume.cuh"
#include "reconConfig.cuh"
#include "matrix4.cuh"
#include "pointSpreadFunction.cuh"
#include "interpFunctions.cuh"
//extern __constant__ Matrix4 d_reconstructedW2I;
//extern __constant__ Matrix4 d_reconstructedI2W;


template <typename T>
class ReconVolume : public Volume<T> {
public:

	void init(int cuda_device, uint3 s, float3 d, const Matrix4<float> & reconWorld2Image, const Matrix4<float> & reconImage2World){

    checkCudaErrors(cudaSetDevice(cuda_device));
    m_size = s;
    m_dim = d;
    checkCudaErrors(cudaMalloc((void **)&m_d_data, m_size.x * m_size.y * m_size.z * sizeof(T)));
    checkCudaErrors(cudaMalloc((void **)&m_d_mask, m_size.x * m_size.y * m_size.z * sizeof(char)));
    checkCudaErrors(cudaMalloc((void **)&m_d_reconstructed_volWeigths, m_size.x * m_size.y * m_size.z * sizeof(T)));
    checkCudaErrors(cudaMalloc((void **)&m_d_confidence_map, m_size.x * m_size.y * m_size.z * sizeof(T)));
    checkCudaErrors(cudaMalloc((void **)&m_d_addon, m_size.x * m_size.y * m_size.z * sizeof(T)));
    reset();
    resetAddonCmap();

    reconstructedW2I = reconWorld2Image;
    reconstructedI2W = reconImage2World;

    m_channelDesc = cudaCreateChannelDesc<float>();

    m_asize.width = this->m_size.x;
    m_asize.height = this->m_size.y;
    m_asize.depth = this->m_size.z;

    // ///////////////////////////////////////////////////////////////////////
    // // test code to fix memcheck error
    // cudaExtent volumeSize = make_cudaExtent(m_asize.width, m_asize.height, m_asize.depth);
    // checkCudaErrors(cudaMalloc3DArray(&m_d_reconstructed_array, &m_channelDesc, volumeSize));
    // //////////////////////////////////////////////////////////////////////

    checkCudaErrors(cudaMalloc3DArray(&m_d_reconstructed_array, &m_channelDesc, m_asize));

  }

  void release(){
    Volume<T>::release();  // relaese cudaFree(m_d_data)
    cudaFree(m_d_mask);
    cudaFree(m_d_reconstructed_volWeigths);
    cudaFree(m_d_confidence_map);
    cudaFree(m_d_addon);
    // added missed cudaFree variables - TODO test 
    // cudaFree(m_d_reconstructed_array);
    if (m_d_data != NULL) { // duplicated after Volume<T>::release() just in case
      cudaFree(m_d_data);
      m_d_data = NULL;
    }

  }

  void reset(){
    //checkCudaErrors(cudaMemset(m_d_mask, 0, m_size.x * m_size.y * m_size.z * sizeof(char)));
    checkCudaErrors(cudaMemset(m_d_data, 0, m_size.x * m_size.y * m_size.z * sizeof(T)));
    checkCudaErrors(cudaMemset(m_d_reconstructed_volWeigths, 0, m_size.x * m_size.y * m_size.z * sizeof(T)));

   }

  void resetAddonCmap()
  {
    checkCudaErrors(cudaMemset(m_d_confidence_map, 0, m_size.x * m_size.y * m_size.z * sizeof(T)));
    checkCudaErrors(cudaMemset(m_d_addon, 0, m_size.x * m_size.y * m_size.z * sizeof(T)));
  }

  void equalize();
  void updateReconTex(int dev);

  //mask has to be same size as reconstruction
  void setMask(char* mask_data)
  {
    checkCudaErrors(cudaMemcpy((m_d_mask), mask_data, m_size.x * m_size.y * m_size.z*sizeof(char), cudaMemcpyHostToDevice));
  }


  __host__ __device__ T* getReconstructed_volWeigthsPtr()
  {
    return m_d_reconstructed_volWeigths;
  }

  __host__ __device__ T* getAddonPtr()
  {
    return m_d_addon;
  }

  __host__ __device__ T* getCMapPtr()
  {
    return m_d_confidence_map;
  }

//#ifdef __CUDACC__
  __device__ void addReconVolWeightValue(const uint3 & pos, const T & d)
  {
    if (pos.x >= m_size.x || pos.y >= m_size.y || pos.z >= m_size.z)
      return;

    unsigned int idx = pos.x + pos.y*m_size.x + pos.z*m_size.x*m_size.y;
    atomicAddInternal(&m_d_reconstructed_volWeigths[idx], d);
  }

  __device__ const T & getReconVolWeightValue(const uint3 & pos)
  {
    if (pos.x >= m_size.x || pos.y >= m_size.y || pos.z >= m_size.z)
      return 0;

    unsigned int idx = pos.x + pos.y*m_size.x + pos.z*m_size.x*m_size.y;
    return m_d_reconstructed_volWeigths[idx];
  }

  __device__ void addReconValue(const uint3 & pos, const T & d)
  {
    if (pos.x >= m_size.x || pos.y >= m_size.y || pos.z >= m_size.z)
      return;

    unsigned int idx = pos.x + pos.y*m_size.x + pos.z*m_size.x*m_size.y;
    atomicAddInternal(&m_d_data[idx], d);
  }

  __device__ void setReconValue(const uint3 & pos, const T & d)
  {
    if (pos.x >= m_size.x || pos.y >= m_size.y || pos.z >= m_size.z)
      return;

    unsigned int idx = pos.x + pos.y*m_size.x + pos.z*m_size.x*m_size.y;
    m_d_data[idx] = d;
  }

  __device__ const T & getReconValue(const uint3 & pos)
  {
    if (pos.x >= m_size.x || pos.y >= m_size.y || pos.z >= m_size.z)
      return 0;

    unsigned int idx = pos.x + pos.y*m_size.x + pos.z*m_size.x*m_size.y;
    return m_d_data[idx];
  }

  __device__ const T & getReconValueInterp(const float3 & pos)
  {
    if (pos.x >= m_size.x && pos.y >= m_size.y && pos.z >= m_size.z &&
      pos.x < 0 && pos.y < 0 && pos.z < 0 )
      return 0;

    return interp(pos, m_d_data, m_size, m_dim);
  }

  __device__ void addAddonValue(const uint3 & pos, const T & d)
  {
    if (pos.x >= m_size.x || pos.y >= m_size.y || pos.z >= m_size.z)
      return;

    unsigned int idx = pos.x + pos.y*m_size.x + pos.z*m_size.x*m_size.y;
    atomicAddInternal(&m_d_addon[idx], d);
  }

  __device__ void setAddonValue(const uint3 & pos, const T & d)
  {
    if (pos.x >= m_size.x || pos.y >= m_size.y || pos.z >= m_size.z)
      return;

    unsigned int idx = pos.x + pos.y*m_size.x + pos.z*m_size.x*m_size.y;
    m_d_addon[idx] = d;
  }

  __device__ const T & getAddonValue(const uint3 & pos)
  {
    if (pos.x >= m_size.x || pos.y >= m_size.y || pos.z >= m_size.z)
      return 0;

    unsigned int idx = pos.x + pos.y*m_size.x + pos.z*m_size.x*m_size.y;
    return m_d_addon[idx];
  }

  __device__ void addCMapValue(const uint3 & pos, const T & d)
  {
    if (pos.x >= m_size.x || pos.y >= m_size.y || pos.z >= m_size.z)
      return;

    unsigned int idx = pos.x + pos.y*m_size.x + pos.z*m_size.x*m_size.y;
    atomicAddInternal(&m_d_confidence_map[idx], d);
  }

  __device__ void setCMapValue(const uint3 & pos, const T & d)
  {
    if (pos.x >= m_size.x || pos.y >= m_size.y || pos.z >= m_size.z)
      return;

    unsigned int idx = pos.x + pos.y*m_size.x + pos.z*m_size.x*m_size.y;
    //atomicExch(&m_d_confidence_map[idx], d);
    m_d_confidence_map[idx] = d;
  }

  __device__ const T & getCMapValue(const uint3 & pos)
  {
    if (pos.x >= m_size.x || pos.y >= m_size.y || pos.z >= m_size.z)
      return 0;

    unsigned int idx = pos.x + pos.y*m_size.x + pos.z*m_size.x*m_size.y;
    return m_d_confidence_map[idx];
  }

  __device__ bool isMasked(const uint3 & pos)
  {
    if (pos.x >= m_size.x || pos.y >= m_size.y || pos.z >= m_size.z ||
      pos.x < 0 || pos.y < 0 || pos.z < 0)
      return false;

    unsigned int idx = pos.x + pos.y*m_size.x + pos.z*m_size.x*m_size.y;
    if(m_d_mask[idx] == -1 || m_d_mask[idx] == 0)
    {
      return false;
    }

    return true;
  }

//#endif

  __device__ const T & getReconValueFromTexture(const uint3 & pos);

  //TODO init mask
  char* m_d_mask;
  T* m_d_reconstructed_volWeigths;
  bool disableBiasCorrection;
  //T* m_d_buffer;
  T* m_d_confidence_map;
  T* m_d_addon;

  //mask
  //disableBiasCorrection

  //TODO was constant memory!
  //goes hopefully in register now
  Matrix4<float> reconstructedW2I;
  Matrix4<float> reconstructedI2W;

  cudaArray* m_d_reconstructed_array;

  using Volume<T>::m_size;
  using Volume<T>::m_dim;
  using Volume<T>::m_d_data;
  //using Volume<T>::atomicAddInternal;
    
  cudaExtent m_asize;
  cudaChannelFormatDesc m_channelDesc;
};


