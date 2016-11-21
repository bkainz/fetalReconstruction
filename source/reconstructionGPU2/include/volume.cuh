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


__device__ inline float round_(float x)
{
  return roundf(x);
}

template <typename T>
class Volume {
public:

  Volume() { m_size = make_uint3(0, 0, 0); m_dim = make_float3(1, 1, 1); m_d_data = NULL; }

#ifdef __CUDACC__
  __device__ T operator[](const uint3 & pos) const {
    const T d = m_d_data[pos.x + pos.y * m_size.x + pos.z * m_size.x * m_size.y];
    return d; //  / 32766.0f
  }

  __device__ T operator[](const ushort3 & pos) const {
    const T d = m_d_data[pos.x + pos.y * m_size.x + pos.z * m_size.x * m_size.y];
    return d; //  / 32766.0f
  }

  __device__ T v(const uint3 & pos) const {
    return operator[](pos);
  }

  __device__ T vs(const uint3 & pos) const {
    return m_d_data[pos.x + pos.y * m_size.x + pos.z * m_size.x * m_size.y];
  }

  __device__ void set(const uint3 & pos, const T & d){
    m_d_data[pos.x + pos.y * m_size.x + pos.z * m_size.x * m_size.y] = d;
  }

  __device__ void set(const ushort3 & pos, const T & d){
    m_d_data[(unsigned int)pos.x + (unsigned int)pos.y * m_size.x + (unsigned int)pos.z * m_size.x * m_size.y] = d;
  }

  __device__ float3 pos(const uint3 & p) const {
    return make_float3((p.x + 0.5f) * m_dim.x / m_size.x, (p.y + 0.5f) * m_dim.y / m_size.y, (p.z + 0.5f) * m_dim.z / m_size.z);
  }

#endif
  virtual void init(uint3 s, float3 d){
    m_size = s;
    m_dim = d;
    checkCudaErrors(cudaMalloc(&m_d_data, m_size.x*m_size.y*m_size.z*sizeof(T)));
  }

  virtual void release(){
    if (m_d_data != NULL)
    {
      cudaFree(m_d_data);
      m_d_data = NULL;
    }
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

#ifdef __CUDACC__
  __device__ T atomicAddInternal(T* address, T val);
#endif

  virtual void copyFromHost(const T* data)
  {
    if (m_d_data == NULL)
    { 
      cudaMalloc((void **)&m_d_data, m_size.x * m_size.y * m_size.z * sizeof(T));
    }
    checkCudaErrors(cudaMemcpy(m_d_data, data, m_size.x*m_size.y*m_size.z*sizeof(T), cudaMemcpyHostToDevice));
  }

  virtual void copyToHost(T* data) const
  {
    checkCudaErrors(cudaMemcpy(data, m_d_data, m_size.x*m_size.y*m_size.z*sizeof(T), cudaMemcpyDeviceToHost));
  }

  __host__ __device__ T* getDataPtr()
  {
	  return m_d_data;
  }

  uint3 m_size;
  float3 m_dim;
  T * m_d_data;
};

