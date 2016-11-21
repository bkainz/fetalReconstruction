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
#include <vector_types.h>
#include <vector_functions.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "matrix4.cuh"
#include "reconConfig.cuh"
#include "patchBasedObject.cuh"

//TODO this could also be a surface
template <typename T>
class PatchBasedVolume : public PatchBasedObject<T> {
public:

  //TODO: SLIC extensions
  PatchBasedVolume() { m_size = make_uint3(0, 0, 0); m_dim = make_float3(1, 1, 1); m_d_data = NULL; }

#ifdef __CUDACC__
  __device__ T operator[](const uint3 & pos) const {
    const T d = m_d_data[pos.x + pos.y * m_size.x + pos.z * m_size.x * m_size.y];
    return d;
  }

  __device__ T operator[](const ushort3 & pos) const {
    const T d = m_d_data[pos.x + pos.y * m_size.x + pos.z * m_size.x * m_size.y];
    return d;
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

  //TODO init with irtkImage
/* PatchBasedVolume deepCopy(const PatchBasedVolume &obj)
  {
    m_size = obj.m_size;
    m_dim = obj.m_dim;
    m_pbbsize = obj.m_pbbsize;
    m_stride = obj.m_stride;
    m_XYZPatchGridSize = obj.m_XYZPatchGridSize;;
    this->init(obj.m_h_stack, /*m_size, m_dim,*//* rigidTransf, m_pbbsize, m_stride);
    checkCudaErrors(cudaMalloc((void**)&d_patches, obj.m_numPatches*sizeof(ImagePatch2D)));
    checkCudaErrors(cudaMemcpy((d_patches), obj.d_patches, obj.m_numPatches*sizeof(ImagePatch2D), cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaMemcpy((m_d_data), obj.m_d_data, obj.m_size.x*obj.m_size.y*obj.m_size.z*sizeof(T), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy((d_m_weightsPtr), obj.d_m_weightsPtr, m_XYZPatchGridSize.x*m_XYZPatchGridSize.y*m_XYZPatchGridSize.z*sizeof(T), cudaMemcpyDeviceToDevice));
    //checkCudaErrors(cudaMemcpy((d_m_biasPtr), obj.d_m_biasPtr, m_XYZPatchGridSize.x*m_XYZPatchGridSize.y*m_XYZPatchGridSize.z*sizeof(T), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy((d_m_simulated_weightsPtr), obj.d_m_simulated_weightsPtr, m_XYZPatchGridSize.x*m_XYZPatchGridSize.y*m_XYZPatchGridSize.z*sizeof(T), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy((d_m_simulated_slicesPtr), obj.d_m_simulated_slicesPtr, m_XYZPatchGridSize.x*m_XYZPatchGridSize.y*m_XYZPatchGridSize.z*sizeof(T), cudaMemcpyDeviceToDevice));
    //checkCudaErrors(cudaMemcpy((d_m_wresidualPtr), obj.d_m_wresidualPtr, m_XYZPatchGridSize.x*m_XYZPatchGridSize.y*m_XYZPatchGridSize.z*sizeof(T), cudaMemcpyDeviceToDevice));
    //checkCudaErrors(cudaMemcpy((d_m_wbPtr), obj.d_m_wbPtr, m_XYZPatchGridSize.x*m_XYZPatchGridSize.y*m_XYZPatchGridSize.z*sizeof(T), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy((d_m_bufferPtr), obj.d_m_bufferPtr, m_XYZPatchGridSize.x*m_XYZPatchGridSize.y*m_XYZPatchGridSize.z*sizeof(T), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy((d_m_simulated_insidePtr), obj.d_m_simulated_insidePtr, m_XYZPatchGridSize.x*m_XYZPatchGridSize.y*m_XYZPatchGridSize.z*sizeof(char), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy((d_m_patchVoxel_count_Ptr), obj.d_m_patchVoxel_count_Ptr, m_XYZPatchGridSize.x*m_XYZPatchGridSize.y*m_XYZPatchGridSize.z*sizeof(int), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy((d_m_PSF_sums_Ptr), obj.d_m_PSF_sums_Ptr, m_XYZPatchGridSize.x*m_XYZPatchGridSize.y*m_XYZPatchGridSize.z*sizeof(T), cudaMemcpyDeviceToDevice));
  }*/

  __host__ void init(irtkGenericImage<T> & stack, irtkRigidTransformation stacktransformation_, uint2 pbbsize, uint2 stride, irtkGenericImage<char>& _mask, T _thickness = 2.5f, bool _superpixel = false, bool useFullSlices = false, bool _debug = false, bool patch_extraction = false, unsigned int stackNo = 0){

    m_size        = make_uint3(stack.GetX(), stack.GetY(), stack.GetZ()); //s;
    m_dim         = make_float3(stack.GetXSize(), stack.GetYSize(), stack.GetZSize()); //d;
    m_mask        = _mask;
    m_thickness   = _thickness;
    //Volume<T>::checkGPUMemory();
    checkCudaErrors(cudaMalloc((void **)&m_d_data, m_size.x * m_size.y * m_size.z * sizeof(T)));
    m_stackTransformation = stacktransformation_;
    m_h_stack = stack;
    stackW2I  = toMatrix4<float>(m_h_stack.GetWorldToImageMatrix());

    m_h_spx_stack = stack;
    printf("m_size = %d x %d x %d \n", m_size.x, m_size.y, m_size.z);
    printf("m_dim  = %f x %f x %f \n", m_dim.x , m_dim.y , m_dim.z);

    int total_pixels = 0;

    if (_superpixel)
    {
      // printf("Generate superpixels ... \n");
      this->generate2DSuperpixelPatches(pbbsize,stride,stackNo,total_pixels,_debug);
    }else{
      // printf("Generate patches ... \n");
      this->generate2DPatches(pbbsize, stride, total_pixels, useFullSlices);
    }

    // calculate overhead pixels
    int pixels = m_size.x * m_size.y * m_size.z;
    int stack_pixels = 0;

    for (int z = 0; z < m_size.z; z++) {
      for (int y = 0; y < m_size.y; y++) {
        for (int x = 0; x < m_size.x; x++) {
          double xx = x;
          double yy = y;
          double zz = z;
          m_h_stack.ImageToWorld(xx,yy,zz);
          m_mask.WorldToImage(xx,yy,zz);
          if (xx < 0 || yy < 0 || zz < 0 || xx >= m_mask.GetX() || yy >= m_mask.GetY() || zz >= m_mask.GetZ()) continue;
          if (m_mask.Get(xx,yy,zz)>0 && m_h_stack.Get(x,y,z)>0) { stack_pixels += 1; }
        }
      }
    }


    int   overhead_pixels = total_pixels - stack_pixels;
    float overhead_ratio  = (1.0 * overhead_pixels) / (1.0 * stack_pixels);

    if (patch_extraction) 
    {
      cout << pixels          << ",";
      cout << stack_pixels    << ",";
      cout << total_pixels    << ",";
      cout << overhead_pixels << ",";
      cout << overhead_ratio  << ",";
      cout << endl;

      return;
    }

    printf("pixels = %d - stack pixels = %d - patch pixels = %d - overhead pixels = %d - overhead ratio = %f \n", pixels, stack_pixels, total_pixels, overhead_pixels, overhead_ratio);

    cout << "done!" << endl;

    m_XYZPatchGridSize = PatchBasedObject<T>::getXYZPatchGridSize();
    unsigned int N = m_XYZPatchGridSize.x * m_XYZPatchGridSize.y * m_XYZPatchGridSize.z;
    numDemonElems = N;
    checkCudaErrors(cudaMalloc((void **)&d_m_weightsPtr, N * sizeof(T)));
    //checkCudaErrors(cudaMalloc((void **)&d_m_biasPtr, N * sizeof(T)));
    checkCudaErrors(cudaMalloc((void **)&d_m_simulated_weightsPtr, N * sizeof(T)));
    checkCudaErrors(cudaMalloc((void **)&d_m_simulated_patchesPtr, N * sizeof(T)));
    //checkCudaErrors(cudaMalloc((void **)&d_m_wresidualPtr, N * sizeof(T)));
    //checkCudaErrors(cudaMalloc((void **)&d_m_wbPtr, N * sizeof(T)));
    checkCudaErrors(cudaMalloc((void **)&d_m_bufferPtr, N * sizeof(T)));
    checkCudaErrors(cudaMalloc((void **)&d_m_simulated_insidePtr, N * sizeof(char)));
    checkCudaErrors(cudaMalloc((void **)&d_m_patchVoxel_count_Ptr, N * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_m_PSF_sums_Ptr, N * sizeof(T)));

    checkCudaErrors(cudaMalloc((void **)&d_m_regPatchesPtr, N * sizeof(T)));
    checkCudaErrors(cudaMalloc((void **)&d_m_PatchesPtr, N * sizeof(T)));

    checkCudaErrors(cudaMemcpy(m_d_data, stack.GetPointerToVoxels(), m_size.x*m_size.y*m_size.z*sizeof(T), cudaMemcpyHostToDevice));

    reset();
    //Volume<T>::checkGPUMemory();
  }

  void reset()
  {
    checkCudaErrors(cudaMemset(d_m_weightsPtr, 0, numDemonElems*sizeof(T)));
    //cudaMemset(d_m_biasPtr, 0, numDemonElems*sizeof(T)));
    checkCudaErrors(cudaMemset(d_m_simulated_weightsPtr, 0, numDemonElems*sizeof(T)));
    checkCudaErrors(cudaMemset(d_m_simulated_patchesPtr, 0, numDemonElems*sizeof(T)));
    //cudaMemset(d_m_wresidualPtr, 0, numDemonElems*sizeof(T)));
    //cudaMemset(d_m_wbPtr, 0, numDemonElems*sizeof(T)));
    checkCudaErrors(cudaMemset(d_m_bufferPtr, 0, numDemonElems*sizeof(T)));
    checkCudaErrors(cudaMemset(d_m_simulated_insidePtr, 0, numDemonElems*sizeof(char)));
    checkCudaErrors(cudaMemset(d_m_patchVoxel_count_Ptr, 0, numDemonElems*sizeof(int)));
    checkCudaErrors(cudaMemset(d_m_PSF_sums_Ptr, 0, numDemonElems*sizeof(T)));

    checkCudaErrors(cudaMemset(d_m_regPatchesPtr, 0, numDemonElems*sizeof(T)));
    checkCudaErrors(cudaMemset(d_m_PatchesPtr, 0, numDemonElems*sizeof(T)));
  }

  void release(){
    checkCudaErrors(cudaFree(m_d_data));
    m_d_data = NULL;
    PatchBasedObject<T>::release();
    checkCudaErrors(cudaFree(d_m_weightsPtr));
    //cudaFree(d_m_biasPtr);
    checkCudaErrors(cudaFree(d_m_simulated_weightsPtr));
    checkCudaErrors(cudaFree(d_m_simulated_patchesPtr));
    //cudaFree(d_m_wresidualPtr);
    //cudaFree(d_m_wbPtr);
    checkCudaErrors(cudaFree(d_m_bufferPtr));
    checkCudaErrors(cudaFree(d_m_simulated_insidePtr));
    checkCudaErrors(cudaFree(d_m_PSF_sums_Ptr));

    checkCudaErrors(cudaFree(d_m_regPatchesPtr));
    checkCudaErrors(cudaFree(d_m_PatchesPtr));
  }

  __device__ __host__ T* getDataPointer(){ return m_d_data; };

//#ifdef __CUDACC__
  //in one
  __device__ virtual const T & getValueFromPatchCoords(const uint3 & patch_cords) const
  {
    ImagePatch2D<T> p = d_patches[patch_cords.z];

    float3 scoord = make_float3((patch_cords.x), (patch_cords.y ) , 0);
    scoord = stackW2I*p.I2W*scoord;

    if((scoord.x) >= 0 && (scoord.x) < m_size.x && 
      (scoord.y) >= 0 && (scoord.y) < m_size.y &&
      scoord.z >= 0 && scoord.z < m_size.z)
    {
      unsigned int idx = scoord.x + scoord.y * m_size.x + scoord.z * m_size.x * m_size.y;
      return m_d_data[idx];
    }

    return 0.0f;
  }

  __device__ void setWeightValue(const uint3 & pos, const T & d)
  {
    if (pos.x >= m_XYZPatchGridSize.x || pos.y >= m_XYZPatchGridSize.y || pos.z >= m_XYZPatchGridSize.z)
      return;

    unsigned int idx = pos.x + pos.y*m_XYZPatchGridSize.x + pos.z*m_XYZPatchGridSize.x*m_XYZPatchGridSize.y;
    d_m_weightsPtr[idx] = d;
  }

  __device__ const T & getWeightValue(const uint3 & pos)
  {
    if (pos.x >= m_XYZPatchGridSize.x || pos.y >= m_XYZPatchGridSize.y || pos.z >= m_XYZPatchGridSize.z)
      return 0;

    unsigned int idx = pos.x + pos.y*m_XYZPatchGridSize.x + pos.z*m_XYZPatchGridSize.x*m_XYZPatchGridSize.y;
    return d_m_weightsPtr[idx];
  }

  __device__ void setPSFsumsValue(const uint3 & pos, const T & d)
  {
    if (pos.x >= m_XYZPatchGridSize.x || pos.y >= m_XYZPatchGridSize.y || pos.z >= m_XYZPatchGridSize.z)
      return;

    unsigned int idx = pos.x + pos.y*m_XYZPatchGridSize.x + pos.z*m_XYZPatchGridSize.x*m_XYZPatchGridSize.y;
    d_m_PSF_sums_Ptr[idx] = d;
  }

  __device__ const T & getPSFsumsValue(const uint3 & pos)
  {
    if (pos.x >= m_XYZPatchGridSize.x || pos.y >= m_XYZPatchGridSize.y || pos.z >= m_XYZPatchGridSize.z)
      return 0;

    unsigned int idx = pos.x + pos.y*m_XYZPatchGridSize.x + pos.z*m_XYZPatchGridSize.x*m_XYZPatchGridSize.y;
    return d_m_PSF_sums_Ptr[idx];
  }

  __device__ void incPatchVoxelcountValue(const uint3 & pos)
  {
    if (pos.x >= m_XYZPatchGridSize.x || pos.y >= m_XYZPatchGridSize.y || pos.z >= m_XYZPatchGridSize.z)
      return;

    unsigned int idx = pos.x + pos.y*m_XYZPatchGridSize.x + pos.z*m_XYZPatchGridSize.x*m_XYZPatchGridSize.y;
    d_m_patchVoxel_count_Ptr[idx]++;
  }

  __device__ void setRegPatchValue(const uint3 & pos, const T & d)
  {
    if (pos.x >= m_XYZPatchGridSize.x || pos.y >= m_XYZPatchGridSize.y || pos.z >= m_XYZPatchGridSize.z)
      return;

    unsigned int idx = pos.x + pos.y*m_XYZPatchGridSize.x + pos.z*m_XYZPatchGridSize.x*m_XYZPatchGridSize.y;
    d_m_regPatchesPtr[idx] = d;
  }

  __device__ const T & getRegPatchValue(const uint3 & pos)
  {
    if (pos.x >= m_XYZPatchGridSize.x || pos.y >= m_XYZPatchGridSize.y || pos.z >= m_XYZPatchGridSize.z)
      return 0;

    unsigned int idx = pos.x + pos.y*m_XYZPatchGridSize.x + pos.z*m_XYZPatchGridSize.x*m_XYZPatchGridSize.y;
    return d_m_regPatchesPtr[idx];
  }


  __device__ const T & getBufferValue(const uint3 & pos)
  {
    if (pos.x >= m_XYZPatchGridSize.x || pos.y >= m_XYZPatchGridSize.y || pos.z >= m_XYZPatchGridSize.z)
      return 0;

    unsigned int idx = pos.x + pos.y*m_XYZPatchGridSize.x + pos.z*m_XYZPatchGridSize.x*m_XYZPatchGridSize.y;
    return d_m_bufferPtr[idx];
  }

  //TODO make set/get macro
  __device__ void setPatchValue(const uint3 & pos, const T & d)
  {
    if (pos.x >= m_XYZPatchGridSize.x || pos.y >= m_XYZPatchGridSize.y || pos.z >= m_XYZPatchGridSize.z)
      return;

    unsigned int idx = pos.x + pos.y*m_XYZPatchGridSize.x + pos.z*m_XYZPatchGridSize.x*m_XYZPatchGridSize.y;
    d_m_PatchesPtr[idx] = d;
  }

  __device__ const T & getPatchValue(const uint3 & pos)
  {
    if (pos.x >= m_XYZPatchGridSize.x || pos.y >= m_XYZPatchGridSize.y || pos.z >= m_XYZPatchGridSize.z)
      return 0;

    unsigned int idx = pos.x + pos.y*m_XYZPatchGridSize.x + pos.z*m_XYZPatchGridSize.x*m_XYZPatchGridSize.y;
    return d_m_PatchesPtr[idx];
  }

  __device__ void setSimulatedPatchValue(const uint3 & pos, const T & d)
  {
    if (pos.x >= m_XYZPatchGridSize.x || pos.y >= m_XYZPatchGridSize.y || pos.z >= m_XYZPatchGridSize.z)
      return;

    unsigned int idx = pos.x + pos.y*m_XYZPatchGridSize.x + pos.z*m_XYZPatchGridSize.x*m_XYZPatchGridSize.y;
    d_m_simulated_patchesPtr[idx] = d;
  }

  __device__ const T & getSimulatedPatchValue(const uint3 & pos)
  {
    if (pos.x >= m_XYZPatchGridSize.x || pos.y >= m_XYZPatchGridSize.y || pos.z >= m_XYZPatchGridSize.z)
      return 0;

    unsigned int idx = pos.x + pos.y*m_XYZPatchGridSize.x + pos.z*m_XYZPatchGridSize.x*m_XYZPatchGridSize.y;
    return d_m_simulated_patchesPtr[idx];
  }

  __device__ void setSimulatedWeight(const uint3 & pos, const T & d)
  {
    if (pos.x >= m_XYZPatchGridSize.x || pos.y >= m_XYZPatchGridSize.y || pos.z >= m_XYZPatchGridSize.z)
      return;

    unsigned int idx = pos.x + pos.y*m_XYZPatchGridSize.x + pos.z*m_XYZPatchGridSize.x*m_XYZPatchGridSize.y;
    d_m_simulated_weightsPtr[idx] = d;
  }

  __device__ const T & getSimulatedWeight(const uint3 & pos)
  {
    if (pos.x >= m_XYZPatchGridSize.x || pos.y >= m_XYZPatchGridSize.y || pos.z >= m_XYZPatchGridSize.z)
      return 0;

    unsigned int idx = pos.x + pos.y*m_XYZPatchGridSize.x + pos.z*m_XYZPatchGridSize.x*m_XYZPatchGridSize.y;
    return d_m_simulated_weightsPtr[idx];
  }

  __device__ void setSimulatedInside(const uint3 & pos, const T & d)
  {
    if (pos.x >= m_XYZPatchGridSize.x || pos.y >= m_XYZPatchGridSize.y || pos.z >= m_XYZPatchGridSize.z)
      return;

    unsigned int idx = pos.x + pos.y*m_XYZPatchGridSize.x + pos.z*m_XYZPatchGridSize.x*m_XYZPatchGridSize.y;
    d_m_simulated_insidePtr[idx] = d;
  }

  __device__ const char & getSimulatedInside(const uint3 & pos)
  {
    if (pos.x >= m_XYZPatchGridSize.x || pos.y >= m_XYZPatchGridSize.y || pos.z >= m_XYZPatchGridSize.z)
      return 0;

    unsigned int idx = pos.x + pos.y*m_XYZPatchGridSize.x + pos.z*m_XYZPatchGridSize.x*m_XYZPatchGridSize.y;
    return d_m_simulated_insidePtr[idx];
  }

  __device__ void setBufferValue(const uint3 & pos, const T & d)
  {
    if (pos.x >= m_XYZPatchGridSize.x || pos.y >= m_XYZPatchGridSize.y || pos.z >= m_XYZPatchGridSize.z)
      return;

    unsigned int idx = pos.x + pos.y*m_XYZPatchGridSize.x + pos.z*m_XYZPatchGridSize.x*m_XYZPatchGridSize.y;
    d_m_bufferPtr[idx] = d;
  }

//#endif

  __device__ __host__ T* getWeigthDataPtr()
  {
    return d_m_weightsPtr;
  }

  __device__ __host__ int* getpatchVoxel_countDataPtr()
  {
    return d_m_patchVoxel_count_Ptr;
  }

  __device__ __host__ T* getRegPatchesPtr()
  {
    return d_m_regPatchesPtr;
  }

  __device__ __host__ T* getDataPtr()
  {
    return m_d_data;
  }

  __device__ __host__ T* getBufferPtr()
  {
    return d_m_bufferPtr;
  }

  __device__ __host__ T* getPatchesPtr()
  {
    return d_m_PatchesPtr;
  }

  __device__ __host__ char* getSimInsidePtr()
  {
    return d_m_simulated_insidePtr;
  }

  __device__ __host__ T* getSimWeightsPtr()
  {
    return d_m_simulated_weightsPtr;
  }

  __device__ __host__ T* getSimPatchesPtr()
  {
    return d_m_simulated_patchesPtr;
  }
protected:
  int numDemonElems;

  //demons:
  T *d_m_weightsPtr;
  //T *d_m_biasPtr;
  T *d_m_simulated_weightsPtr;
  T *d_m_simulated_patchesPtr;
  //T *d_m_wresidualPtr;
  //T *d_m_wbPtr;
  T *d_m_bufferPtr;
  char *d_m_simulated_insidePtr;
  int *d_m_patchVoxel_count_Ptr;
  T *d_m_PSF_sums_Ptr;
  Matrix4<float> stackW2I;

  //test if non recon space patches are also good for registration
  T* d_m_regPatchesPtr;
  T* d_m_PatchesPtr;

  //why are they not inherited when using gcc? stupid c++ standard in gcc
  using Volume<T>::m_size;
  using Volume<T>::m_dim;
  using Volume<T>::m_d_data;
  using PatchBasedObject<T>::m_XYZPatchGridSize;
  using PatchBasedObject<T>::m_numPatches;
  using PatchBasedObject<T>::m_pbbsize;
  using PatchBasedObject<T>::m_stride;
  using PatchBasedObject<T>::d_patches;
  using PatchBasedObject<T>::m_h_stack;
  using PatchBasedObject<T>::m_h_spx_stack;
  using PatchBasedObject<T>::m_stackTransformation;
  using PatchBasedObject<T>::m_mask;
  using PatchBasedObject<T>::m_thickness;
};
