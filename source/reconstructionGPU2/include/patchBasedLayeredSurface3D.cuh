/*=========================================================================
Library   : Image Registration Toolkit (IRTK)
Copyright : Imperial College, Department of Computing
Visual Information Processing (VIP), 2011 onwards
Date      : $Date: 2013-11-15 14:36:30 +0100 (Fri, 15 Nov 2013) $
Version   : $Revision: 1 $
Changes   : $Author: bkainz, msteinberger $

Copyright (c) 2014, Bernhard Kainz, Markus Steinberger,
Maria Murgasova, Kevin Keraudren
All rights reserved.

If you use this work for research we would very much appreciate if you cite
Bernhard Kainz, Markus Steinberger, Maria Kuklisova-Murgasova, Christina Malamateniou,
Wolfgang Wein, Thomas Torsney-Weir, Torsten Moeller, Mary Rutherford,
Joseph V. Hajnal and Daniel Rueckert:
Fast Volume Reconstruction from Motion Corrupted Stacks of 2D Slices
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

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

//#include "patchBasedObject.cuh"
//#include "patchBasedVolume.cuh"

//TODO make proper class -- see below (float/double problem)
//currently borrowed from 2014 implementation
template <typename T>
class LayeredSurface3D {
public:
  uint3 size;
  float3 dim;
  cudaArray * array;
  cudaSurfaceObject_t surface;
  cudaExtent extent;

  void init(uint3 s, float3 d){
    size = s;
    dim = d;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
    extent.width = size.x; extent.height = size.y; extent.depth = size.z;
    checkCudaErrors(cudaMalloc3DArray(&array, &channelDesc, extent, cudaArrayLayered | cudaArraySurfaceLoadStore));

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = array;
    checkCudaErrors(cudaCreateSurfaceObject(&surface, &resDesc));
  }

  void release(){
    checkCudaErrors(cudaDestroySurfaceObject(surface));
    checkCudaErrors(cudaFreeArray(array));
    array = NULL;
  }

  void copyFromHost(const T* data)
  {
    cudaMemcpy3DParms params = { 0 };
    params.srcPtr = make_cudaPitchedPtr(const_cast<T*>(data), extent.width*sizeof(T), extent.width, extent.height);
    params.dstArray = array;
    params.kind = cudaMemcpyDefault;
    params.extent = extent;
    checkCudaErrors(cudaMemcpy3D(&params));
  }

  void copyToHost(T* data) const
  {
    cudaMemcpy3DParms params = { 0 };
    params.srcArray = array;
    params.dstPtr = make_cudaPitchedPtr(data, extent.width*sizeof(T), extent.width, extent.height);
    params.kind = cudaMemcpyDefault;
    params.extent = extent;
    checkCudaErrors(cudaMemcpy3D(&params));
  }

  void copyFromOther(const LayeredSurface3D& other)
  {
    cudaMemcpy3DParms params = { 0 };
    params.srcArray = other.array;
    params.dstArray = array;
    params.kind = cudaMemcpyDefault;
    params.extent = extent;
    checkCudaErrors(cudaMemcpy3D(&params));
  }

  void copyFromDevice(const T* data)
  {
    cudaMemcpy3DParms params = { 0 };
    params.srcPtr = make_cudaPitchedPtr(const_cast<T*>(data), extent.width*sizeof(T), extent.width, extent.height);
    params.dstArray = array;
    params.kind = cudaMemcpyDeviceToDevice;
    params.extent = extent;
    checkCudaErrors(cudaMemcpy3D(&params));
  }

  void copyToDevice(T* data) const
  {
    cudaMemcpy3DParms params = { 0 };
    params.srcArray = array;
    params.dstPtr = make_cudaPitchedPtr(data, extent.width*sizeof(T), extent.width, extent.height);
    params.kind = cudaMemcpyDeviceToDevice;
    params.extent = extent;
    checkCudaErrors(cudaMemcpy3D(&params));
  }

  void print(uint3 offset = make_uint3(0, 0, 0), uint3 range = make_uint3(0, 0, 0), float ignore = std::numeric_limits<float>::quiet_NaN())
  {
    if (range.x == 0)
      range.x = extent.width - offset.x;
    if (range.y == 0)
      range.y = extent.height - offset.y;
    if (range.z == 0)
      range.z = extent.depth - offset.z;
    std::vector<T> temp(range.x*range.y*range.z);
    cudaMemcpy3DParms params = { 0 };
    params.srcArray = array;
    params.srcPos = make_cudaPos(offset.x, offset.y, offset.z);
    params.dstPtr = make_cudaPitchedPtr(&temp[0], range.x*sizeof(T), range.x, range.y);
    params.kind = cudaMemcpyDefault;
    params.extent = make_cudaExtent(range.x, range.y, range.z);
    checkCudaErrors(cudaMemcpy3D(&params));

    printf("showing %d %d %d:\n", range.x, range.y, range.z);
    if (ignore == std::numeric_limits<float>::quiet_NaN())
      for (unsigned z = 0; z < range.z; ++z)
      {
        for (unsigned y = 0; y < range.y; ++y)
        {
          for (unsigned x = 0; x < range.x; ++x)
          {
            printf(" %f ", temp[x + y*range.x + z*range.x*range.y]);
          }
          printf("\n");
        }
        printf("\n");
      }
    else
      for (unsigned z = 0; z < range.z; ++z)
        for (unsigned y = 0; y < range.y; ++y)
          for (unsigned x = 0; x < range.x; ++x)
          {
            float v = temp[x + y*range.x + z*range.x*range.y];
            if (v != ignore)
              printf("%d,%d,%d: %f\n", x, y, z, v);
          }
  }
};


//layered texture allows to read from and write into a texture (array)
//however interpolation is only possible with type != double
//TODO implment for double
/*
texture<int2,1> my_texture;

static __inline__ __device__ double fetch_double(texture<int2, 1> t, int i)
{
int2 v = tex1Dfetch(t,i);
return __hiloint2double(v.y, v.x);
}
//and write int2
*/

//TODO make proper class
/*
template <typename T>
class PatchBasedLayeredSurface3D : public PatchBasedVolume<T> {
public:
  __host__ void init(irtkGenericImage<T> & stack, irtkRigidTransformation stacktransformation_, uint2 pbbsize, uint2 stride){

    PatchBasedVolume<T>::init(stack, stacktransformation_, pbbsize, stride);

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
    m_extent.width = m_size.x; m_extent.height = m_size.y; m_extent.depth = m_size.z;
    checkCudaErrors(cudaMalloc3DArray(&m_array, &channelDesc, m_extent, cudaArrayLayered | cudaArraySurfaceLoadStore));

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = m_array;
    checkCudaErrors(cudaCreateSurfaceObject(&m_surface, &resDesc));
  }

  void release(){
    PatchBasedVolume<T>::release();
    checkCudaErrors(cudaDestroySurfaceObject(m_surface));
    checkCudaErrors(cudaFreeArray(m_array));
    m_array = NULL;
  }

  void copyFromHost(const T* data)
  {
    cudaMemcpy3DParms params = { 0 };
    params.srcPtr = make_cudaPitchedPtr(const_cast<T*>(data), m_extent.width*sizeof(T), m_extent.width, m_extent.height);
    params.dstArray = m_array;
    params.kind = cudaMemcpyDefault;
    params.extent = m_extent;
    checkCudaErrors(cudaMemcpy3D(&params));
  }

  void copyToHost(T* data) const
  {
    cudaMemcpy3DParms params = { 0 };
    params.srcArray = m_array;
    params.dstPtr = make_cudaPitchedPtr(data, m_extent.width*sizeof(T), m_extent.width, m_extent.height);
    params.kind = cudaMemcpyDefault;
    params.extent = m_extent;
    checkCudaErrors(cudaMemcpy3D(&params));
  }

  void copyFromOther(const PatchBasedLayeredSurface3D& other)
  {
    cudaMemcpy3DParms params = { 0 };
    params.srcArray = other.m_array;
    params.dstArray = m_array;
    params.kind = cudaMemcpyDefault;
    params.extent = m_extent;
    checkCudaErrors(cudaMemcpy3D(&params));
  }

  void copyFromDevice(const T* data)
  {
    cudaMemcpy3DParms params = { 0 };
    params.srcPtr = make_cudaPitchedPtr(const_cast<T*>(data), m_extent.width*sizeof(T), m_extent.width, m_extent.height);
    params.dstArray = m_array;
    params.kind = cudaMemcpyDeviceToDevice;
    params.extent = m_extent;
    checkCudaErrors(cudaMemcpy3D(&params));
  }

#ifdef __CUDACC__
  //in one
  __device__ const T & getValueFromPatchCoords(const uint3 & patch_cords) const
  {
    ImagePatch2D p = d_patches[patch_cords.z];

    float3 scoord = make_float3((patch_cords.x), (patch_cords.y ) , 0);
    scoord = stackW2I*p.I2W*scoord;

    if((scoord.x) >= 0 && (scoord.x) < m_size.x && 
      (scoord.y) >= 0 && (scoord.y) < m_size.y &&
      scoord.z >= 0 && scoord.z < m_size.z)
    {
      //TODO supports only nn double types
      return surf2DLayeredread<float>(m_surface, scoord.x*4, scoord.y, scoord.z, cudaBoundaryModeClamp);
    }

    return 0.0f;
  }

  __device__ void setValueFromPatchCoords(const uint3 & patch_cords, const T & d){
    ImagePatch2D p = d_patches[patch_cords.z];

    float3 scoord = make_float3((patch_cords.x), (patch_cords.y ) , 0);
    scoord = stackW2I*p.I2W*scoord;

    if((scoord.x) >= 0 && (scoord.x) < m_size.x && 
      (scoord.y) >= 0 && (scoord.y) < m_size.y &&
      scoord.z >= 0 && scoord.z < m_size.z)
    {
      surf2DLayeredwrite(d, m_surface, scoord.x*4, scoord.y, scoord.z, cudaBoundaryModeZero);
    }
  }
#endif
  
  void print(uint3 offset = make_uint3(0, 0, 0), uint3 range = make_uint3(0, 0, 0), float ignore = std::numeric_limits<float>::quiet_NaN())
  {
    if (range.x == 0)
      range.x = m_extent.width - offset.x;
    if (range.y == 0)
      range.y = m_extent.height - offset.y;
    if (range.z == 0)
      range.z = m_extent.depth - offset.z;
    std::vector<T> temp(range.x*range.y*range.z);
    cudaMemcpy3DParms params = { 0 };
    params.srcArray = m_array;
    params.srcPos = make_cudaPos(offset.x, offset.y, offset.z);
    params.dstPtr = make_cudaPitchedPtr(&temp[0], range.x*sizeof(T), range.x, range.y);
    params.kind = cudaMemcpyDefault;
    params.extent = make_cudaExtent(range.x, range.y, range.z);
    checkCudaErrors(cudaMemcpy3D(&params));

    printf("showing %d %d %d:\n", range.x, range.y, range.z);
    if (ignore == std::numeric_limits<float>::quiet_NaN())
      for (unsigned z = 0; z < range.z; ++z)
      {
        for (unsigned y = 0; y < range.y; ++y)
        {
          for (unsigned x = 0; x < range.x; ++x)
          {
            printf(" %f ", temp[x + y*range.x + z*range.x*range.y]);
          }
          printf("\n");
        }
        printf("\n");
      }
    else
      for (unsigned z = 0; z < range.z; ++z)
        for (unsigned y = 0; y < range.y; ++y)
          for (unsigned x = 0; x < range.x; ++x)
          {
            float v = temp[x + y*range.x + z*range.x*range.y];
            if (v != ignore)
              printf("%d,%d,%d: %f\n", x, y, z, v);
          }
  }
  __device__ __host__ cudaSurfaceObject_t getSurface(){ return m_surface; };
  __device__ __host__ cudaExtent getExtent(){ return m_extent; };

  cudaArray * m_array;

private:
  cudaSurfaceObject_t m_surface;
  cudaExtent m_extent;
};
*/