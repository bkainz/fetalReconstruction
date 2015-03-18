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
#pragma once
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>


struct POINT3D
{
  short x;
  short y;
  short z;
  float value;
};

//would be nice to have a proper CUDA Matrix class blas?
struct Matrix4 {
  float4 data[4];
  //   inline __host__ __device__ float3 get_translation() const {
  //       return make_float3(data[0].w, data[1].w, data[2].w);
  //   }
};

inline __host__ __device__ float dot(float4 a, float4 b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline __host__ __device__ float dot(float3 a, float3 b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ float4 operator+(const float4 & a, const float4 & b){
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

inline __host__ __device__ float3 operator+(const float3 & a, const float3 & b){
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ float4 operator-(const float4 & a, const float4 & b){
  return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

inline __host__ __device__ float3 operator-(const float3 & a, const float3 & b){
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ float3 operator*(const float3 & a, const float & b){
  return make_float3(a.x*b, a.y*b, a.z*b);
}

inline __host__ __device__ float4 operator*(const Matrix4 & M, const float4 & v){
  return make_float4(dot(M.data[0], v),
    dot(M.data[1], v),
    dot(M.data[2], v),
    dot(M.data[3], v));
}

inline __host__ __device__ void identityM(Matrix4 & M){

  M.data[0].x = 1.0;
  M.data[0].y = 0.0;
  M.data[0].z = 0.0;
  M.data[0].w = 0.0;
  M.data[1].x = 0.0;
  M.data[1].y = 1.0;
  M.data[1].z = 0.0;
  M.data[1].w = 0.0;
  M.data[2].x = 0.0;
  M.data[2].y = 0.0;
  M.data[2].z = 1.0;
  M.data[2].w = 0.0;
  M.data[3].x = 0.0;
  M.data[3].y = 0.0;
  M.data[3].z = 0.0;
  M.data[3].w = 1.0;
  //return M;
}

inline __host__ __device__ float3 operator*(const Matrix4 & M, const float3 & v){

  float a, b, c;

  // Pre-multiply point with transformation matrix
  a = M.data[0].x*v.x + M.data[0].y*v.y + M.data[0].z*v.z + M.data[0].w;
  b = M.data[1].x*v.x + M.data[1].y*v.y + M.data[1].z*v.z + M.data[1].w;
  c = M.data[2].x*v.x + M.data[2].y*v.y + M.data[2].z*v.z + M.data[2].w;

  // Copy result back
  return make_float3(a, b, c);
}


inline __host__ __device__ float3 normalize(float3 v)
{
  float invLen = sqrt(dot(v, v));
  return v * invLen;
}


inline __host__ __device__ float3 rotate(const Matrix4 & M, const float3 & v){
  return make_float3(
    dot(make_float3(M.data[0].x, M.data[0].y, M.data[0].z), v),
    dot(make_float3(M.data[1].x, M.data[1].y, M.data[1].z), v),
    dot(make_float3(M.data[2].x, M.data[2].y, M.data[2].z), v));
}

inline __host__ __device__ Matrix4 operator*(const Matrix4 & A, const Matrix4 & B){

  Matrix4 tmp;
  for (int i = 0; i < 4; ++i)
  {
    tmp.data[i].x = A.data[i].x * B.data[0].x + A.data[i].y * B.data[1].x + A.data[i].z * B.data[2].x + A.data[i].w * B.data[3].x;
    tmp.data[i].y = A.data[i].x * B.data[0].y + A.data[i].y * B.data[1].y + A.data[i].z * B.data[2].y + A.data[i].w * B.data[3].y;
    tmp.data[i].z = A.data[i].x * B.data[0].z + A.data[i].y * B.data[1].z + A.data[i].z * B.data[2].z + A.data[i].w * B.data[3].z;
    tmp.data[i].w = A.data[i].x * B.data[0].w + A.data[i].y * B.data[1].w + A.data[i].z * B.data[2].w + A.data[i].w * B.data[3].w;
  }
  return tmp;
}


typedef std::vector<POINT3D> VOXELCOEFFS;
typedef std::vector<std::vector<VOXELCOEFFS> > SLICECOEFFS;

struct Ref {
  Ref(void * d = NULL) : data(d) {}
  void * data;
};

struct Host {
  Host() : data(NULL) {}
  ~Host() { cudaFreeHost(data); }

  void alloc(unsigned int size) { cudaHostAlloc(&data, size, cudaHostAllocDefault); }
  void * data;

};

struct Device {
  Device() : data(NULL) {}
  ~Device() { cudaFree(data); }

  void alloc(unsigned int size) { cudaMalloc(&data, size); }
  void * data;

};

struct HostDevice {
  HostDevice() : data(NULL) {}
  ~HostDevice() { cudaFreeHost(data); }

  void alloc(unsigned int size) { cudaHostAlloc(&data, size, cudaHostAllocMapped); }
  void * getDevice() const {
    void * devicePtr;
    cudaHostGetDevicePointer(&devicePtr, data, 0);
    return devicePtr;
  }
  void * data;

  void release(){
    if (data != NULL)
    {
      cudaFree(data);
      data = NULL;
    }
  }
};

inline __device__ uint2 thr2pos2(){
#ifdef __CUDACC__
  return make_uint2( __umul24(blockDim.x, blockIdx.x) + threadIdx.x,
    __umul24(blockDim.y, blockIdx.y) + threadIdx.y);
#else
  return make_uint2(0, 0);
#endif
}

template <typename T>
struct Volume {
  uint3 size;
  float3 dim;
  T * data;

  Volume() { size = make_uint3(0, 0, 0); dim = make_float3(1, 1, 1); data = NULL; }
#ifdef __CUDACC__
  __device__ T operator[]( const uint3 & pos ) const {
    const T d = data[pos.x + pos.y * size.x + pos.z * size.x * size.y];
    return d; //  / 32766.0f
  }

  __device__ T operator[]( const ushort3 & pos ) const {
    const T d = data[pos.x + pos.y * size.x + pos.z * size.x * size.y];
    return d; //  / 32766.0f
  }

  __device__ T v(const uint3 & pos) const {
    return operator[](pos);
  }

  __device__ T vs(const uint3 & pos) const {
    return data[pos.x + pos.y * size.x + pos.z * size.x * size.y];
  }

  __device__ void set(const uint3 & pos, const T & d ){
    data[pos.x + pos.y * size.x + pos.z * size.x * size.y] = d;
  }

  __device__ void set(const ushort3 & pos, const T & d ){
    data[(unsigned int)pos.x + (unsigned int)pos.y * size.x + (unsigned int)pos.z * size.x * size.y] = d;
  }

  __device__ float3 pos( const uint3 & p ) const {
    return make_float3((p.x + 0.5f) * dim.x / size.x, (p.y + 0.5f) * dim.y / size.y, (p.z + 0.5f) * dim.z / size.z);
  }

#endif
  void init(uint3 s, float3 d){
    size = s;
    dim = d;
    cudaMalloc((void **)&data, size.x * size.y * size.z * sizeof(T));
  }

  void release(){
    cudaFree(data);
    data = NULL;
  }
};

template <typename T>
struct LayeredSurface3D {
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




template <typename T, typename Allocator = Ref>
struct Slice : public Allocator {
  typedef T PIXEL_TYPE;
  uint2 size;

  Slice() : Allocator() { size = make_uint2(0, 0); }
  Slice(const uint2 & s) { alloc(s); }

  void alloc(const uint2 & s){
    if (s.x == size.x && s.y == size.y)
      return;
    Allocator::alloc(s.x * s.y * sizeof(T));
    size = s;
  }
#ifdef __CUDACC__
  __device__ T & el(){
    return operator[](thr2pos2());
  }

  __device__ const T & el() const {
    return operator[](thr2pos2());
  }

  __device__ T & operator[](const uint2 & pos ){
    return static_cast<T *>(Allocator::data)[pos.x + size.x * pos.y];
  }

  __device__ const T & operator[](const uint2 & pos ) const {
    return static_cast<const T *>(Allocator::data)[pos.x + size.x * pos.y];
  }
#endif
  Slice<T> getDeviceSlice() {
    return Slice<T>(size, Allocator::getDevice());
  }

  operator Slice<T>() {
    return Slice<T>(size, Allocator::data);
  }

  template <typename A1>
  Slice<T, Allocator> & operator=(const Slice<T, A1> & other){
    Slice_copy(*this, other, size.x * size.y * sizeof(T));
    return *this;
  }

  T * data() {
    return static_cast<T *>(Allocator::data);
  }

  const T * data() const {
    return static_cast<const T *>(Allocator::data);
  }
};

template <typename T>
struct Slice<T, Ref> : public Ref{
  typedef T PIXEL_TYPE;
  uint2 size;

  Slice() { size = make_uint2(0, 0); }
  Slice(const uint2 & s, void * d) : Ref(d), size(s) {}
#ifdef __CUDACC__
  __device__ T & el(){
    return operator[](thr2pos2());
  }

  __device__ const T & el() const {
    return operator[](thr2pos2());
  }

  __device__ T & operator[](const uint2 & pos ){
    return static_cast<T *>(Ref::data)[pos.x + size.x * pos.y];
  }

  __device__ const T & operator[](const uint2 & pos ) const {
    return static_cast<const T *>(Ref::data)[pos.x + size.x * pos.y];
  }
#endif

  T * data() {
    return static_cast<T *>(Ref::data);
  }

  const T * data() const {
    return static_cast<const T *>(Ref::data);
  }

};
