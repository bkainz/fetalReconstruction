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

#ifndef RECONCONFIG_H
#define RECONCONFIG_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <vector_types.h>
#include <vector_functions.h>

//TODO I don't like this here
using namespace std;
#include <irtkVector.h>
#include <irtkMatrix.h>
#include <irtkImage.h>
#include <irtkTransformation.h>

#include <matrix4.cuh>

inline int divup(int a, int b) { return (a + b - 1) / b; }
inline dim3 divup(uint2 a, dim3 b) { return dim3(divup(a.x, b.x), divup(a.y, b.y)); }
inline dim3 divup(dim3 a, dim3 b) { return dim3(divup(a.x, b.x), divup(a.y, b.y), divup(a.z, b.z)); }

#define CHECK_ERROR(function) if (cudaError_t err = cudaGetLastError()) \
{ \
  printf("CUDA Error in " #function "(), line %d: %s\n", __LINE__, cudaGetErrorString(err)); \
  exit(-err); \
}



struct is_larger_zero
{
  __host__ __device__
    bool operator()(const int &x)
  {
    return (x > 0);
  }
};

/*template<typename T>
inline const Matrix4<T> toMatrix4(const irtkMatrix& mat)
{
  Matrix4<T> mmat;
  mmat.data[0] = make_real4<T>(mat(0, 0), mat(0, 1), mat(0, 2), mat(0, 3));
  mmat.data[1] = make_real4<T>(mat(1, 0), mat(1, 1), mat(1, 2), mat(1, 3));
  mmat.data[2] = make_real4<T>(mat(2, 0), mat(2, 1), mat(2, 2), mat(2, 3));
  mmat.data[3] = make_real4<T>(mat(3, 0), mat(3, 1), mat(3, 2), mat(3, 3));
  return mmat;
}*/

template <typename T>
inline const Matrix4<T> toMatrix4(const irtkMatrix& mat)
{
  Matrix4<T> mmat;
  mmat.data[0] = make_real4<T>(mat(0, 0), mat(0, 1), mat(0, 2), mat(0, 3));
  mmat.data[1] = make_real4<T>(mat(1, 0), mat(1, 1), mat(1, 2), mat(1, 3));
  mmat.data[2] = make_real4<T>(mat(2, 0), mat(2, 1), mat(2, 2), mat(2, 3));
  mmat.data[3] = make_real4<T>(mat(3, 0), mat(3, 1), mat(3, 2), mat(3, 3));
  return mmat;
}

template <typename T>
inline const irtkMatrix fromMatrix4(const Matrix4<T>& mat)
{
  irtkMatrix mmat;
  mmat.Initialize(4, 4);
  mmat.Ident();
  mmat.Put(0, 0, mat.data[0].x);
  mmat.Put(0, 1, mat.data[0].y);
  mmat.Put(0, 2, mat.data[0].z);
  mmat.Put(0, 3, mat.data[0].w);

  mmat.Put(1, 0, mat.data[1].x);
  mmat.Put(1, 1, mat.data[1].y);
  mmat.Put(1, 2, mat.data[1].z);
  mmat.Put(1, 3, mat.data[1].w);

  mmat.Put(2, 0, mat.data[2].x);
  mmat.Put(2, 1, mat.data[2].y);
  mmat.Put(2, 2, mat.data[2].z);
  mmat.Put(2, 3, mat.data[2].w);

  mmat.Put(3, 0, mat.data[3].x);
  mmat.Put(3, 1, mat.data[3].y);
  mmat.Put(3, 2, mat.data[3].z);
  mmat.Put(3, 3, mat.data[3].w);

  return mmat;
}


//configuration section
#define __step  0.00001f
//only for experiments. Real PSF is continous
#define PSF_SIZE 128

//usually the kernel runtime watchdog is activated in windows -- workaround:
#if WIN32 
#define MAX_SLICES_PER_RUN_GAUSS 32
#define MAX_SLICES_PER_RUN 128
#else
#define MAX_SLICES_PER_RUN_GAUSS 1000
#define MAX_SLICES_PER_RUN 5000
#endif
//maximum number of GPUs running in parallel
#define MAX_GPU_COUNT 32 

//use a sinc in-plane gauss through plane PSF 
#define USE_SINC_PSF 1

#define PSF_EPSILON 0.00001f
#define USE_INFINITE_PSF_SUPPORT 1
#define MAX_PSF_SUPPORT 12

#endif