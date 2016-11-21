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

#ifndef Matrix4_CUH
#define Matrix4_CUH

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include <irtkTransformation.h>

#ifndef M_PI
#define M_PI		3.14159265358979323846
#endif 

template<typename T>
struct __builtin_align__(16) real4
{
  T x, y, z, w;
};

template<typename T>
struct __builtin_align__(16) real3
{
  T x, y, z;
};

template<typename T>
__inline__ __host__ __device__ real4<T> make_real4(T x, T y, T z, T w)
{
  real4<T> t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

template<typename T>
__inline__ __host__ __device__ real4<T> make_real3(T x, T y, T z)
{
  real3<T> t; t.x = x; t.y = y; t.z = z; return t;
}


template<typename T>
struct Matrix4 {
  real4<T> data[4];

  //inter-type functions and operator=
  __host__ __device__ operator Matrix4<float>(){
    Matrix4<float> tmp;
    for (int i = 0; i < 4; ++i)
    {
      tmp.data[i].x = (float)(data[i].x);
      tmp.data[i].y = (float)(data[i].y);
      tmp.data[i].z = (float)(data[i].z);
      tmp.data[i].w = (float)(data[i].w);
    }
    return tmp;
  }

  __host__ __device__ operator Matrix4<double>(){
    Matrix4<double> tmp;
    for (int i = 0; i < 4; ++i)
    {
      tmp.data[i].x = (double)(data[i].x);
      tmp.data[i].y = (double)(data[i].y);
      tmp.data[i].z = (double)(data[i].z);
      tmp.data[i].w = (double)(data[i].w);
    }
    return tmp;
  }
};

inline __host__ __device__ float dot(float4 a, float4 b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline __host__ __device__ float dot(float3 a, float3 b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

template<typename T>
inline __host__ __device__ real4<T> operator*(const Matrix4<T> & M, const real4<T> & v){
  return make_real4<T>(dot(M.data[0], v),
    dot(M.data[1], v),
    dot(M.data[2], v),
    dot(M.data[3], v));
}

template<typename T>
inline __host__ __device__ void identityM(Matrix4<T> & M){

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

template<typename T>
inline __host__ __device__ float3 operator*(const Matrix4<T> & M, const float3 & v){

  float a, b, c;

  // Pre-multiply point with transformation matrix
  a = M.data[0].x*v.x + M.data[0].y*v.y + M.data[0].z*v.z + M.data[0].w;
  b = M.data[1].x*v.x + M.data[1].y*v.y + M.data[1].z*v.z + M.data[1].w;
  c = M.data[2].x*v.x + M.data[2].y*v.y + M.data[2].z*v.z + M.data[2].w;

  // Copy result back
  return make_float3(a, b, c);
}

template<typename T>
inline __host__ __device__ real3<T> rotate(const Matrix4<T> & M, const real3<T> & v){
  return make_real3<T>(
    dot(make_real3<T>(M.data[0].x, M.data[0].y, M.data[0].z), v),
    dot(make_real3<T>(M.data[1].x, M.data[1].y, M.data[1].z), v),
    dot(make_real3<T>(M.data[2].x, M.data[2].y, M.data[2].z), v));
}

template<typename T, typename U>
inline __host__ __device__ Matrix4<T> operator*(const Matrix4<T> & A, const Matrix4<U> & B){

  Matrix4<T> tmp;
  for (int i = 0; i < 4; ++i)
  {
    tmp.data[i].x = (T)A.data[i].x * (T)B.data[0].x + (T)A.data[i].y * (T)B.data[1].x + (T)A.data[i].z * (T)B.data[2].x + (T)A.data[i].w * (T)B.data[3].x;
    tmp.data[i].y = (T)A.data[i].x * (T)B.data[0].y + (T)A.data[i].y * (T)B.data[1].y + (T)A.data[i].z * (T)B.data[2].y + (T)A.data[i].w * (T)B.data[3].y;
    tmp.data[i].z = (T)A.data[i].x * (T)B.data[0].z + (T)A.data[i].y * (T)B.data[1].z + (T)A.data[i].z * (T)B.data[2].z + (T)A.data[i].w * (T)B.data[3].z;
    tmp.data[i].w = (T)A.data[i].x * (T)B.data[0].w + (T)A.data[i].y * (T)B.data[1].w + (T)A.data[i].z * (T)B.data[2].w + (T)A.data[i].w * (T)B.data[3].w;
  }
  return tmp;
}

template<typename T>
inline __host__ __device__ float4 operator+(const real4<T> & a, const real4<T> & b){
  return make_real4<T>(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
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

template <typename T>
inline __device__ __host__ void Matrix2Parameters(Matrix4<T>& m, T* params)
{
  T tmp;
  T TOL = 0.000001f;

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

template <typename T>
inline __device__ __host__ void Parameters2Matrix(T *params, Matrix4<T>& mat)
{
  T tx = params[TX];
  T ty = params[TY];
  T tz = params[TZ];

  T rx = params[RX];
  T ry = params[RY];
  T rz = params[RZ];

  T cosrx = cos(rx*(M_PI / 180.0f));
  T cosry = cos(ry*(M_PI / 180.0f));
  T cosrz = cos(rz*(M_PI / 180.0f));
  T sinrx = sin(rx*(M_PI / 180.0f));
  T sinry = sin(ry*(M_PI / 180.0f));
  T sinrz = sin(rz*(M_PI / 180.0f));

  // Create a transformation whose transformation matrix is an identity matrix
  identityM<T>(mat);

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
}


#endif