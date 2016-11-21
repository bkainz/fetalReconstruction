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

inline __host__ __device__ float fracf(float v)
{
  return v - floorf(v);
}
inline __host__ __device__ float3 fracf(float3 v)
{
  return make_float3(fracf(v.x), fracf(v.y), fracf(v.z));
}
inline __host__ __device__ int3 operator-(int3 a, int3 b)
{
  return make_int3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ int3 operator+(int3 a, int3 b)
{
  return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ int3 make_int3(float3 a)
{
  return make_int3(int(a.x), int(a.y), int(a.z));
}
inline __host__ __device__ int3 make_int3(uint3 a)
{
  return make_int3(int(a.x), int(a.y), int(a.z));
}
inline __host__ __device__ float3 floorf(float3 v)
{
  return make_float3(floorf(v.x), floorf(v.y), floorf(v.z));
}
inline __host__ __device__ int3 max(int3 a, int3 b)
{
  return make_int3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}
inline __host__ __device__ int3 min(int3 a, int3 b)
{
  return make_int3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}
inline __host__ __device__ int3 make_int3(int s)
{
  return make_int3(s, s, s);
}

template <typename T>
__device__ T v(const uint3 & pos, T* data, const uint3 size) {
  T val = 0.0f;
  if (pos.x < size.x && pos.y < size.y && pos.z < size.z)
    val = data[pos.x + pos.y * size.x + pos.z * size.x * size.y]; 
 
  return val;
}

//ugly compatibility hack 
//TODO double!
template <typename T>
__device__ T interp(const float3 & pos, T* data, const uint3 size, const float3 dim) {
  //const float3 scaled_pos = make_float3((pos.x * size.x / dim.x) - 0.5f, (pos.y * size.y / dim.y) - 0.5f, (pos.z * size.z / dim.z) - 0.5f);
  const float3 scaled_pos = make_float3((pos.x), (pos.y), (pos.z));

  const int3 base = make_int3(floorf(scaled_pos));
  const float3 factor = fracf(scaled_pos);
  const int3 lower = max(base, make_int3(0));
  const int3 upper = min(base + make_int3(1), make_int3(size) - make_int3(1));
  return (T)(v(make_uint3(lower.x, lower.y, lower.z), data, size) * (1 - factor.x) * (1 - factor.y) * (1 - factor.z)
    + v(make_uint3(upper.x, lower.y, lower.z), data, size) * factor.x * (1 - factor.y) * (1 - factor.z)
    + v(make_uint3(lower.x, upper.y, lower.z), data, size) * (1 - factor.x) * factor.y * (1 - factor.z)
    + v(make_uint3(upper.x, upper.y, lower.z), data, size) * factor.x * factor.y * (1 - factor.z)
    + v(make_uint3(lower.x, lower.y, upper.z), data, size) * (1 - factor.x) * (1 - factor.y) * factor.z
    + v(make_uint3(upper.x, lower.y, upper.z), data, size) * factor.x * (1 - factor.y) * factor.z
    + v(make_uint3(lower.x, upper.y, upper.z), data, size) * (1 - factor.x) * factor.y * factor.z
    + v(make_uint3(upper.x, upper.y, upper.z), data, size) * factor.x * factor.y * factor.z);
}