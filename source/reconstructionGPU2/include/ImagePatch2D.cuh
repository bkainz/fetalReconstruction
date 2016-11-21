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

#ifndef PATCH_BASED_ImagePatch2D_CUH
#define PATCH_BASED_ImagePatch2D_CUH

#include "matrix4.cuh"

template <typename T>
class ImagePatch2D
{
public:
  //uint2 m_bbsize;
  //min index = lower left corner
  //int3 m_pos; // position of the patch
  //char* m_d_SLICmask; //TODO -- fill and use
  //float3 m_origin; //tmp memory for origin reset
  Matrix4<T> Mo; //tmp memory for origin reset
  Matrix4<T> InvMo;
  Matrix4<T> RI2W; //origin resetted I2W
  Matrix4<T> I2W; //image to world matrix
  Matrix4<T> W2I; //world to image matrix
  Matrix4<T> Transformation; //patch to reconstruction transformation
  Matrix4<T> InvTransformation;
  T scale; //scale of the whole patch
  T patchWeight; //weight of the whole patch from EM

  char spxMask[64*64] = {'0'};

};

#endif
