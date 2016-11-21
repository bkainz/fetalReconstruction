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
#ifndef gaussFilterConvolution_cuh
#define gaussFilterConvolution_cuh


#include <stdio.h>

#define IMUL(a, b) __mul24(a, b)

#define BLOCK_SIZE 16
#define BLOCK_SIZE_SK_1         64
#define BLOCK_SIZE_SK_2         8

#if 0
#define CHECK_ERROR(function) if (cudaError_t err = cudaGetLastError()) \
{ \
  printf("CUDA Error in " #function "(), line %d: %s\n", __LINE__, cudaGetErrorString(err)); \
}
#else
//#define CHECK_ERROR(function)
#endif

#define MAX_LENGfloatH_SK BLOCK_SIZE_SK_1-1
#define MAX_LENGTH_SK BLOCK_SIZE_SK_1-1
#define SUM_BLOCK 64

int FilterGaussStack(cudaSurfaceObject_t input, cudaSurfaceObject_t output, cudaSurfaceObject_t temp, 
                 unsigned int width, unsigned int height, unsigned int slices, float sigma);

int iDivUp(int a, int b);

#endif