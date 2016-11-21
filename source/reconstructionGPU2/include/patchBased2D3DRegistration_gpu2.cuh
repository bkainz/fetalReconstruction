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

#include "patchBasedVolume.cuh"
#include "reconVolume.cuh"
#include <irtkMatrix.h>

//TODO document this as soon as optimized in block level reduction
template <typename T>
class PatchBased2D3DRegistration_gpu2
{
public:
  PatchBased2D3DRegistration_gpu2(int cuda_device, PatchBasedVolume<T>* inputStack, ReconVolume<T>* reconstruction);
  ~PatchBased2D3DRegistration_gpu2();

  virtual void run();

protected:
 // void prepareReconTex(int dev);

private:
  PatchBasedVolume<T>* m_inputStack;
  ReconVolume<T>* m_reconstruction;
  cudaArray* m_d_reconstructed_array;

  unsigned int m_NumberOfLevels;
  unsigned int m_NumberOfSteps;
  unsigned int m_NumberOfIterations;
  T m_Epsilon;
  unsigned int m_N;
  int m_cuda_device; 

  T* m_Blurring;
  T* m_LengthOfSteps;

  Matrix4<T>* dev_recon_matrices_orig;
  Matrix4<T>* dev_recon_matrices;

  streambuf* strm_buffer;
  streambuf* strm_buffer_e;
  ofstream file;
  ofstream file_e;
};


