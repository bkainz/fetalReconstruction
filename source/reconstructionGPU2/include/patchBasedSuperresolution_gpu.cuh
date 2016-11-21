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

template <typename T>
class patchBasedSuperresolution_gpu
{
public:
  patchBasedSuperresolution_gpu(T _min_intensity, T _max_intensity, bool _adaptive = false);
  ~patchBasedSuperresolution_gpu();

  virtual void run(int _cuda_device, PatchBasedVolume<T>* _inputStack, ReconVolume<T>* _reconstruction);
  virtual void regularize(int rdevice, ReconVolume<T>* _reconstruction);
  virtual void updatePatchWeights();

private:
  PatchBasedVolume<T>* m_inputStack;
  ReconVolume<T>* m_reconstruction;

  int m_cuda_device;

  T m_alpha;
  T m_lambda;
  T m_delta;
  bool m_adaptive;
  T m_min_intensity;
  T m_max_intensity;

};