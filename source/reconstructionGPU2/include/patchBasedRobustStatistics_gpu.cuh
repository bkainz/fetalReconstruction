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
class patchBasedRobustStatistics_gpu
{
public:
  patchBasedRobustStatistics_gpu(std::vector<PatchBasedVolume<T> > & _inputStacks);
  ~patchBasedRobustStatistics_gpu();
  
  void updateInputStacks(std::vector<PatchBasedVolume<T> > & _inputStacks);
  void initializeEMValues();
  void InitializeRobustStatistics(T _min_intensity, T _max_intensity, int cuda_device = 0);
  void EStep();
  void MStep(int iter);
  void Scale();
  void debugON(){ m_debug = true; };

private:
  T m_sigma_s_gpu;
  T m_mix_gpu;
  T m_mix_s_gpu;
  T m_m_gpu;
  T m_sigma_gpu;
  T m_mean_s_gpu;
  T m_mean_s2_gpu;
  T m_step;
  T m_sigma_s2_gpu;
  std::vector<PatchBasedVolume<T> > m_inputStacks;
  int m_cuda_device;
  bool m_debug;

  streambuf* strm_buffer;
  streambuf* strm_buffer_e;
  ofstream file;
  ofstream file_e;

};
