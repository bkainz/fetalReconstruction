#pragma once

#include "patchBasedVolume.cuh"
#include "reconVolume.cuh"
#include <irtkMatrix.h>

template <typename T>
class PatchBased2D3DRegistration_gpu
{
public:
  PatchBased2D3DRegistration_gpu(int cuda_device, PatchBasedVolume<T>* inputStack, ReconVolume<T>* reconstruction);
  ~PatchBased2D3DRegistration_gpu();

  virtual void run();

protected:
  void prepareReconTex(int dev);
  void evaluateCostsMultiplePatches(int active_patches, int patches, int level, float targetBlurring,
    int writeoffset, int writestep, int writenum);

private:
  PatchBasedVolume<T>* m_inputStack;
  ReconVolume<T>* m_reconstruction;
  cudaArray* m_d_reconstructed_array;

  int* dev_active_patches;
  int* dev_active_patches2;
  int* dev_active_patches_prev;

  float* dev_temp_float;
  int* dev_temp_int;

  unsigned int m_NumberOfLevels;
  unsigned int m_NumberOfSteps;
  unsigned int m_NumberOfIterations;
  T m_Epsilon;
  unsigned int m_N;
  int m_cuda_device; 

  T* m_Blurring;
  T* m_LengthOfSteps;

  T* dev_recon_similarities;
  T* dev_recon_gradient;
  Matrix4<float>* dev_recon_matrices_orig;
  Matrix4<float>* dev_recon_matrices;

  streambuf* strm_buffer;
  streambuf* strm_buffer_e;
  ofstream file;
  ofstream file_e;
};


