#pragma once

#include "../reconConfig.cuh"
#include "../matrix4.cuh"

struct transformations
{
  Matrix4<float> targetI2W;
  Matrix4<float> targetW2I;
  Matrix4<float> sourceI2W;
  Matrix4<float> sourceW2I;
  Matrix4<float> Tmat;
  uint3 tsize;
  uint3 ssize;
};


class ccSimilarityMetric
{
public:
  ccSimilarityMetric(int cuda_device);
  ~ccSimilarityMetric();

  float evaluate();
  void setSource(const short* h_in, uint3 source_size_, float3 source_dim_);
  void setTarget(const short* h_in, uint3 target_size_);
  void setTransformations(transformations _t){ m_t = _t; };
  void prepareReconTex();
  void checkGPUMemory();
  void freeGPU();

  float interp(const float3 & pos, float* data, const uint3 size, const float3 dim);
  float v(const uint3 & pos, float* data, const uint3 size);

  cudaArray* m_d_reconstructed_array;
  float* d_source;
  float* d_target;
  uint3 source_size;
  uint3 target_size;
  int device;
  float3 source_dim;

  transformations m_t;
  /*float* d_m_f;
  float* d_sums;
  int* d_counts;
  float* d_averages;
  float* d_results;*/
};

