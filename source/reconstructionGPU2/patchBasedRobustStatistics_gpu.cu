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
#include "patchBasedRobustStatistics_gpu.cuh"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/functional.h>
#include <thrust/tuple.h>
#include <thrust/transform_reduce.h>
#include <math.h>

//The globally constant point spread function
extern __constant__ PointSpreadFunction<float> _PSF;

using namespace thrust;

template <typename T>
__global__ void resetScaleAndWeights(PatchBasedVolume<T> inputStack)
{
  const unsigned int idx = blockIdx.x* blockDim.x + threadIdx.x;

  uint3 vSize = inputStack.getXYZPatchGridSize();
  
  if (idx >= vSize.z)
  return;

  inputStack.getImagePatch2D(idx).scale = 1.0f;
  inputStack.getImagePatch2D(idx).patchWeight = 1.0f;
}

template <typename T>
__global__ void InitializeEMValuesKernel(PatchBasedVolume<T> inputStack)
{
  const uint3 pos = make_uint3(blockIdx.x* blockDim.x + threadIdx.x,
    blockIdx.y* blockDim.y + threadIdx.y,
    blockIdx.z* blockDim.z + threadIdx.z);

  uint3 vSize = inputStack.getXYZPatchGridSize();

  if (pos.x >= vSize.x || pos.y >= vSize.y || pos.z >= vSize.z)
    return;

  T s = inputStack.getPatchValue(pos);

  if (s != -1 && s != 0)
  {
    inputStack.setWeightValue(pos, 1);
  }
  else
  {
    inputStack.setWeightValue(pos, 0);
  }
}

template <typename T>
void patchBasedRobustStatistics_gpu<T>::initializeEMValues()
{
  for (int i = 0; i < m_inputStacks.size(); i++)
  {
    resetScaleAndWeights << < divup(m_inputStacks[i].getXYZPatchGridSize().z, 512), 512 >> >(m_inputStacks[i]);
    checkCudaErrors(cudaDeviceSynchronize());

    dim3 blockSize3 = dim3(8, 8, 8);
    dim3 gridSize3 = divup(dim3(m_inputStacks[i].getXYZPatchGridSize().x, m_inputStacks[i].getXYZPatchGridSize().y,
      m_inputStacks[i].getXYZPatchGridSize().z), blockSize3);
    InitializeEMValuesKernel<T> << <gridSize3, blockSize3 >> >(m_inputStacks[i]);
    CHECK_ERROR(InitializeEMValuesKernel);
    checkCudaErrors(cudaDeviceSynchronize());
  }
}

template <typename T>
inline __host__ __device__ T G_(T x, T s)
{
  return __step*exp(-x*x / (2.0f*s)) / (sqrt(6.28f*s));
}

template <typename T>
inline __host__ __device__ T M_(T m)
{
  return m*__step;
}

template <typename T>
__global__ void EStepKernel(PatchBasedVolume<T> inputStack, T _m, T _sigma, T _mix)
{
  const uint3 pos = make_uint3(blockIdx.x* blockDim.x + threadIdx.x,
    blockIdx.y* blockDim.y + threadIdx.y,
    blockIdx.z* blockDim.z + threadIdx.z);

  uint3 vSize = inputStack.getXYZPatchGridSize();

  if (pos.x >= vSize.x || pos.y >= vSize.y || pos.z >= vSize.z)
    return;

  T s = inputStack.getPatchValue(pos);
  T sw = inputStack.getWeightValue(pos);

  if ((s == -1) || sw <= 0)
    return;

  T ss = inputStack.getSimulatedPatchValue(pos); 
  ImagePatch2D<T> patch = inputStack.getImagePatch2D(pos.z);
  T scale = patch.scale;
  T patchVal = s * scale;

  patchVal -= ss;

  //Gaussian distribution for inliers (likelihood)
  T g = G_(patchVal, _sigma);
  //Uniform distribution for outliers (likelihood)
  T m = M_(_m);

  T weight = (g * _mix) / (g *_mix + m * (1.0 - _mix));
  if (sw > 0)
  {
    inputStack.setWeightValue(pos, weight);
  }
  else
  {
    inputStack.setWeightValue(pos, 0.0f);
  }

}

template <typename T>
struct transformPatchPotential
{
  __host__ __device__
    thrust::tuple<T, T> operator()(const thrust::tuple<T, T>& a)
  {

    if (thrust::get<1>(a) > 0.99)
    {
      return thrust::make_tuple(((1.0 - thrust::get<0>(a)) * (1.0 - thrust::get<0>(a))), 1.0);
    }
    else
    {
      return thrust::make_tuple(0.0f, 0.0f);
    }

  }
};

template <typename T>
struct reducePatchPotential
{
  __host__ __device__
    thrust::tuple<T, T> operator()(const thrust::tuple<T, T>& a, const thrust::tuple<T, T>& b)
  {
    return thrust::make_tuple(thrust::get<0>(a) +thrust::get<0>(b), thrust::get<1>(a) +thrust::get<1>(b));
  }
};

template <typename T>
__global__ void copyFromWeightsAndScales(PatchBasedVolume<T> inputStack, unsigned int ofs, T* scales, T* weights)
{
  const unsigned int idx = blockIdx.x* blockDim.x + threadIdx.x;

  uint3 vSize = inputStack.getXYZPatchGridSize();

  if (idx >= vSize.z)
    return;

  ImagePatch2D<T> patch = inputStack.getImagePatch2D(idx);

  scales[idx + ofs] = patch.scale;
  weights[idx + ofs] = patch.patchWeight;

}


template <typename T>
__global__ void copyToWeightsAndScales(PatchBasedVolume<T> inputStack, unsigned int ofs, T* scales, T* weights)
{
  const unsigned int idx = blockIdx.x* blockDim.x + threadIdx.x;

  uint3 vSize = inputStack.getXYZPatchGridSize();

  if (idx >= vSize.z)
    return;
  
  inputStack.getImagePatch2D(idx).scale = scales[idx + ofs];
  inputStack.getImagePatch2D(idx).patchWeight = weights[idx + ofs];
}

template <typename T>
__global__ void copyToScales(PatchBasedVolume<T> inputStack, unsigned int ofs, T* scales)
{
  const unsigned int idx = blockIdx.x* blockDim.x + threadIdx.x;

  uint3 vSize = inputStack.getXYZPatchGridSize();

  if (idx >= vSize.z)
    return;

  inputStack.getImagePatch2D(idx).scale = scales[idx + ofs];
}

template <typename T>
void patchBasedRobustStatistics_gpu<T>::EStep()
{

  //TODO remove:
  m_debug = true;

  printf("EStep_gpu\n");
  cerr.rdbuf(file_e.rdbuf());
  cout.rdbuf(file.rdbuf());

  unsigned int inputIndex;
  unsigned int numPatches = 0;
  for (int i = 0; i < m_inputStacks.size(); i++)
  {
    numPatches += m_inputStacks[i].getXYZPatchGridSize().z;
  }

  std::vector<T> patch_potential(numPatches, 0);

  for (int i = 0; i < m_inputStacks.size(); i++)
  {
    dim3 blockSize3 = dim3(8, 8, 8);
    dim3 gridSize3 = divup(dim3(m_inputStacks[i].getXYZPatchGridSize().x, m_inputStacks[i].getXYZPatchGridSize().y,
      m_inputStacks[i].getXYZPatchGridSize().z), blockSize3);
    EStepKernel<T> << <gridSize3, blockSize3 >> >(m_inputStacks[i], m_m_gpu, m_sigma_gpu, m_mix_gpu);
    CHECK_ERROR(EStepKernel);
    checkCudaErrors(cudaDeviceSynchronize());
  }

  for (int i = 0; i < m_inputStacks.size(); i++)
  {
    unsigned int N = m_inputStacks[i].getXYZPatchGridSize().x*m_inputStacks[i].getXYZPatchGridSize().y;
    for (unsigned int j = 0; j < m_inputStacks[i].getXYZPatchGridSize().z; j++)
    {
      thrust::device_ptr<T> d_w(m_inputStacks[i].getWeigthDataPtr() + (j*N));//w->data());
      thrust::device_ptr<T> d_sw(m_inputStacks[i].getSimWeightsPtr() + (j*N));//sw->data());
      thrust::tuple<T, T> out = thrust::transform_reduce(
        thrust::make_zip_iterator(thrust::make_tuple<thrust::device_ptr<T>, thrust::device_ptr<T> >(d_w, d_sw)),
        thrust::make_zip_iterator(thrust::make_tuple<thrust::device_ptr<T>, thrust::device_ptr<T> >(d_w + N, d_sw + N)),
        transformPatchPotential<T>(), thrust::make_tuple<T, T>(0.0, 0.0), reducePatchPotential<T>());

      if (thrust::get<1>(out) > 0)
      {
        patch_potential[j] = sqrt(thrust::get<0>(out) / thrust::get<1>(out));
      }
      else
      {
        patch_potential[j] = -1; // patch has no unpadded voxels
      }
    }
  }

  //////////////////////////////////////////////////////////////////////
  //CPU part
  //can stay on CPU

  //Todo force-exclude patches predefined by a user, set their potentials to -1
  //for (unsigned int i = 0; i < _force_excluded.size(); i++)
  //  patch_potential[_force_excluded[i]] = -1;

  //TODO
  //exclude patches identified as having small overlap with ROI, set their potentials to -1
  //for (unsigned int i = 0; i < _small_patches.size(); i++)
  //  patch_potential_gpu[_small_patches[i]] = -1;
  T* d_scale_gpu;
  T* d_patch_weight_gpu;
  checkCudaErrors(cudaMalloc(&d_scale_gpu, sizeof(T)*numPatches));
  checkCudaErrors(cudaMalloc(&d_patch_weight_gpu, sizeof(T)*numPatches));
  std::vector<T> h_scale_gpu(numPatches, 0);
  std::vector<T> h_patch_weight_gpu(numPatches, 0);

  unsigned int ofs = 0;
  for (int i = 0; i < m_inputStacks.size(); i++)
  {
    copyFromWeightsAndScales << < divup(m_inputStacks[i].getXYZPatchGridSize().z, 512), 512 >> >(m_inputStacks[i], ofs, d_scale_gpu, d_patch_weight_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
    ofs += m_inputStacks[i].getXYZPatchGridSize().z;
  }

  checkCudaErrors(cudaMemcpy(&h_scale_gpu[0], d_scale_gpu, sizeof(T)*numPatches, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(&h_patch_weight_gpu[0], d_patch_weight_gpu, sizeof(T)*numPatches, cudaMemcpyDeviceToHost));

  //these are unrealistic scales pointing at misregistration - exclude the corresponding patches
  for (inputIndex = 0; inputIndex < patch_potential.size(); inputIndex++)
    if ((h_scale_gpu[inputIndex] < 0.2) || (h_scale_gpu[inputIndex] > 5)) {
      patch_potential[inputIndex] = -1;
    }

  // exclude unrealistic transformations
  if (m_debug) {
    cout << setprecision(4);
    cout << endl << "Patch potentials GPU: ";
    for (inputIndex = 0; inputIndex < patch_potential.size(); inputIndex++)
      cout << patch_potential[inputIndex] << " ";
    cout << endl << "Patch weights GPU: ";
    for (inputIndex = 0; inputIndex < h_patch_weight_gpu.size(); inputIndex++)
      cout << h_patch_weight_gpu[inputIndex] << " ";
    cout << endl << "Patch scales GPU: ";
    for (inputIndex = 0; inputIndex < patch_potential.size(); inputIndex++)
      cout << h_scale_gpu[inputIndex] << " ";
    cout << endl;
  }


  //Calulation of patch-wise robust statistics parameters.
  //This is theoretically M-step,
  //but we want to use latest estimate of patch potentials
  //to update the parameters

  //Calculate means of the inlier and outlier potentials
  double sum = 0, den = 0, sum2 = 0, den2 = 0, maxs = 0, mins = 1;
  for (inputIndex = 0; inputIndex < patch_potential.size(); inputIndex++)
    if (patch_potential[inputIndex] >= 0) {
      //calculate means
      sum += patch_potential[inputIndex] * h_patch_weight_gpu[inputIndex];
      den += h_patch_weight_gpu[inputIndex];
      sum2 += patch_potential[inputIndex] * (1.0 - h_patch_weight_gpu[inputIndex]);
      den2 += (1.0 - h_patch_weight_gpu[inputIndex]);

      //calculate min and max of potentials in case means need to be initalized
      if (patch_potential[inputIndex] > maxs)
        maxs = patch_potential[inputIndex];
      if (patch_potential[inputIndex] < mins)
        mins = patch_potential[inputIndex];
    }

  if (den > 0)
    m_mean_s_gpu = (T)(sum / den);
  else
    m_mean_s_gpu = (T)mins;

  if (den2 > 0)
    m_mean_s2_gpu = (T)(sum2 / den2);
  else
    m_mean_s2_gpu = (T)((maxs + m_mean_s_gpu) / 2.0);

  //Calculate the variances of the potentials
  sum = 0;
  den = 0;
  sum2 = 0;
  den2 = 0;
  for (inputIndex = 0; inputIndex < patch_potential.size(); inputIndex++)
    if (patch_potential[inputIndex] >= 0) {
      sum += (patch_potential[inputIndex] - m_mean_s_gpu) * (patch_potential[inputIndex] - m_mean_s_gpu)
        * h_patch_weight_gpu[inputIndex];
      den += h_patch_weight_gpu[inputIndex];

      sum2 += (patch_potential[inputIndex] - m_mean_s2_gpu) * (patch_potential[inputIndex] - m_mean_s2_gpu)
        * (1 - h_patch_weight_gpu[inputIndex]);
      den2 += (1 - h_patch_weight_gpu[inputIndex]);

    }

  //_sigma_s
  if ((sum > 0) && (den > 0)) {
    m_sigma_s_gpu = (T)(sum / den);
    //do not allow too small sigma
    if (m_sigma_s_gpu < m_step * m_step / 6.28)
      m_sigma_s_gpu = (T)(m_step * m_step / 6.28);
  }
  else {
    m_sigma_s_gpu = 0.025f;
    if (m_debug) {
      if (sum <= 0)
        cout << "All patches are equal. ";
      if (den < 0) //this should not happen
        cout << "All patches are outliers. ";
      cout << "Setting sigma to " << sqrt(m_sigma_s_gpu) << endl;
    }
  }

  //sigma_s2
  if ((sum2 > 0) && (den2 > 0)) {
    m_sigma_s2_gpu = (T)(sum2 / den2);
    //do not allow too small sigma
    if (m_sigma_s2_gpu < m_step * m_step / 6.28)
      m_sigma_s2_gpu = (T)(m_step * m_step / 6.28);
  }
  else {
    m_sigma_s2_gpu = (m_mean_s2_gpu - m_mean_s_gpu) * (m_mean_s2_gpu - m_mean_s_gpu) / 4;
    //do not allow too small sigma
    if (m_sigma_s2_gpu < m_step * m_step / 6.28)
      m_sigma_s2_gpu = (T)(m_step * m_step / 6.28);

    if (m_debug) {
      if (sum2 <= 0)
        cout << "All patches are equal. ";
      if (den2 <= 0)
        cout << "All patches inliers. ";
      cout << "Setting sigma_s2 to " << sqrt(m_sigma_s2_gpu) << endl;
    }
  }

  //Calculate patch weights
  double gs1, gs2;
  for (inputIndex = 0; inputIndex < patch_potential.size(); inputIndex++) {
    //Patch does not have any voxels in volumetric ROI
    if (patch_potential[inputIndex] == -1) {
      h_patch_weight_gpu[inputIndex] = 0;
      continue;
    }

    //All patches are outliers or the means are not valid
    if ((den <= 0) || (m_mean_s2_gpu <= m_mean_s_gpu)) {
      h_patch_weight_gpu[inputIndex] = 1;
      continue;
    }

    //likelihood for inliers
    if (patch_potential[inputIndex] < m_mean_s2_gpu)
      gs1 = G_(patch_potential[inputIndex] - m_mean_s_gpu, m_sigma_s_gpu);
    else
      gs1 = 0;

    //likelihood for outliers
    if (patch_potential[inputIndex] > m_mean_s_gpu)
      gs2 = G_(patch_potential[inputIndex] - m_mean_s2_gpu, m_sigma_s2_gpu);
    else
      gs2 = 0;

    //calculate patch weight
    double likelihood = gs1 * m_mix_s_gpu + gs2 * (1 - m_mix_s_gpu);
    if (likelihood > 0)
      h_patch_weight_gpu[inputIndex] = (T)(gs1 * m_mix_s_gpu / likelihood);
    else {
      if (patch_potential[inputIndex] <= m_mean_s_gpu)
        h_patch_weight_gpu[inputIndex] = 1;
      if (patch_potential[inputIndex] >= m_mean_s2_gpu)
        h_patch_weight_gpu[inputIndex] = 0;
      if ((patch_potential[inputIndex] < m_mean_s2_gpu) && (patch_potential[inputIndex] > m_mean_s_gpu)) //should not happen
        h_patch_weight_gpu[inputIndex] = 1;
    }
  }

  //Update _mix_s this should also be part of MStep
  sum = 0;
  int num = 0;
  for (inputIndex = 0; inputIndex < patch_potential.size(); inputIndex++)
    if (patch_potential[inputIndex] >= 0) {
      sum += h_patch_weight_gpu[inputIndex];
      num++;
    }

  if (num > 0)
    m_mix_s_gpu = (T)(sum / num);
  else {
    cout << "All patches are outliers. Setting _mix_s to 0.9." << endl;
    m_mix_s_gpu = 0.9f;
  }

  if (m_debug) {
    cout << setprecision(3);
    cout << "Patch robust statistics parameters GPU: ";
    cout << "means: " << m_mean_s_gpu << " " << m_mean_s2_gpu << "  ";
    cout << "sigmas: " << sqrt(m_sigma_s_gpu) << " " << sqrt(m_sigma_s2_gpu) << "  ";
    cout << "proportions: " << m_mix_s_gpu << " " << 1 - m_mix_s_gpu << endl;
    cout << "Patch weights GPU: ";
    for (inputIndex = 0; inputIndex < h_patch_weight_gpu.size(); inputIndex++)
      cout << h_patch_weight_gpu[inputIndex] << " ";
    cout << endl;
  }

  //update GPU patches
  checkCudaErrors(cudaMemcpy(d_scale_gpu, &h_scale_gpu[0], sizeof(T)*numPatches, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_patch_weight_gpu, &h_patch_weight_gpu[0], sizeof(T)*numPatches, cudaMemcpyHostToDevice));

  ofs = 0;
  for (int i = 0; i < m_inputStacks.size(); i++)
  {
    copyToWeightsAndScales << < divup(m_inputStacks[i].getXYZPatchGridSize().z, 512), 512 >> >(m_inputStacks[i], ofs, d_scale_gpu, d_patch_weight_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
    ofs += m_inputStacks[i].getXYZPatchGridSize().z;
  }

  checkCudaErrors(cudaFree(d_scale_gpu));
  checkCudaErrors(cudaFree(d_patch_weight_gpu));
  cout.rdbuf(strm_buffer);
  cerr.rdbuf(strm_buffer_e);
}


template <typename T>
struct transformMStep3DNoBias
{
  __host__ __device__
    thrust::tuple<T, T, T, T, T> operator()(const thrust::tuple<T, T, T, T, T>& v)
    //thrust::tuple<sigma_, mix_, count, e, e> //this order is very important for the thrust optimization
  {
    const T s_ = thrust::get<0>(v);
    const T w_ = thrust::get<1>(v);
    const T ss_ = thrust::get<2>(v);
    const T sw_ = thrust::get<3>(v);
    const T scale = thrust::get<4>(v);

    T sigma_ = 0.0f;
    T mix_ = 0.0f;
    T count = 0.0f;
    T e = 0.0f;

    thrust::tuple<T, T, T, T, T> t;

    if (s_ != -1.0f  && sw_ > 0.99f)
    {
      e = (s_ * scale) - ss_;
      sigma_ = e * e * w_;
      mix_ = w_;
      count = 1.0;
      T e1 = e;
      t = thrust::make_tuple(sigma_, mix_, count, e, e1);
    }
    else
    {
      t = thrust::make_tuple(0.0, 0.0, 0.0, FLT_MAX, FLT_MIN);
    }
    return t;
  }
};

template <typename T>
struct reduceMStep
{
  __host__ __device__
    thrust::tuple<T, T, T, T, T> operator()(const thrust::tuple<T, T, T, T, T>& a,
    const thrust::tuple<T, T, T, T, T>& b)
  {
    return thrust::make_tuple(thrust::get<0>(a)+thrust::get<0>(b), thrust::get<1>(a)+thrust::get<1>(b),
      thrust::get<2>(a)+thrust::get<2>(b), min(thrust::get<3>(a), thrust::get<3>(b)),
      max(thrust::get<4>(a), thrust::get<4>(b)));
  }
};


template <typename T>
__global__ void initBufferWithScales(PatchBasedVolume<T> inputStack)
{
  const uint3 pos = make_uint3(blockIdx.x* blockDim.x + threadIdx.x,
    blockIdx.y* blockDim.y + threadIdx.y,
    blockIdx.z* blockDim.z + threadIdx.z);

  uint3 vSize = inputStack.getXYZPatchGridSize();

  if (pos.x >= vSize.x || pos.y >= vSize.y || pos.z >= vSize.z)
    return;

  inputStack.setBufferValue(pos, inputStack.getImagePatch2D(pos.z).scale);
}

template <typename T>
void patchBasedRobustStatistics_gpu<T>::MStep(int iter)
{
  printf("MStep_gpu\n");
  cerr.rdbuf(file_e.rdbuf());
  cout.rdbuf(file.rdbuf());

  T num = 0;
  T min_ = std::numeric_limits<T>::max();
  T max_ = std::numeric_limits<T>::min();
  T sigma = 0;
  T mix = 0;
  
  thrust::tuple<T, T, T, T, T> results;
  for (int i = 0; i < m_inputStacks.size(); i++)
  {

    thrust::device_ptr<T> d_s(m_inputStacks[i].getPatchesPtr());
    thrust::device_ptr<T> d_w(m_inputStacks[i].getWeigthDataPtr());
    thrust::device_ptr<T> d_ss(m_inputStacks[i].getSimPatchesPtr());
    thrust::device_ptr<T> d_sw(m_inputStacks[i].getSimWeightsPtr());
    thrust::device_ptr<T> d_buf(m_inputStacks[i].getBufferPtr());

    unsigned int N1 = m_inputStacks[i].getXYZPatchGridSize().x*m_inputStacks[i].getXYZPatchGridSize().y;
    
    //thrust::constant_iterator<int> first(h_scales[j]);
    for (unsigned int ii = 0; ii < m_inputStacks.size(); ii++)
    {
      dim3 blockSize3 = dim3(8, 8, 8);
      dim3 gridSize3 = divup(dim3(m_inputStacks[ii].getXYZPatchGridSize().x, m_inputStacks[ii].getXYZPatchGridSize().y,
        m_inputStacks[ii].getXYZPatchGridSize().z), blockSize3);
      initBufferWithScales<T> << <gridSize3, blockSize3 >> >(m_inputStacks[ii]);
      CHECK_ERROR(initBufferWithScales);
      checkCudaErrors(cudaDeviceSynchronize());
    }

    unsigned int N3 = m_inputStacks[i].getXYZPatchGridSize().x*m_inputStacks[i].getXYZPatchGridSize().y*m_inputStacks[i].getXYZPatchGridSize().z;

    results = thrust::transform_reduce(thrust::make_zip_iterator(thrust::make_tuple(d_s, d_w, d_ss, d_sw, d_buf)),
      thrust::make_zip_iterator(thrust::make_tuple(d_s + N3, d_w + N3, d_ss + N3, d_sw + N3, d_buf + N3)), transformMStep3DNoBias<T>(),
      thrust::make_tuple<T, T, T, T, T>(0.0, 0.0, 0.0, 0.0, 0.0), reduceMStep<T>());

    sigma += get<0>(results);
    mix += get<1>(results);
    num += get<2>(results);
    min_ = min(min_, get<3>(results));
    max_ = max(max_, get<4>(results));
  }

  if (mix > 0) {
    m_sigma_gpu = sigma / mix;
  }
  else {
    printf("Something went wrong: sigma= %f mix= %f\n", sigma, mix);
    //exit(1);
  }
  if (m_sigma_gpu < m_step * m_step / 6.28f)
    m_sigma_gpu = m_step * m_step / 6.28f;
  if (iter > 1)
    m_mix_gpu = mix / num;

  //Calculate m
  m_m_gpu = 1.0f / (max_ - min_);

  std::cout.precision(10);
  if (m_debug) {
    cout << "Voxel-wise robust statistics parameters GPU: ";
    cout << "sigma = " << sqrt(m_sigma_gpu) << " mix = " << m_mix_gpu << " ";
    cout << " m = " << m_m_gpu << endl;
  }
  

  cout.rdbuf(strm_buffer);
  cerr.rdbuf(strm_buffer_e);
}

template <typename T>
struct transformScalenoBias
{
  transformScalenoBias(){}

  __host__ __device__
    thrust::tuple<T, T> operator()(const thrust::tuple<T, T, T, T>& v)
  {
    T s_ = thrust::get<0>(v);
    const T w_ = thrust::get<1>(v);
    const T ss_ = thrust::get<2>(v);
    const T sw_ = thrust::get<3>(v);

    if ((s_ == -1.0f) || sw_ <= 0.99f)
    {
      return thrust::make_tuple(0.0f, 0.0f);
    }
    else
    {
      T scalenum = w_ * s_ * ss_;
      T scaleden = w_ * s_ * s_;
      return thrust::make_tuple(scalenum, scaleden);
    }
  }
};

template <typename T>
struct reduceScale
{
  reduceScale(){}

  __host__ __device__
    thrust::tuple<T, T> operator()(const thrust::tuple<T, T>& a, const thrust::tuple<T, T>& b)
  {
    return thrust::make_tuple(thrust::get<0>(a)+thrust::get<0>(b), thrust::get<1>(a)+thrust::get<1>(b));
  }
};

template <typename T>
void patchBasedRobustStatistics_gpu<T>::Scale()
{
  printf("Scale_gpu\n");
  cerr.rdbuf(file_e.rdbuf());
  cout.rdbuf(file.rdbuf());

  std::vector<T> scale_vec;

  //TODO reduce this on GPU
  for (int i = 0; i < m_inputStacks.size(); i++)
  {
    unsigned int N = m_inputStacks[i].getXYZPatchGridSize().x*m_inputStacks[i].getXYZPatchGridSize().y;
    for (int j = 0; j < m_inputStacks[i].getXYZPatchGridSize().z; j++)
    {
      thrust::device_ptr<T> d_s(m_inputStacks[i].getPatchesPtr() + (j*N));
      thrust::device_ptr<T> d_w(m_inputStacks[i].getWeigthDataPtr() + (j*N));
      thrust::device_ptr<T> d_ss(m_inputStacks[i].getSimPatchesPtr() + (j*N));
      thrust::device_ptr<T> d_sw(m_inputStacks[i].getSimWeightsPtr() + (j*N));

      thrust::tuple<T, T> out = thrust::make_tuple<T, T>(0.0f, 0.0f);

      out = thrust::transform_reduce(
        thrust::make_zip_iterator(thrust::make_tuple<thrust::device_ptr<T>, thrust::device_ptr<T>, thrust::device_ptr<T>, thrust::device_ptr<T> >(d_s, d_w, d_ss, d_sw)),
        thrust::make_zip_iterator(thrust::make_tuple<thrust::device_ptr<T>, thrust::device_ptr<T>, thrust::device_ptr<T>, thrust::device_ptr<T> >(d_s + N, d_w + N, d_ss + N, d_sw + N)), transformScalenoBias<T>(),
        thrust::make_tuple<T, T>(0.0, 0.0), reduceScale<T>());

      if (thrust::get<1>(out) != 0.0)
      {
        scale_vec.push_back(thrust::get<0>(out) / thrust::get<1>(out));
        //scale_vec[j] = thrust::get<0>(out) / thrust::get<1>(out);
      }
      else
      {
        scale_vec.push_back(1.0);
        //scale_vec[j] = 1.0;
      }

    }
  }

  unsigned int numPatches = 0;
  for (int i = 0; i < m_inputStacks.size(); i++)
  {
    numPatches += m_inputStacks[i].getXYZPatchGridSize().z;
  }

  T* d_scale_gpu;
  checkCudaErrors(cudaMalloc(&d_scale_gpu, sizeof(T)*numPatches));
  checkCudaErrors(cudaMemcpy(d_scale_gpu, &scale_vec[0], sizeof(T)*numPatches, cudaMemcpyHostToDevice));

  //copyToPatches
  unsigned int ofs = 0;
  for (int i = 0; i < m_inputStacks.size(); i++)
  {
    copyToScales << < divup(m_inputStacks[i].getXYZPatchGridSize().z, 512), 512 >> >(m_inputStacks[i], ofs, d_scale_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
    ofs += m_inputStacks[i].getXYZPatchGridSize().z;
  }

  if (m_debug ) {
    cout << setprecision(3);
    cout << "Patch scale GPU = ";
    for (unsigned int inputIndex = 0; inputIndex < scale_vec.size(); ++inputIndex)
      cout << inputIndex << ":" << scale_vec[inputIndex] << " ";
    cout << endl;
    }

  checkCudaErrors(cudaFree(d_scale_gpu));
  cout.rdbuf(strm_buffer);
  cerr.rdbuf(strm_buffer_e);
}

template <typename T>
struct transformRS
{
  __host__ __device__
    thrust::tuple<T, unsigned int> operator()(const thrust::tuple<T, char, T, T>& v)
  {
    T s_ = thrust::get<0>(v);
    char si_ = thrust::get<1>(v);
    T ss_ = thrust::get<2>(v);
    T sw_ = thrust::get<3>(v);

    thrust::tuple<T, unsigned int> t = thrust::make_tuple(0.0f, 0);

    if (s_ != -1 && si_ == 1 && sw_ > 0.99)
    {
      T sval = s_ - ss_;
      t = thrust::make_tuple<T, unsigned int>(sval*sval, 1);
    }
    return t; // return t <sigma,num>
  }
};

template <typename T>
struct reduceRS
{
  __host__ __device__
    thrust::tuple<T, unsigned int> operator()(const thrust::tuple<T, unsigned int>& a,
    const thrust::tuple<T, unsigned int>& b)
  {
    return thrust::make_tuple<T, unsigned int>(thrust::get<0>(a)+thrust::get<0>(b), thrust::get<1>(a)+thrust::get<1>(b));
  }
};



template <typename T>
struct square
{
  __host__ __device__
    thrust::tuple<T, unsigned int> operator()(const thrust::tuple<T, unsigned int>& a,
    const thrust::tuple<T, unsigned int>& b)
  {
    return make_tuple<T, unsigned int>(thrust::get<0>(a)+thrust::get<0>(b), thrust::get<1>(a)+thrust::get<1>(b));
  }
};

template <typename T>
void patchBasedRobustStatistics_gpu<T>::InitializeRobustStatistics(T _min_intensity, T _max_intensity, int cuda_device)
{ 
  //TODO
  //if patch does not have an overlap with ROI, set its weight to zero
  //Force exclusion of patches predefined by user

  T sa = 0;
  T sb = 0;

  m_cuda_device = cuda_device;
  checkCudaErrors(cudaSetDevice(m_cuda_device));
  
  for (int i = 0; i < m_inputStacks.size(); i++)
  {
    thrust::device_ptr<T> d_s(m_inputStacks[i].getPatchesPtr());
    thrust::device_ptr<char> d_si(m_inputStacks[i].getSimInsidePtr());
    thrust::device_ptr<T> d_ss(m_inputStacks[i].getSimPatchesPtr());
    thrust::device_ptr<T> d_sw(m_inputStacks[i].getSimWeightsPtr());

    unsigned int N = m_inputStacks[i].getXYZPatchGridSize().x*m_inputStacks[i].getXYZPatchGridSize().y*
      m_inputStacks[i].getXYZPatchGridSize().z;

    //thrust::make_zip_iterator(thrust::make_tuple(d_s, d_si, d_ss, d_sw));
    //thrust::make_zip_iterator(thrust::make_tuple(d_s + N, d_si + N, d_ss + N, d_sw + N));

    
    //thrust::transform_reduce(InputIterator first, InputIterator last, UnaryFunction unary_op, OutputType init, BinaryFunction binary_op)
    thrust::tuple<T, unsigned int> out = thrust::transform_reduce(
      thrust::make_zip_iterator(thrust::make_tuple<thrust::device_ptr<T>, thrust::device_ptr<char>, thrust::device_ptr<T>, thrust::device_ptr<T> >(d_s, d_si, d_ss, d_sw)),
      thrust::make_zip_iterator(thrust::make_tuple<thrust::device_ptr<T>, thrust::device_ptr<char>, thrust::device_ptr<T>, thrust::device_ptr<T> >(d_s + N, d_si + N, d_ss + N, d_sw + N)), 
      transformRS<T>(),
      thrust::make_tuple<T, unsigned int>(0.0, 0), 
      reduceRS<T>());

    sa += get<0>(out);
    sb += (T)get<1>(out);

  }
  
  // printf("sa = %f - sb = %f\n",sa,sb);

  if (sb == 0)
  {
    printf("ERROR: sb = 0!! no sigma computed! exiting!\n");
    exit(-1);
  }
  //initialize sigma for inlier voxel errors
  m_sigma_gpu = sa / sb;
  //initialize sigma for patch-wise robust statistics
  m_sigma_s_gpu = 0.025f;
  //initialize mixing proportion for inlier class in voxel-wise robust statistics (correctly matched voxels)
  m_mix_gpu = 0.9f;
  //initialize mixing proportion for outlier class in patch-wise robust statistics
  m_mix_s_gpu = 0.9f;
  //Initialize value for uniform distribution according to the range of intensities
  m_m_gpu = (T)(1.0f / (2.1f * _max_intensity - 1.9f * _min_intensity));

  if (m_debug)
  {
    std::cout << "Initializing robust statistics GPU: " << "sigma=" << sqrt(m_sigma_gpu) << " " << "m=" << m_m_gpu
      << " " << "mix=" << m_mix_gpu << " " << "mix_s=" << m_mix_s_gpu << std::endl;
  }
}



template <typename T>
patchBasedRobustStatistics_gpu<T>::patchBasedRobustStatistics_gpu(std::vector<PatchBasedVolume<T> > & _inputStacks) :
m_inputStacks(_inputStacks), m_debug(false)
{

  m_step = 0.0001;
  strm_buffer = cout.rdbuf();
  strm_buffer_e = cerr.rdbuf();
  file.open("log-EM.txt");
  file_e.open("log-EM-error.txt");
}

template <typename T>
void patchBasedRobustStatistics_gpu<T>::updateInputStacks(std::vector<PatchBasedVolume<T> > & _inputStacks)
{
  m_inputStacks.clear();
  m_inputStacks = _inputStacks;
}

template <typename T>
patchBasedRobustStatistics_gpu<T>::~patchBasedRobustStatistics_gpu()
{

}

template class patchBasedRobustStatistics_gpu < float > ;
template class patchBasedRobustStatistics_gpu < double > ;