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
#ifndef RECONSTRUCTION_CUDA_CUH
#define RECONSTRUCTION_CUDA_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <cufft.h>

#include <thrust/version.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/functional.h>
#include <thrust/advance.h>
#include <thrust/tuple.h>
#include <thrust/count.h>
#include "recon_volumeHelper.cuh"

#include "GPUWorker.h"
/////////////////////////////////////////////////
//configuration section
#define __step  0.0001f
//only for experiments. Real PSF is continous
#define PSF_SIZE 128

//usually the kernel runtime watchdog is activated in windows -- workaround:
#if WIN32 
#define MAX_SLICES_PER_RUN_GAUSS 32
#define MAX_SLICES_PER_RUN 128
#else
#define MAX_SLICES_PER_RUN_GAUSS 1000
#define MAX_SLICES_PER_RUN 5000
#endif
//maximum number of GPUs running in parallel
#define MAX_GPU_COUNT 32 

//use a sinc in-plane gauss through plane PSF 
#define USE_SINC_PSF 1

#define PSF_EPSILON 0.00001
#define USE_INFINITE_PSF_SUPPORT 1
#define MAX_PSF_SUPPORT 16
//configuration section end
/////////////////////////////////////////////////

#if 1
#define CHECK_ERROR(function) if (cudaError_t err = cudaGetLastError()) \
{ \
  printf("CUDA Error in " #function "(), line %d: %s\n", __LINE__, cudaGetErrorString(err)); \
  exit(-err); \
}
#else
#define CHECK_ERROR(function)
#endif

inline int divup(int a, int b) { return (a + b - 1) / b; }
inline dim3 divup(uint2 a, dim3 b) { return dim3(divup(a.x, b.x), divup(a.y, b.y)); }
inline dim3 divup(dim3 a, dim3 b) { return dim3(divup(a.x, b.x), divup(a.y, b.y), divup(a.z, b.z)); }

class Reconstruction {
public:
  Reconstruction(std::vector<int> dev, bool multiThreadedGPU = true);
  ~Reconstruction();

  Volume<float> maskC_;

  //needed for stack-wise processing
  Volume<float> v_slices;

  //Multi-GPU
  std::vector< Volume<float> > dev_v_slices;
  std::vector< Volume<float> > dev_v_bias;
  std::vector< Volume<float> > dev_v_weights;
  std::vector< Volume<float> > dev_v_simulated_weights;
  std::vector< Volume<float> > dev_v_simulated_slices;
  std::vector< Volume<float> > dev_v_wresidual;
  std::vector< Volume<float> > dev_v_wb;
  std::vector< Volume<float> > dev_v_buffer;
  std::vector< Volume<char> > dev_v_simulated_inside;
  std::vector< Volume<int> > dev_sliceVoxel_count_;
  std::vector< Volume<float> > dev_v_PSF_sums_;

  std::vector<uint2> dev_slice_range_offset;
  std::vector<int*> dev_d_slice_sicesX;
  std::vector<int*> dev_d_slice_sicesY;

  std::vector<Matrix4*> dev_d_slicesI2W;
  std::vector<Matrix4*> dev_d_slicesW2I;
  std::vector<Matrix4*> dev_d_slicesTransformation;
  std::vector<Matrix4*> dev_d_slicesInvTransformation;

  std::vector< Volume<float> > dev_reconstructed_;
  std::vector< Volume<float> > dev_reconstructed_volWeigths;
  std::vector< Volume<float> > dev_mask_;
  std::vector< Volume<float> > dev_addon_;
  std::vector< Volume<float> > dev_bias_;
  std::vector< Volume<float> > dev_confidence_map_;
  std::vector< Volume<float> > dev_volume_weights_;

  //std::vector< Volume<float> > dev_reconRegUncertainty_;
  //std::vector< Volume<float> > dev_scaleSamplingUncertainty_;

  std::vector<float*> dev_d_scales;
  std::vector<float*> dev_d_slice_weights;

  std::vector<float3*> dev_d_sliceDims;
  std::vector<int*> dev_d_sliceDim;

  cudaStream_t streams[MAX_GPU_COUNT];
  //Multi-GPU end

  bool d_slice_sized_allocated;
  bool d_scale_weight_allocated;
  bool d_sliceMatrices_allocated;
  bool d_sliceDims_allocated;

  float reconstructedVoxelSize;

  std::vector<float> h_scales;
  std::vector<float> h_slices_weights;

  std::vector<uint3> stack_sizes;
  uint3 PSFsize;

  std::vector<int> slices_per_device;

  unsigned int num_slices_;

  std::vector<int> devicesToUse;

  bool _useCPUReg;
  bool _debugGPU;

  GPUWorkerSync GPU_sync;
#if !USE_BOOST
  std::vector<std::shared_ptr<std::thread> > GPU_threads;
  std::vector<std::shared_ptr<GPUWorkerCommunicator> > GPU_workers;
#else
  bc::vector<boost::shared_ptr<boost::thread> > GPU_threads;
  bc::vector<boost::shared_ptr<GPUWorkerCommunicator> > GPU_workers;
#endif
  bool multiThreadedGPU;
  void startThread(int i);

  //debugFuctions
  //only for debugging
  Volume<int> sliceVoxel_count_;
  Volume<float> v_PSF_sums_;
  Volume<float> v_bias;
  Volume<float> v_weights;
  Volume<float> v_simulated_weights;
  Volume<float> v_simulated_slices;
  Volume<char> v_simulated_inside;
  Volume<float> v_wresidual;
  Volume<float> v_wb;
  Volume<float> v_buffer;

  void combineWeights(float* weights);

  void debugWeights(float* weights);
  void debugBias(float* bias);
  void debugNormalizeBias(float* nbias);
  void debugSmoothMask(float* smoothMask);
  void debugSimslices(float* simslices);
  void debugSimweights(float* simweights);
  void debugConfidenceMap(float* cmap);
  void debugAddon(float* addon);
  void debugSiminside(char* siminside);
  void debugv_PSF_sums(float* v_PSF_sums);
  void testCPUReg(std::vector<Matrix4>& transf_);
  void getSlicesVol_debug(float* h_imdata);
  void getRegSlicesVol_debug(float* h_imdata);
  void syncGPUrecon(float* reconstructed);
  void getVolWeights(float* weights);
  void debugRegSlicesVolume(float* regSlicesHost);

  //init/update functions
  void updateStackSizes(std::vector<uint3> stack_sizes_){ stack_sizes = stack_sizes_; };
  void InitializeEMValues();
  void InitializeEMValuesOnX(int dev);
  void syncCPU(float* reconstructed);
  void initStorageVolumes(uint3 size, float3 dim);
  void initStorageVolumesOnX(uint3 size, float3 dim, int start, int end, int dev);
  void FillSlices(float* sdata, std::vector<int> sizesX, std::vector<int> sizesY);
  void FillSlicesOnX(float* sdata, std::vector<int>& sizesX, std::vector<int>& sizesY, float* d_ldata, float * ldata, int dev, bool alloc);
  void generatePSFVolume(float* CPUPSF, uint3 PSFsize_, float3 sliceVoxelDim, float3 PSFdim, Matrix4 PSFI2W, Matrix4 PSFW2I, float _quality_factor);
  void generatePSFVolumeOnX(float* CPUPSF, uint3 PSFsize_, float3 sliceVoxelDim, float3 PSFdim, Matrix4 PSFI2W, Matrix4 PSFW2I, float _quality_factor, int dev);
  void setSliceDims(std::vector<float3> slice_dims, float _quality_factor);
  void setSliceDimsOnX(std::vector<float3>& slice_dims, std::vector<int>& sliceDim, bool allocate, int dev);
  void SetSliceMatrices(std::vector<Matrix4> matSliceTransforms, std::vector<Matrix4> matInvSliceTransforms,
    std::vector<Matrix4>& matsI2Winit, std::vector<Matrix4>& matsW2Iinit, std::vector<Matrix4>& matsI2W, std::vector<Matrix4>& matsW2I,
    Matrix4 reconI2W, Matrix4 reconW2I);
  void SetSliceMatricesOnX(std::vector<Matrix4> matSliceTransforms, std::vector<Matrix4> matInvSliceTransforms,
    std::vector<Matrix4>& matsI2Winit, std::vector<Matrix4>& matsW2Iinit, std::vector<Matrix4>& matsI2W,
    std::vector<Matrix4>& matsW2I, Matrix4 reconI2W, Matrix4 reconW2I, int dev, bool alloc);
  void UpdateSliceWeights(std::vector<float> slices_weights);
  void UpdateSliceWeightsOnX(std::vector<float>& slices_weights, int dev);
  void InitReconstructionVolume(uint3 s, float3 dim, float* data, float sigma_bias); // allocates the volume and image data on the device
  void InitReconstructionVolumeOnX(uint3 s, float3 dim, float* data, float sigma_bias, int dev);
  void InitSlices(int num_slices);
  void UpdateReconstructed(const uint3 vsize, float* data); //temporary for CPU GPU sync
  void UpdateReconstructedOnX(const uint3 vsize, float* data, int dev);
  void SyncConfidenceMapAddon(float* cmdata, float* addondata);
  void UpdateScaleVector(std::vector<float> scales, std::vector<float> slices_weights);//{h_scales = scales; /*thrust::copy(scales.begin(), scales.end(), scale_.begin());*/};
  void UpdateScaleVectorOnX(std::vector<float>& scales, std::vector<float>& slices_weights, int dev, bool alloc);
  void CalculateScaleVector(std::vector<float>& scale_vec);
  void CalculateScaleVectorOnX(std::vector<float>& scale_vec, int dev);
  void setMask(uint3 s, float3 dim, float* data, float sigma_bias);
  void setMaskOnX(uint3 s, float3 dim, float* data, float sigma_bias, int dev);
  void cleanUpOnX(int dev);
  void setPachBased(bool value){ _patchBased = value; };

  //--------------------------------------------------------------------------------------------
  // superpixel (spx)
  void setSupepixelBased(bool value){ _superpixelBased = value; };
  //--------------------------------------------------------------------------------------------

  //calculation functions
  void NormaliseBias(int iter, float sigma_bias);
  void NormaliseBiasOnX(int iter, Volume<float>& dev_bias_accbuf_, Volume<float>& dev_volume_weights_accbuf_, float sigma_bias, int dev);
  void EStep(float _m, float _sigma, float _mix, std::vector<float>& slice_potential);
  void EStepOnX(float _m, float _sigma, float _mix, std::vector<float>& slice_potential, float* d_lweigthsdata, int dev);
  void MStep(int iter, float _step, float& _sigma, float& _mix, float& _m);
  void MStepOnX(thrust::tuple<float, float, float, float, float> &results, int dev);
  void SimulateSlices(std::vector<bool>& slice_inside);
  void SimulateSlicesOnX(std::vector<bool>::iterator slice_inside, int dev);
  void SyncSimulatedCPU(int slice_idx, float* ssdata, float* swdata, float* sidata);
  void InitializeRobustStatistics(float& _sigma);
  void CorrectBias(float sigma_bias, bool _global_bias_correction);
  void CorrectBiasOnX(float sigma_bias, bool _global_bias_correction, float* d_lbiasdata, int dev);
  void Superresolution(int iter, std::vector<float> _slice_weight, bool _adaptive, float alpha,
    float _min_intensity, float _max_intensity, float delta, float lambda, bool _global_bias_correction, float sigma_bias,
    float _low_intensity_cutoff);
  void SuperresolutionOnX1(int N, Volume<float>& dev_addon_accbuf_, Volume<float>& dev_cmap_accbuf_, Volume<float>& original, int dev);

  void transformManualMaskGPU(float* manualMaskdata, unsigned int numSlices, uint2 sliceSize, float* outdata);

  void maskVolume();
  void ScaleVolume();
  void RestoreSliceIntensities(std::vector<float> stack_factors_, std::vector<int> stack_index_);

  void GaussianReconstruction(std::vector<int>& voxel_num);
  void GaussianReconstructionOnX1(int& voxel_num, Volume<float>& dev_reconstructed_accbuf_, int dev);
  void GaussianReconstructionOnX2(int& voxel_num, int dev);
  typedef enum { TX, TY, TZ, RX, RY, RZ, SX, SY, SZ, SXY, SYZ, SXZ, SYX, SZY, SZX }
  irtkHomogeneousTransformationParameterIndex;
  void Matrix2Parameters(Matrix4 m, float* params);
  Matrix4 Parameters2Matrix(float *params);

  void SetDevicesToUse(std::vector<int>& dev);

  //registration
  bool d_sliceResMatrices_allocated;
  bool dev_d_slicesOfs_allocated;
  bool regStorageInit;
  std::vector< Matrix4* > dev_d_slicesResampledI2W;
  std::vector< Matrix4* > dev_d_slicesOfs;

  std::vector< LayeredSurface3D<float> > dev_regSlices;
  std::vector< LayeredSurface3D<float> > dev_v_slices_resampled;
  std::vector< LayeredSurface3D<float> > dev_v_slices_resampled_float;
  std::vector< LayeredSurface3D<float> > dev_temp_slices;
  std::vector< Matrix4* > dev_recon_matrices;
  std::vector< Matrix4* > dev_recon_matrices_orig;
  std::vector< float* > dev_recon_params;
  std::vector< float* > dev_recon_similarities;
  std::vector< float* > dev_recon_gradient;
  std::vector< int* > dev_active_slices, dev_active_slices2, dev_active_slices_prev;
  std::vector< float* > dev_temp_float;
  std::vector< int* > dev_temp_int;

  //TODO need them to copy back for debug
  Volume<float> regSlices;
  Volume<float> v_slices_resampled;

  std::vector< cudaArray* > dev_reconstructed_array;
  bool reconstructed_arrays_init;

  float* _Blurring;
  float* _LengthOfSteps;
  float _Epsilon;
  int _NumberOfLevels;
  int _NumberOfSteps;
  int _NumberOfIterations;
  float h_quality_factor;
  bool _patchBased;
  bool _disableBiasC;

  //--------------------------------------------------------------------------------------------
  // superpixel (spx)
  bool _superpixelBased;
  //--------------------------------------------------------------------------------------------

  void initRegStorageVolumes(uint3 size, float3 dim);
  void initRegStorageVolumesOnX(uint3 size, float3 dim, bool init, int dev);
  void FillRegSlices(float* sdata, std::vector<Matrix4> slices_resampledI2W);
  void FillRegSlicesOnX(float* sdata, std::vector<Matrix4>& slices_resampledI2W, bool allocate, int dev);

  void updateResampledSlicesI2W(std::vector<Matrix4> ofsSlice);
  void updateResampledSlicesI2WOnX(std::vector<Matrix4>& ofsSlice, bool allocate, int dev);
  void registerSlicesToVolume(std::vector<Matrix4>& transf_);
  void registerSlicesToVolumeOnX1(int global_slice, std::vector<Matrix4>& h_trans, int dev);
  void registerSlicesToVolumeOnX2(int dev);
  void registerMultipleSlicesToVolume(std::vector<Matrix4>& transf_, int global_slice, int dev_slices, int dev = 0);
  void evaluateCostsMultipleSlices(int active_slices, int slices, int level, float targetBlurring, int writeoffset, int writestep, int writenum, int dev = 0);
  void prepareSliceToVolumeReg();
  void prepareSliceToVolumeRegOnX(bool allocate, int dev);

};



#endif //RECONSTRUCTION_CUDA_CUH