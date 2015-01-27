/*=========================================================================
Library   : Image Registration Toolkit (IRTK)
Module    : $Id: irtkReconstructionCuda.cc 1 2013-11-15 14:36:30 bkainz $
Copyright : Imperial College, Department of Computing
Visual Information Processing (VIP), 2011 onwards
Date      : $Date: 2013-11-15 14:36:30 +0100 (Fri, 15 Nov 2013) $
Version   : $Revision: 1 $
Changes   : $Author: bkainz $

Copyright (c) 2014, Bernhard Kainz, Markus Steinberger,
Maria Murgasova, Kevin Keraudren
All rights reserved.

If you use this work for research we would very much appreciate if you
could cite
TODO: update as soon as accepted
Bernhard Kainz, Markus Steinberger, Christina Malamateniou, Wolfgang Wein, Maria Murgasova,
Kevin Keraudren, Thomas Torsney-Weir, Mary Rutherford, Joseph  V. Hajnal, and Daniel Rueckert:
Generalized Fast Volume Reconstruction from Motion Corrupted 2D Slices.
IEEE Transactions on Medical Imaging, under review, 2014

IRTK IS PROVIDED UNDER THE TERMS OF THIS CREATIVE
COMMONS PUBLIC LICENSE ("CCPL" OR "LICENSE"). THE WORK IS PROTECTED BY
COPYRIGHT AND/OR OTHER APPLICABLE LAW. ANY USE OF THE WORK OTHER THAN
AS AUTHORIZED UNDER THIS LICENSE OR COPYRIGHT LAW IS PROHIBITED.

BY EXERCISING ANY RIGHTS TO THE WORK PROVIDED HERE, YOU ACCEPT AND AGREE
TO BE BOUND BY THE TERMS OF THIS LICENSE. TO THE EXTENT THIS LICENSE MAY BE
CONSIDERED TO BE A CONTRACT, THE LICENSOR GRANTS YOU THE RIGHTS CONTAINED
HERE IN CONSIDERATION OF YOUR ACCEPTANCE OF SUCH TERMS AND CONDITIONS.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
=========================================================================*/

#ifndef GPUWORKER_CUDA_CUH
#define GPUWORKER_CUDA_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <vector_types.h>
#include <vector_functions.h>

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
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>
#include <thrust/count.h>
#include "recon_volumeHelper.cuh"

#if !USE_BOOST
//use c++11 in std in case of VS 2012
#include <thread>
#include <condition_variable>
#else
#include <boost/make_shared.hpp>
#include <boost/thread.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/ref.hpp>
#include <boost/thread/condition_variable.hpp>
#include <boost/container/vector.hpp>
namespace bc = boost::container;
#endif

struct Reconstruction;

class GPUWorkerSync
{
#if !USE_BOOST
  std::mutex mutexController,
    mutexWorker;
  std::condition_variable conditionController,
    conditionWorker;
#else
  boost::mutex mutexController,
    mutexWorker;
  boost::condition_variable conditionController,
    conditionWorker;
#endif

  int GPU_count;
  volatile int count;
public:
  GPUWorkerSync() : GPU_count(0), count(0) { }
  void completed();
  void runNextRound();
  void runNoSync();
  template<class FLaunchData>
  void startup(int toLaunch, void(*starter)(int, FLaunchData&), FLaunchData& data)
  {
#if !USE_BOOST
    std::unique_lock<std::mutex> lockController(mutexController);
    {
      std::unique_lock<std::mutex> lockworker(mutexWorker);
      GPU_count = toLaunch;
      count = 0;
      for (int i = 0; i < toLaunch; ++i)
      {
        starter(i, data);
      }
    }
#else
    boost::unique_lock<boost::mutex> lockController(mutexController);
    {
      boost::unique_lock<boost::mutex> lockworker(mutexWorker);
      GPU_count = toLaunch;
      count = 0;
      for (int i = 0; i < toLaunch; ++i)
      {
        starter(i, data);
      }
    }

#endif
    conditionController.wait(lockController);
  }

};


class GPUWorkerCommunicator
{
public:
  GPUWorkerSync& workerSync;
  Reconstruction& reconstruction;
  volatile int step;
  int GPU;

  //generatePSFVolume // step = 1
  float* _CPUPSF; uint3 _PSFsize_; float3 _sliceVoxelDim; float3 _PSFdim; Matrix4 _PSFI2W; Matrix4 _PSFW2I; float _quality_factor; bool _use_SINC;
  //setSliceDims // step = 2
  std::vector<float3>* _slice_dims; std::vector<int>* _sliceDim; bool _allocate;
  //setSliceMatrices // step = 3
  std::vector<Matrix4>* _matSliceTransforms, *_matInvSliceTransforms, *_matsI2Winit, *_matsW2Iinit, *_matsI2W, *_matsW2I;
  Matrix4 _reconI2W, _reconW2I;
  bool _allocSetSliceMatrices;
  // setMask // step = 4
  uint3 _s; float3 _dim; float* _data; float _sigma_bias;
  // InitReconstructionVolume // step = 5

  // UpdateSliceWeights // step = 6
  std::vector<float> *_slices_weights;
  // UpdateScaleVector // step = 7
  std::vector<float> *_scales;
  // InitStorageVolumes // step = 8
  uint3 _size; int _end, _start;
  // FillSlices // step = 9
  float* _sdata, *_d_ldata, *_ldata; std::vector<int>* _sizesX, *_sizesY;
  // UpdateReconstructed // step = 10
  uint3 _vsize;
  // CorrectBias // step 11
  float _global_bias_correction; 
  float * _d_lbiasdata;
  // SuperresolutionOn // step 12
  int _N;
  Volume<float> *_dev_addon_accbuf_, *_dev_cmap_accbuf_, *_original;

  // GaussianReconstructionOnX1 // step 14
  Volume<float>* _dev_reconstructed_accbuf_;
  // GaussianReconstructionOnX2 // step 15
  int *_voxel_num;
  // NormaliseBiasOnX // step 16
  int _iter;
  Volume<float>* _dev_bias_accbuf_;
  Volume<float>* dev_volume_weights_accbuf_;
  // SimulateSlicesOnX // step 17
  std::vector<bool>::iterator _slice_inside;
  // EStep // step 18
  float _m, _sigma, _mix;
  std::vector<float>* _slice_potential;
  float * _d_lweightsdata;
  // MStepOnX // step 19
  thrust::tuple<float, float, float, float, float> *_results;
  // CalculateScaleVector // step 20
  std::vector<float>* _scale_vec;
  // InitializeEMValues // step 21
  // SliceToVolumeReg // step 22
  // updateResampledSlicesI2W // step 23
  std::vector<Matrix4>* _ofsSlice;
  // registerSlicesToVolume1 // step 24
  std::vector<Matrix4> *_h_trans;
  int _global_slice;
  // registerSlicesToVolume2 // step 25
  // initRegStorageVolumes // step 26
  bool _init;
  // FillRegSlicesOnX // step 27
  std::vector<Matrix4>* _slices_resampledI2W;

  bool _debug;

public:
  GPUWorkerCommunicator(Reconstruction& reconstruction, GPUWorkerSync& workerSync, int GPU) : reconstruction(reconstruction), workerSync(workerSync), step(-2), GPU(GPU)
  { }
  void prepareGeneratePSFVolume(float* CPUPSF, uint3 PSFsize_, float3 sliceVoxelDim,
    float3 PSFdim, Matrix4 PSFI2W, Matrix4 PSFW2I, float quality_factor, bool use_SINC)
  {
    step = 1;
    _CPUPSF = CPUPSF; _PSFsize_ = PSFsize_; _sliceVoxelDim = sliceVoxelDim; _PSFdim = PSFdim; _PSFI2W = PSFI2W; _quality_factor = quality_factor; _use_SINC = use_SINC;
  }

  void prepareSetSliceDims(std::vector<float3>& slice_dims, std::vector<int>& sliceDim, bool allocate)
  {
    step = 2;
    _slice_dims = &slice_dims; _sliceDim = &sliceDim; _allocate = allocate;
  }

  void prepareSetSliceMatrices(std::vector<Matrix4>& matSliceTransforms, std::vector<Matrix4>& matInvSliceTransforms,
    std::vector<Matrix4>& matsI2Winit, std::vector<Matrix4>& matsW2Iinit, std::vector<Matrix4>& matsI2W, std::vector<Matrix4>& matsW2I, Matrix4 reconI2W, Matrix4 reconW2I, bool alloc)
  {
    step = 3;
    _matSliceTransforms = &matSliceTransforms; _matInvSliceTransforms = &matInvSliceTransforms; _matsI2Winit = &matsI2Winit; _matsW2Iinit = &matsW2Iinit; _matsI2W = &matsI2W; _matsW2I = &matsW2I;
    _reconI2W = reconI2W; _reconW2I = reconW2I;
    _allocSetSliceMatrices = alloc;
  }

  void prepareSetMask(uint3 s, float3 dim, float* data, float sigma_bias)
  {
    step = 4;
    _s = s; _dim = dim; _data = data; _sigma_bias = sigma_bias;
  }

  void prepareInitReconstructionVolume(uint3 s, float3 dim, float* data, float sigma_bias)
  {
    step = 5;
    _s = s; _dim = dim; _data = data; _sigma_bias = sigma_bias;
  }

  void prepareUpdateSliceWeights(std::vector<float>& slices_weights)
  {
    step = 6;
    _slices_weights = &slices_weights;
  }

  void prepareUpdateScaleVector(std::vector<float>& scales, std::vector<float>& slices_weights, bool alloc)
  {
    step = 7;
    _scales = &scales;  _slices_weights = &slices_weights; _allocate = alloc;
  }

  void prepareInitStorageVolumes(uint3 size, float3 dim, int start, int end)
  {
    step = 8;
    _size = size; _dim = dim;
    _start = start; _end = end;
  }
  void prepareFillSlices(float* sdata, std::vector<int>& sizesX, std::vector<int>& sizesY, float* d_ldata, float* ldata, bool alloc)
  {
    step = 9;
    _sdata = sdata; _sizesX = &sizesX; _sizesY = &sizesY; _allocate = alloc; _d_ldata = d_ldata; _ldata = ldata;
  }

  void prepareUpdateReconstructed(const uint3 vsize, float* data)
  {
    step = 10;
    _vsize = vsize; _data = data;
  }
  void prepareCorrectBias(float sigma_bias, bool global_bias_correction, float* d_lbiasdata)
  {
    step = 11;
    _sigma_bias = sigma_bias; 
    _global_bias_correction = global_bias_correction; 
    _d_lbiasdata = d_lbiasdata;
  }
  void prepareSuperresolution(int N, Volume<float> &dev_addon_accbuf_, Volume<float> &dev_cmap_accbuf_, Volume<float> &original)
  {
    step = 12;
    _dev_addon_accbuf_ = &dev_addon_accbuf_;
    _dev_cmap_accbuf_ = &dev_cmap_accbuf_;
    _original = &original;
    _N = N;
  }

  void prepareGaussianReconstruction1(int &voxel_num, Volume<float>& dev_reconstructed_accbuf_)
  {
    step = 14;
    _dev_reconstructed_accbuf_ = &dev_reconstructed_accbuf_;
    _voxel_num = &voxel_num;
  }
  void prepareGaussianReconstruction2(int &voxel_num)
  {
    step = 15;
    _voxel_num = &voxel_num;
  }

  void prepareNormaliseBias(int iter, Volume<float>& dev_bias_accbuf_, float sigma_bias)
  {
    step = 16;
    _iter = iter;
    _dev_bias_accbuf_ = &dev_bias_accbuf_;
    _sigma_bias = sigma_bias;
  }

  void prepareSimulateSlices(std::vector<bool>::iterator slice_inside)
  {
    step = 17;
    _slice_inside = slice_inside;
  }

  void prepareEStep(float m, float sigma, float mix, std::vector<float>& slice_potential, float* d_lweigthsdata)
  {
    step = 18;
    _m = _m;
    _sigma = sigma;
    _mix = mix;
    _slice_potential = &slice_potential;
    _d_lweightsdata = d_lweigthsdata;

  }

  void prepareMStep(thrust::tuple<float, float, float, float, float> &results)
  {
    step = 19;
    _results = &results;
  }
  void prepareCalculateScaleVector(std::vector<float>& scale_vec)
  {
    step = 20;
    _scale_vec = &scale_vec;
  }

  void prepareInitializeEMValues()
  {
    step = 21;
  }

  void preparePrepareSliceToVolumeReg(bool allocate)
  {
    step = 22;
    _allocate = allocate;
  }

  void prepareUpdateResampledSlicesI2W(std::vector<Matrix4>& ofsSlice, bool allocate)
  {
    step = 23;
    _ofsSlice = &ofsSlice;
    _allocate = allocate;
  }

  void prepareRegisterSlicesToVolume1(int global_slice, std::vector<Matrix4>& h_trans)
  {
    step = 24;
    _global_slice = global_slice;
    _h_trans = &h_trans;
  }
  void prepareRegisterSlicesToVolume2()
  {
    step = 25;
  }

  void prepareInitRegStorageVolumes(uint3 size, float3 dim, bool init)
  {
    step = 26;
    _size = size;
    _dim = dim;
    _init = init;
  }
  void prepareFillRegSlices(float* sdata, std::vector<Matrix4>& slices_resampledI2W, bool allocate)
  {
    step = 27;
    _sdata = sdata;
    _slices_resampledI2W = &slices_resampledI2W;
    _allocate = allocate;
  }


#if USE_BOOST 
  void operator() ()
  {
    execute();
  }
#endif
  void execute();
  void end();

  // TODO InitializeRobustStatistics - this function has to return values...
};


#endif //GPUWORKER_CUDA_CUH