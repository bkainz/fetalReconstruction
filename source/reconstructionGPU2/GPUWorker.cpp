/*=========================================================================
Library   : Image Registration Toolkit (IRTK)
Copyright : Imperial College, Department of Computing
Visual Information Processing (VIP), 2011 onwards
Date      : $Date: 2013-11-15 14:36:30 +0100 (Fri, 15 Nov 2013) $
Version   : $Revision: 1 $
Changes   : $Author: msteinberger, bkainz $

Copyright (c) 2014, Bernhard Kainz, Markus Steinberger,
Maria Murgasova, Kevin Keraudren
All rights reserved.

If you use this work for research we would very much appreciate if you cite
Bernhard Kainz, Markus Steinberger, Maria Kuklisova-Murgasova, Christina Malamateniou,
Wolfgang Wein, Thomas Torsney-Weir, Torsten Moeller, Mary Rutherford,
Joseph V. Hajnal and Daniel Rueckert:
Fast Volume Reconstruction from Motion Corrupted 2D Slices.
IEEE Transactions on Medical Imaging, in press, 2015

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

#include "GPUWorker.h"
#include "reconstruction_cuda2.cuh"

#if !USE_BOOST

void GPUWorkerSync::completed()
{
  std::unique_lock<std::mutex> lockworker(mutexWorker);
  int ccount;
  ccount = ++count;

  if (ccount >= GPU_count)
  {
    std::unique_lock<std::mutex> lockcontroller(mutexController);
    conditionController.notify_all();
  }
  conditionWorker.wait(lockworker);
}
void GPUWorkerSync::runNextRound()
{
  std::unique_lock<std::mutex> lockController(mutexController);
  {
    std::unique_lock<std::mutex> lockworker(mutexWorker);
    count = 0;
    conditionWorker.notify_all();
  }
  conditionController.wait(lockController);
}

#else

void GPUWorkerSync::completed()
{
  boost::unique_lock<boost::mutex> lockworker(mutexWorker);
  int ccount;
  ccount = ++count;

  if (ccount >= GPU_count)
  {
    boost::unique_lock<boost::mutex> lockcontroller(mutexController);
    conditionController.notify_all();
  }
  conditionWorker.wait(lockworker);
}
void GPUWorkerSync::runNextRound()
{
  boost::unique_lock<boost::mutex> lockController(mutexController);
  {
    boost::unique_lock<boost::mutex> lockworker(mutexWorker);
    count = 0;
    conditionWorker.notify_all();
  }
  conditionController.wait(lockController);
}

#endif

void GPUWorkerSync::runNoSync()
{
#if !USE_BOOST
  std::unique_lock<std::mutex> lockController(mutexController);
  {
    std::unique_lock<std::mutex> lockworker(mutexWorker);
    count = 0;
    conditionWorker.notify_all();
  }
#else
  boost::unique_lock<boost::mutex> lockController(mutexController);
  {
    boost::unique_lock<boost::mutex> lockworker(mutexWorker);
    count = 0;
    conditionWorker.notify_all();
  }
#endif
}


void GPUWorkerCommunicator::execute()
{
  while (true)
  {
    if (reconstruction._debugGPU)
      printf("going to exec %d on %d\n", step, GPU);

    switch (step)
    {

    case -1:
    case -2:
      break;
    default:
      std::cout << "Warning unknown step " << step << " expected to launch by GPU " << GPU << std::endl;
      break;

      //generatePSFVolume // step = 1
    case 1:
      reconstruction.generatePSFVolumeOnX(_CPUPSF, _PSFsize_, _sliceVoxelDim, _PSFdim, _PSFI2W, _PSFW2I, _quality_factor, _use_SINC, GPU);
      break;

      //setSliceDims // step = 2
    case 2:
      reconstruction.setSliceDimsOnX(*_slice_dims, *_sliceDim, _allocate, GPU);
      break;

      //setSliceMatrices // step = 3
    case 3:
      reconstruction.SetSliceMatricesOnX(*_matSliceTransforms, *_matInvSliceTransforms, *_matsI2Winit, *_matsW2Iinit, *_matsI2W, *_matsW2I, _reconI2W, _reconW2I, GPU, _allocSetSliceMatrices);
      break;

      // setMask // step = 4
    case 4:
      reconstruction.setMaskOnX(_s, _dim, _data, _sigma_bias, GPU);
      break;

      // InitReconstructionVolume // step = 5
    case 5:
      reconstruction.InitReconstructionVolumeOnX(_s, _dim, _data, _sigma_bias, GPU);
      break;

      // UpdateSliceWeights // step = 6
    case 6:
      reconstruction.UpdateSliceWeightsOnX(*_slices_weights, GPU);
      break;

      // prepareUpdateScaleVector // step = 7
    case 7:
      reconstruction.UpdateScaleVectorOnX(*_scales, *_slices_weights, GPU, _allocate);
      break;

      // InitStorageVolumes // step = 8
    case 8:
      reconstruction.initStorageVolumesOnX(_size, _dim, _start, _end, GPU);
      break;

      // FillSlices // step = 9
    case 9:
      reconstruction.FillSlicesOnX(_sdata, *_sizesX, *_sizesY, _d_ldata, _ldata, GPU, _allocate);
      break;

      // UpdateReconstructed // step = 10
    case 10:
      reconstruction.UpdateReconstructedOnX(_vsize, _data, GPU);
      break;

      // CorrectBias // step 11
    case 11:
      reconstruction.CorrectBiasOnX(_global_bias_correction, _global_bias_correction, _d_lbiasdata, GPU);
      break;

      // SuperresolutionOn // step 12
    case 12:
      reconstruction.SuperresolutionOnX1(_N, *_dev_addon_accbuf_, *_dev_cmap_accbuf_, *_original, GPU);
      break;

      // GaussianReconstructionOnX1 // step 14
    case 14:
      reconstruction.GaussianReconstructionOnX1(*_voxel_num, *_dev_reconstructed_accbuf_, GPU);
      break;

      // GaussianReconstructionOnX2 // step 15
    case 15:
      reconstruction.GaussianReconstructionOnX2(*_voxel_num, GPU);
      break;

      // NormaliseBiasOnX // step 16
    case 16:
      reconstruction.NormaliseBiasOnX(_iter, *_dev_bias_accbuf_, *dev_volume_weights_accbuf_, _sigma, GPU);
      break;

      // SimulateSlicesOnX // step 17
    case 17:
      reconstruction.SimulateSlicesOnX(_slice_inside, GPU);
      break;

      // EStep // step 18
    case 18:
      reconstruction.EStepOnX(_m, _sigma, _mix, *_slice_potential, _d_lweightsdata, GPU);
      break;

      // MStepOnX // step 19
    case 19:
      reconstruction.MStepOnX(*_results, GPU);
      break;

      // CalculateScaleVector // step 20
    case 20:
      reconstruction.CalculateScaleVectorOnX(*_scale_vec, GPU);
      break;

      // InitializeEMValues // step 21
    case 21:
      reconstruction.InitializeEMValuesOnX(GPU);
      break;

      // prepareSliceToVolumeReg // step 22
    case 22:
      reconstruction.prepareSliceToVolumeRegOnX(_allocate, GPU);
      break;

      // updateResampledSlicesI2W // step 23
    case 23:
      reconstruction.updateResampledSlicesI2WOnX(*_ofsSlice, _allocate, GPU);
      break;

      // registerSlicesToVolume1 // step 24
    case 24:
      reconstruction.registerSlicesToVolumeOnX1(_global_slice, *_h_trans, GPU);
      break;

      // registerSlicesToVolume2 // step 25
    case 25:
      reconstruction.registerSlicesToVolumeOnX2(GPU);
      break;

      // initRegStorageVolumes // step 26
    case 26:
      reconstruction.initRegStorageVolumesOnX(_size, _dim, _init, GPU);
      break;

      // FillRegSlicesOnX // step 27
    case 27:
      reconstruction.FillRegSlicesOnX(_sdata, *_slices_resampledI2W, _allocate, GPU);
      break;
    }

    if (step != -1)
      workerSync.completed();
    else
      break;
  }
}
void GPUWorkerCommunicator::end()
{
  step = -1;
}

