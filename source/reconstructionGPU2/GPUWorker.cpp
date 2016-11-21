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
      reconstruction.generatePSFVolumeOnX(_CPUPSF, _PSFsize_, _sliceVoxelDim, _PSFdim, _PSFI2W, _PSFW2I, _quality_factor, GPU);
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

