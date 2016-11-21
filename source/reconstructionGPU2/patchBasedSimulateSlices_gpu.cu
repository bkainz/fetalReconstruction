#include "volume.cuh"
#include "patchBasedVolume.cuh"
#include "patchBasedLayeredSurface3D.cuh"
#include "reconVolume.cuh"
#include "reconConfig.cuh"
#include "pointSpreadFunction.cuh"
#include <irtkImage.h>
#include <thrust/inner_product.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/count.h>

//The globally constant point spread function
extern __constant__ PointSpreadFunction _PSF;

template <typename T>
__global__ void patchBasedSimulatePatchesKernel(PatchBasedVolume<T> inputStack, ReconVolume<T> reconstruction)
{
  //patch based coordinates
  const uint3 pos = make_uint3(blockIdx.x* blockDim.x + threadIdx.x,
    blockIdx.y* blockDim.y + threadIdx.y,
    blockIdx.z* blockDim.z + threadIdx.z);

  uint3 vSize = inputStack.getXYZPatchGridSize();

  if (/*pos.x >= vSize.x || pos.y >= vSize.y ||*/  pos.z >= vSize.z)
    return;

  //from input data with patch calculation
  //float s = inputStack.getValueFromPatchCoords(pos);

  //from patch buffer
  float s = inputStack.getPatchValue(pos);

  if ((s == -1.0f))
    return;

  float sume = inputStack.getPSFsumsValue(pos); //v_PSF_sums[idx];
  if (sume == 0.0f)
    return;

  ImagePatch2D patch = inputStack.getImagePatch2D(pos.z);
  s = s *patch.scale;

  float simulated_sliceV = 0;
  bool patch_inside = 0;
  float weight = 0;
  float3 patchPos = make_float3(pos.x, pos.y, 0);
  float3 patchDim = inputStack.getDim();

  float size_inv = 2.0f * _PSF.m_quality_factor / reconstruction.m_dim.x;
  int xDim = round_((patchDim.x * size_inv));
  int yDim = round_((patchDim.y * size_inv));
  int zDim = round_((patchDim.z * size_inv));

  //truncate if value gets close to epsilon
  int dim = MAX_PSF_SUPPORT;
  int centre = (MAX_PSF_SUPPORT - 1) / 2;

  Matrix4 combInvTrans = patch.W2I * (patch.InvTransformation * reconstruction.reconstructedI2W);
  float3 psfxyz;
  float3 _psfxyz = reconstruction.reconstructedW2I*(patch.Transformation*  (patch.I2W * patchPos));
  psfxyz = make_float3(round_(_psfxyz.x), round_(_psfxyz.y), round_(_psfxyz.z));

  for (int z = 0; z < dim; z++) {
    for (int y = 0; y < dim; y++) {
      float oldPSF = FLT_MAX;
      for (int x = 0; x < dim; x++)
      {
        float3 ofsPos;
        float psfval = _PSF.getPSFParamsPrecomp(ofsPos, psfxyz, make_int3(x - centre, y - centre, z - centre), combInvTrans, patchPos, patchDim);
        if (abs(oldPSF - psfval) < PSF_EPSILON) continue;
        oldPSF = psfval;

        uint3 apos = make_uint3(round_(ofsPos.x), round_(ofsPos.y), round_(ofsPos.z)); //NN
        if (apos.x < reconstruction.m_size.x && apos.y < reconstruction.m_size.y && apos.z < reconstruction.m_size.z
          && reconstruction.m_d_mask[apos.x + apos.y*reconstruction.m_size.x + apos.z*reconstruction.m_size.x*reconstruction.m_size.y] != 0)
        {
          psfval /= sume;
          simulated_sliceV += psfval * reconstruction.getReconValueFromTexture(apos);
         // simulated_sliceV += psfval * reconstructed[apos];
          weight += psfval;

          patch_inside = 1;
        }
      }
    }
  }

  if (weight > 0)
  {
    inputStack.setSimulatedPatchValue(pos, simulated_sliceV / weight);
    inputStack.setSimulatedWeight(pos, weight);
    inputStack.setSimulatedInside(pos, patch_inside);
  }


}

template <typename T>
void patchBasedSimulatePatches_gpu(int cuda_device, PatchBasedVolume<T> & inputStack, ReconVolume<T> & reconstruction)
{
  printf("patchBasedSimulateSlices_gpu\n");

  //TODO patch batch wise for kernel 2s watchdogs necesary?
  checkCudaErrors(cudaSetDevice(cuda_device));

  reconstruction.updateReconTex(cuda_device);

  dim3 blockSize3 = dim3(8, 8, 8); //max 1024 threads
  dim3 gridSize3 = divup(dim3(inputStack.getXYZPatchGridSize().x, inputStack.getXYZPatchGridSize().y,
    inputStack.getXYZPatchGridSize().z), blockSize3);
  patchBasedSimulatePatchesKernel<T> << <gridSize3, blockSize3 >> >(inputStack, reconstruction);
  CHECK_ERROR(patchBasedPSFReconstructionKernel);
  checkCudaErrors(cudaDeviceSynchronize());

}

template void patchBasedSimulatePatches_gpu<float>(int cuda_device, PatchBasedVolume<float> & inputStack, ReconVolume<float> & reconstruction);
template void patchBasedSimulatePatches_gpu<double>(int cuda_device, PatchBasedVolume<double> & inputStack, ReconVolume<double> & reconstruction);