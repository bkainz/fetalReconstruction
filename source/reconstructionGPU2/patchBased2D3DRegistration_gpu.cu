
#include "patchBased2D3DRegistration_gpu.cuh"
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
#include <GPUGauss/patchBasedGaussFilterConvolution.cuh>

texture<float, 3, cudaReadModeElementType > reconstructedTex_;  // texture<ReturnType, Dimension, ReadMode> Name;

void advance_cursor() {
  static int pos = 0;
  char cursor[4] = { '/', '-', '\\', '|' };
  printf("%c\b", cursor[pos]);
  fflush(stdout);
  pos = (pos + 1) % 4;
}

__global__ void initActivePatches(int* buffer, int num)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < num)
    buffer[i] = i;
}

template <typename T>
__global__ void genenerateRegistrationPatches(PatchBasedVolume<T> inputStack, ReconVolume<T> reconstruction,
  int* activelayers, Matrix4<float>* transfMats, int insliceofs)
{
  const uint3 pos = make_uint3(blockIdx.x * blockDim.x + threadIdx.x,
    blockIdx.y * blockDim.y + threadIdx.y,
    activelayers[blockIdx.z]);

  uint3 vSize = inputStack.getXYZPatchGridSize();
  if (pos.x >= vSize.x || pos.y >= vSize.y || pos.z >= vSize.z)
    return;

  ImagePatch2D<T> patch = inputStack.getImagePatch2D(pos.z);

  Matrix4<float> Tr = transfMats[pos.z];
  float3 patchPos = make_float3((float)pos.x, (float)pos.y, insliceofs * 2);
  float3 wpos = Tr*patch.RI2W*patchPos;//Tr*patch.I2W*patchPos;  
  float3 volumePos = reconstruction.reconstructedW2I * wpos;
  
  //TODO -> this will only work for float texture!!
  float val = tex3D(reconstructedTex_, volumePos.x / reconstruction.m_size.x, volumePos.y / reconstruction.m_size.y, volumePos.z / reconstruction.m_size.z);

  if (val < 0)
    val = -1.0f;

  inputStack.setRegPatchValue(pos, val);
}



template <typename T>
void PatchBased2D3DRegistration_gpu<T>::prepareReconTex(int dev)
{
  checkCudaErrors(cudaSetDevice(dev));

  //works only for float interpolation -- will fail for double!!
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

  cudaExtent asize;
  asize.width = m_reconstruction->m_size.x;
  asize.height = m_reconstruction->m_size.y;
  asize.depth = m_reconstruction->m_size.z;

  checkCudaErrors(cudaMalloc3DArray(&m_d_reconstructed_array, &channelDesc, asize));
  
  cudaMemcpy3DParms copyParams = { 0 };
  copyParams.srcPtr = make_cudaPitchedPtr((void*)m_reconstruction->m_d_data, m_reconstruction->m_size.x*sizeof(float),
    m_reconstruction->m_size.x, m_reconstruction->m_size.y);
  copyParams.dstArray = m_d_reconstructed_array;//a1;// dev_reconstructed_array[dev];
  copyParams.extent = asize;
  copyParams.kind = cudaMemcpyDeviceToDevice;
  checkCudaErrors(cudaMemcpy3D(&copyParams));

  reconstructedTex_.addressMode[0] = cudaAddressModeBorder;
  reconstructedTex_.addressMode[1] = cudaAddressModeBorder;
  reconstructedTex_.addressMode[2] = cudaAddressModeBorder;
  reconstructedTex_.filterMode = cudaFilterModeLinear;
  reconstructedTex_.normalized = true;

  checkCudaErrors(cudaBindTextureToArray(reconstructedTex_, m_d_reconstructed_array, channelDesc));
  CHECK_ERROR(cudaBindTextureToArray);
}


template <typename T>
__global__ void averageIf(PatchBasedVolume<T> layers, int* activelayers, float* sum, int* count, int width, int height)
{
  extern __shared__ int reductionSpace[];
  int localid = threadIdx.x + blockDim.x*threadIdx.y;
  int patch = blockIdx.z;
  if (activelayers != 0)
    patch = activelayers[blockIdx.z];
  int threads = blockDim.x*blockDim.y;

  int myActive = 0;
  float myCount = 0;
  for (int y = blockIdx.y*blockDim.y + threadIdx.y; y < height; y += blockDim.y*gridDim.y)
    for (int x = blockIdx.x*blockDim.x + threadIdx.x; x < width; x += blockDim.x*gridDim.x)
    {
      float val = layers.getPatchValue(make_uint3(x, y, patch));
      if (val > -1.0f)
        ++myActive, myCount += val;
    }

  float* f_reduction = reinterpret_cast<float*>(reductionSpace + threads);
  reductionSpace[localid] = myActive;
  f_reduction[localid] = myCount;
  __syncthreads();

  for (int n = threads / 2; n > 1; n /= 2)
  {
    if (localid < n)
      reductionSpace[localid] += reductionSpace[localid + n];
      f_reduction[localid] += f_reduction[localid + n];
    __syncthreads();
  }

  if (threadIdx.x == 0)
  {
    atomicAdd(sum + blockIdx.z, f_reduction[0] + f_reduction[1]);
    atomicAdd(count + blockIdx.z, reductionSpace[0] + reductionSpace[1]);
  }
}


template <typename T>
__global__ void computeNCCAndReduce(PatchBasedVolume<T> layersA, int* activelayersA, PatchBasedVolume<T> layersB, const float* sums, const int *counts,
  float* results, int width, int height, int patches, int level)
{
  int layerA = activelayersA[blockIdx.z];
  int layerB = blockIdx.z;
  int threads = blockDim.x*blockDim.y;

  float avg_a = sums[layerB];
  float avg_b = sums[patches + layerB];
  if (avg_a != 0)   avg_a /= counts[layerB];
  if (avg_b != 0)   avg_b /= counts[patches + layerB];

  float3 values = make_float3(0, 0, 0);

  for (int y = blockIdx.y*blockDim.y + threadIdx.y; y < height; y += blockDim.y*gridDim.y)
    for (int x = blockIdx.x*blockDim.x + threadIdx.x; x < width; x += blockDim.x*gridDim.x)
    {
      float a = layersA.getBufferValue(make_uint3(x, y, layerA));//surf2DLayeredread<float>(layersA, x * 4, y, layerA, cudaBoundaryModeClamp);
      float b = layersB.getRegPatchValue(make_uint3(x, y, layerB));//surf2DLayeredread<float>(layersB, x * 4, y, layerB, cudaBoundaryModeClamp);
      int lin = y * width + x;
      if (a >= 0.0f && b >= 0.0f && lin % level == 0)
      {
        float s = a - avg_a;
        float stex = b - avg_b;
        values = values + make_float3(s*stex, s*s, stex*stex);
      }
    }

  extern __shared__ float reduction[];
  int localid = threadIdx.x + blockDim.x*threadIdx.y;
  reduction[localid] = values.x;
  reduction[localid + threads] = values.y;
  reduction[localid + 2 * threads] = values.z;

  __syncthreads();

  for (int n = threads / 2; n > 1; n /= 2)
  {
    if (localid < n)
      reduction[localid] = reduction[localid] + reduction[localid + n],
      reduction[localid + threads] = reduction[localid + threads] + reduction[localid + threads + n],
      reduction[localid + 2 * threads] = reduction[localid + 2 * threads] + reduction[localid + 2 * threads + n];
    __syncthreads();
  }

  //write results
  if (threadIdx.x == 0)
  {
    atomicAdd(results + blockIdx.z * 3 + 0, reduction[0] + reduction[1]);
    atomicAdd(results + blockIdx.z * 3 + 1, reduction[threads] + reduction[threads + 1]);
    atomicAdd(results + blockIdx.z * 3 + 2, reduction[2 * threads] + reduction[2 * threads + 1]);
  }
}

__global__ void addNccValues(const float* prevData, float* result, int patches)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < patches)
  {
    float norm = prevData[3 * tid + 1] * prevData[3 * tid + 2];
    float res = 0;
    if (norm > 0)
      res = prevData[3 * tid] / sqrtf(norm);
    result[tid] += res;
  }
}

__global__ void writeSimilarities(const float* nvccResults, int* activelayers, int writestep, int writenum, float* similarities, int active_patches, int patches)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < active_patches)
  {
    float res = nvccResults[tid];
    int patch = activelayers[tid];
    for (int i = 0; i < writenum; ++i)
      similarities[patches*writestep*i + patch] = res;
  }
}

template <typename T>
void PatchBased2D3DRegistration_gpu<T>::evaluateCostsMultiplePatches(int active_patches, int patches, int level, float targetBlurring,
  int writeoffset, int writestep, int writenum)
{
  if (active_patches == 0)
    return;

  dim3 redblock(32, 32);
  dim3 patchessize(m_inputStack->getXYZPatchGridSize().x, m_inputStack->getXYZPatchGridSize().y, m_inputStack->getXYZPatchGridSize().z);
  dim3 redgrid = divup(patchessize, dim3(redblock.x * 2, redblock.y * 2));

  checkCudaErrors(cudaMemset(dev_temp_float, 0, 6 * active_patches*sizeof(float)));
  checkCudaErrors(cudaMemset(dev_temp_int, 0, 2 * active_patches*sizeof(int)));
  
  averageIf << <redgrid, redblock, redblock.x*redblock.y*(sizeof(float) + sizeof(int)) >> >(
    *m_inputStack, dev_active_patches, dev_temp_float, dev_temp_int,
    m_inputStack->getXYZPatchGridSize().x, m_inputStack->getXYZPatchGridSize().y);

  //TODO test if three-fold is necessary
  for (int insofs = -1; insofs <= 1; insofs++)
  {
    dim3 blockSize3 = dim3(16, 16, 1);
    dim3 gridSize3 = divup(dim3(m_inputStack->getXYZPatchGridSize().x, m_inputStack->getXYZPatchGridSize().y, active_patches), blockSize3);
    float3 fsize = make_float3(m_reconstruction->m_size.x, m_reconstruction->m_size.y, m_reconstruction->m_size.z);
   
    genenerateRegistrationPatches<T> << < gridSize3, blockSize3 >> >(*m_inputStack, *m_reconstruction, dev_active_patches, dev_recon_matrices, insofs);
    CHECK_ERROR(genenerateRegistrationPatches);
    //TODO -- don't know if this is necessary
    patchBasedFilterGaussStack(m_cuda_device, m_inputStack->getRegPatchesPtr(), m_inputStack->getRegPatchesPtr(),
      m_inputStack->getXYZPatchGridSize(), active_patches, targetBlurring);
    CHECK_ERROR(patchBasedFilterGaussStack);

    averageIf << <redgrid, redblock, redblock.x*redblock.y*(sizeof(float) + sizeof(int)) >> >(
      *m_inputStack, 0, dev_temp_float + active_patches, dev_temp_int + active_patches,
      m_inputStack->getXYZPatchGridSize().x, m_inputStack->getXYZPatchGridSize().y);

    // set ncc temp values to zero
    checkCudaErrors(cudaMemset(dev_temp_float + 2 * patches, 0, 3 * patches*sizeof(float)));

    // compute ncc 
    computeNCCAndReduce << <redgrid, redblock, redblock.x*redblock.y*sizeof(float) * 3 >> >(*m_inputStack,
      dev_active_patches, *m_inputStack, dev_temp_float, dev_temp_int, dev_temp_float + 3 * active_patches, 
      m_inputStack->getXYZPatchGridSize().x, m_inputStack->getXYZPatchGridSize().y, active_patches, level + 1);

    addNccValues << <divup(active_patches, 512), 512 >> >(dev_temp_float + 3 * active_patches, dev_temp_float + 2 * active_patches, active_patches);
  }

  writeSimilarities << <divup(active_patches, 512), 512 >> >(dev_temp_float + 2 * active_patches, dev_active_patches, writestep, writenum, dev_recon_similarities + writeoffset*patches, active_patches, patches);

  //checkCudaErrors(cudaDeviceSynchronize());
  std::vector<float> sim(3 * patches);
  cudaMemcpy(&sim[0], dev_recon_similarities, sizeof(float) * 3 * patches, cudaMemcpyDeviceToHost);
  for (int i = 0; i < 3 * patches; ++i)
    std::cout << sim[i] << " ";
  
  std::cout << std::endl;
}


__global__ void adjustSamplingMatrixForCentralDifferences(const Matrix4<float>* inmatrices,
  Matrix4<float>* outmatrices, int* activeMask, int activePatches, int patches, int part, float step)
{
  const float pi = 3.14159265358979323846f;
  //  const float One80dvPi =  57.2957795131f;

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= activePatches)
    return;

  int patch = activeMask[i];

  //TODO check
  //
  //outmatrices[patch] = inmatrices[patch];
  Matrix4<float>& out = outmatrices[patch];
  const Matrix4<float>& in = inmatrices[patch];
  out = in;
  

  if (part < 3)
  {
    // translation is easy
    out.data[part].w = in.data[part].w + step;
  }
  else
  {
    // rotation is more complex
    float TOL = 0.000001f;
    float tmp = asinf(-1.0f * in.data[0].z);

    float p_rot[3];

    if (fabsf(cos(tmp)) > TOL) {
      p_rot[0] = atan2f(in.data[1].z, in.data[2].z);
      p_rot[1] = tmp;
      p_rot[2] = atan2f(in.data[0].y, in.data[0].x);
    }
    else {
      //m(0,2) is close to +1 or -1
      p_rot[0] = atan2f(-in.data[0].z*in.data[1].x, -in.data[0].z*in.data[2].x);
      p_rot[1] = tmp;
      p_rot[2] = 0;
    }

    // parameter space is in deg so convert to rad
    p_rot[part - 3] += step*pi / 180.0f;

    float cosrx = cos(p_rot[0]);
    float cosry = cos(p_rot[1]);
    float cosrz = cos(p_rot[2]);
    float sinrx = sin(p_rot[0]);
    float sinry = sin(p_rot[1]);
    float sinrz = sin(p_rot[2]);

    out.data[0].x = cosry*cosrz;
    out.data[0].y = cosry*sinrz;
    out.data[0].z = -sinry;

    out.data[1].x = (sinrx*sinry*cosrz - cosrx*sinrz);
    out.data[1].y = (sinrx*sinry*sinrz + cosrx*cosrz);
    out.data[1].z = sinrx*cosry;

    out.data[2].x = (cosrx*sinry*cosrz + sinrx*sinrz);
    out.data[2].y = (cosrx*sinry*sinrz - sinrx*cosrz);
    out.data[2].z = cosrx*cosry;
  }
}

template <typename T>
__global__ void initDevReconMatrices(PatchBasedVolume<T> inputStack, Matrix4<float>* origTransfMats, Matrix4<float>* TransfMats)
{
  const unsigned int idx = blockIdx.x* blockDim.x + threadIdx.x;

  uint3 vSize = inputStack.getXYZPatchGridSize();
  if (idx >= inputStack.getXYZPatchGridSize().z)
    return;

  ImagePatch2D<T> patch = inputStack.getImagePatch2D(idx);
  Matrix4<float> resetMat = patch.Transformation*patch.Mo;
  origTransfMats[idx] = resetMat;
  TransfMats[idx] = resetMat;
}

template <typename T>
__global__ void resetOriginDevReconMatrices(PatchBasedVolume<T> inputStack, Matrix4<float>* TransfMats)
{
  const unsigned int idx = blockIdx.x* blockDim.x + threadIdx.x;

  uint3 vSize = inputStack.getXYZPatchGridSize();
  if (idx >= inputStack.getXYZPatchGridSize().z)
    return;

  ImagePatch2D<T> patch = inputStack.getImagePatch2D(idx);
  Matrix4<float> resetMat = TransfMats[idx] * patch.InvMo;
  TransfMats[idx] = resetMat;

}

template <typename T>
__global__ void finishDevReconMatrices(PatchBasedVolume<T> inputStack, Matrix4<float>* TransfMats, Matrix4<float>* InvTransfMats)
{
  const unsigned int idx = blockIdx.x* blockDim.x + threadIdx.x;

  uint3 vSize = inputStack.getXYZPatchGridSize();
  if (idx >= inputStack.getXYZPatchGridSize().z)
    return;

  //TODO -- reactivate
  //inputStack.setTransformationMatrix(idx, TransfMats[idx]);
  //inputStack.setInvTransformationMatrix(idx, InvTransfMats[idx]);
}


__global__ void computeGradientCentralDiff(const float* similarities, float* gradient, int* activeMask, 
  int activePatches, int patches, int p)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= activePatches)
    return;
  int patch = activeMask[i];

  float dx = similarities[patch] - similarities[patches + patch];
  gradient[p*patches + patch] = dx;
  if (p == 0)
    gradient[6 * patches + patch] = dx*dx;
  else
    gradient[6 * patches + patch] += dx*dx;
}

__global__ void normalizeGradient(float* gradient, int* activeMask, int activePatches, int patches)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= activePatches)
    return;

  int patch = activeMask[i];

  float norm = gradient[6 * patches + patch];
  if (norm > 0)
    norm = 1.0f / sqrtf(norm);

  for (int j = 0; j < 6; ++j)
    gradient[j*patches + patch] *= norm;
}

__global__ void copySimilarity(float* similarities, int active_patches, int patches, int* activeMask, int target, int source)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= active_patches)
    return;
  int patch = activeMask[i];
  similarities[target*patches + patch] = similarities[source*patches + patch];
}

__global__ void gradientStep(Matrix4<float>* matrices, const float* gradient, int active_patches, int patches, int* activeMask, float step)
{
  const float pi = 3.14159265358979323846f;

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= active_patches)
    return;
  int patch = activeMask[i];

  Matrix4<float>& matrix = matrices[patch];

  // translation is easy
  for (int p = 0; p < 3; ++p)
    matrix.data[p].w = matrix.data[p].w + step*gradient[p*patches + patch];


  // rotation is more complex
  float TOL = 0.000001f;
  float tmp = asinf(-1.0f * matrix.data[0].z);

  float p_rot[3];

  if (fabsf(cos(tmp)) > TOL) {
    p_rot[0] = atan2f(matrix.data[1].z, matrix.data[2].z);
    p_rot[1] = tmp;
    p_rot[2] = atan2f(matrix.data[0].y, matrix.data[0].x);
  }
  else {
    //m(0,2) is close to +1 or -1
    p_rot[0] = atan2f(-matrix.data[0].z*matrix.data[1].x, -matrix.data[0].z*matrix.data[2].x);
    p_rot[1] = tmp;
    p_rot[2] = 0;
  }

  // parameter space is in deg so convert to rad
  for (int p = 0; p < 3; ++p)
    p_rot[p] += gradient[(p + 3)*patches + patch] * step*pi / 180.0f;

  float cosrx = cos(p_rot[0]);
  float cosry = cos(p_rot[1]);
  float cosrz = cos(p_rot[2]);
  float sinrx = sin(p_rot[0]);
  float sinry = sin(p_rot[1]);
  float sinrz = sin(p_rot[2]);

  matrix.data[0].x = cosry*cosrz;
  matrix.data[0].y = cosry*sinrz;
  matrix.data[0].z = -sinry;

  matrix.data[1].x = (sinrx*sinry*cosrz - cosrx*sinrz);
  matrix.data[1].y = (sinrx*sinry*sinrz + cosrx*cosrz);
  matrix.data[1].z = sinrx*cosry;

  matrix.data[2].x = (cosrx*sinry*cosrz + sinrx*sinrz);
  matrix.data[2].y = (cosrx*sinry*sinrz - sinrx*cosrz);
  matrix.data[2].z = cosrx*cosry;
}

__device__ int dev_active_patch_count = 0;

__global__ void checkImprovement(int* newActiveMask, int activePatches, int patches, 
    const int* activeMask, float* similarities, int cursim, int prev, float eps)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= patches)
    return;

  int active = 0;
  int patch = -1;
  
  if (tid < activePatches /*&& idx1 < patches * 5 && idx2 < patches * 5*/)
  {
    patch = activeMask[tid];
    active = similarities[cursim*patches + patch] > similarities[prev*patches + patch] + eps ? 1 : 0;
   //active = similarities[idx1] > similarities[idx2] + eps ? 1 : 0;
  }

  // run prefix sum
  extern __shared__ int prefixSumSpace[];
  prefixSumSpace[tid] = active;
  int offset = 1;
  int n = blockDim.x;
  int overallcount = 0;

  for (int d = n >> 1; d > 0; d /= 2)
  {
    __syncthreads();
    if (tid < d)
    {
      int ai = offset*(2 * tid + 1) - 1;
      int bi = offset*(2 * tid + 2) - 1;
      prefixSumSpace[bi] += prefixSumSpace[ai];
    }
    offset *= 2;
  }

  if (tid == 0)
  {
    overallcount = prefixSumSpace[n - 1];
    prefixSumSpace[n - 1] = 0;
  }

  for (int d = 1; d < n; d *= 2)
  {
    offset /= 2;
    __syncthreads();
    if (tid < d)
    {
      int ai = offset*(2 * tid + 1) - 1;
      int bi = offset*(2 * tid + 2) - 1;
      float t = prefixSumSpace[ai];
      prefixSumSpace[ai] = prefixSumSpace[bi];
      prefixSumSpace[bi] += t;
    }
  }

  // get global array offset
  __shared__ int globalOffset;
  if (tid == 0)
  {
    globalOffset = atomicAdd(&dev_active_patch_count, overallcount);
  }
  __syncthreads();

  
  if (active == 1)
  {
    unsigned int idx = globalOffset + prefixSumSpace[tid];
    if (idx < patches)
      newActiveMask[idx] = patch;
  }

}



template <typename T>
void PatchBased2D3DRegistration_gpu<T>::run()
{

  printf("patchBased2D3DRegistration_gpu ");
  cerr.rdbuf(file_e.rdbuf());
  cout.rdbuf(file.rdbuf());
  //TODO patch batch wise for kernel 2s watchdogs necesary?
  checkCudaErrors(cudaSetDevice(m_cuda_device));

  //TODO: test double reduction accuracy

  //first test: we copy first the patch data into a non surfaced (size restrictions) memory area
  //then we do the same as in the previous version for now. TODO improve to single layer analytical gradient optimization

  //TESTS:
  //- perhaps patches that are not resampled to recon space are also ok -- if not: additional stack of resampled patches

  ////////////////////////////////////////////7
  //Reset origin to avoid drift!
  ////////////////////////////////////////////7
 
  int dev_patches = m_inputStack->getXYZPatchGridSize().z;

  initDevReconMatrices<T> << <divup(dev_patches, 512), 512 >> >(*m_inputStack, dev_recon_matrices_orig, dev_recon_matrices);

  //TODO test float and double

 /* Matrix4<float>* h_InvRecon_matrices_ = new Matrix4<float>[dev_patches];
  checkCudaErrors(cudaMemcpy(h_InvRecon_matrices_, dev_recon_matrices, sizeof(Matrix4<float>)*dev_patches, cudaMemcpyDeviceToHost));
  for (int i = 0; i < dev_patches; i++)
  {
    if (i > dev_patches / 2 && i < dev_patches / 2 + 10)
    {
      irtkMatrix Tmat = fromMatrix4<float>(h_InvRecon_matrices_[i]);
      Tmat.Print();
    }
  }*/

  //for levels loop
  for (int level = m_NumberOfLevels - 1; level >= 0; --level)
  {
    // settings for level
    T _TargetBlurring = m_Blurring[level];
    T _StepSize = m_LengthOfSteps[level];

    printf("_StepSize: %f\n", _StepSize);

    //printf("targetBlurring %f %f %f \n", _Blurring[0], _Blurring[1], _Blurring[2]);
    int activePatches = m_inputStack->getXYZPatchGridSize().z;//all for now
    // using buffer for different level of blurring
    patchBasedFilterGaussStack(m_cuda_device, m_inputStack->getPatchesPtr(), m_inputStack->getBufferPtr(),
      m_inputStack->getXYZPatchGridSize(), activePatches, _TargetBlurring);
    CHECK_ERROR(patchBasedFilterGaussStack);

    for (int st = 0; st < m_NumberOfSteps; st++)
    {
      //active all patches
      activePatches = dev_patches;
      initActivePatches << <divup(activePatches, 512), 512 >> >(dev_active_patches, dev_patches);
      CHECK_ERROR(initActivePatches);

      for (int iter = 0; iter < m_NumberOfIterations; iter++)
      {
        evaluateCostsMultiplePatches(activePatches, dev_patches, level, _TargetBlurring, 0, 1, 3);

        for (int p = 0; p < 6; ++p)
        {
          adjustSamplingMatrixForCentralDifferences << <divup(activePatches, 512), 512 >> >(
            dev_recon_matrices_orig, dev_recon_matrices, 
            dev_active_patches, activePatches, dev_patches, p, _StepSize);
          evaluateCostsMultiplePatches(activePatches, dev_patches, level, _TargetBlurring, 3, 0, 1);
          adjustSamplingMatrixForCentralDifferences << <divup(activePatches, 512), 512 >> >(
            dev_recon_matrices_orig, dev_recon_matrices,
            dev_active_patches, activePatches, dev_patches, p, -_StepSize);
          evaluateCostsMultiplePatches(activePatches, dev_patches, level, _TargetBlurring, 4, 0, 1);
          
          computeGradientCentralDiff << <divup(activePatches, 512), 512 >> >(
            dev_recon_similarities + 3 * m_inputStack->getXYZPatchGridSize().z, dev_recon_gradient, 
            dev_active_patches, activePatches, m_inputStack->getXYZPatchGridSize().z, p);
        }

        normalizeGradient << <divup(activePatches, 512), 512 >> >(dev_recon_gradient, dev_active_patches, activePatches, dev_patches);

        int prevActivePatches = activePatches;
        checkCudaErrors(cudaMemcpy(dev_active_patches_prev, dev_active_patches, activePatches*sizeof(int), cudaMemcpyDeviceToDevice));

        do {
          copySimilarity << <divup(activePatches, 512), 512 >> >(dev_recon_similarities, 
            activePatches, dev_patches, dev_active_patches, 2, 0);
          // step along gradient and generate matrix
          gradientStep << <divup(activePatches, 512), 512 >> >(dev_recon_matrices, dev_recon_gradient, activePatches, dev_patches, dev_active_patches, _StepSize);
          // run evaluate cost for similarity
          evaluateCostsMultiplePatches(activePatches, dev_patches, level, _TargetBlurring, 0, 1, 1);
  
          // check how many are still improving and compact (similarity > new_similarity + _Epsilon)
          int h_active_patch_count = 0;
          checkCudaErrors(cudaMemcpyToSymbol(dev_active_patch_count, &h_active_patch_count, sizeof(int)));

          //printf("%d %d %fKB\n", activePatches, dev_patches, (dev_patches * sizeof(int))/1024.0f);
          checkImprovement << <divup(activePatches, 512), 512, dev_patches * sizeof(int) >> >(
            dev_active_patches2, activePatches, dev_patches, dev_active_patches, dev_recon_similarities, 0, 2, m_Epsilon);
          checkCudaErrors(cudaDeviceSynchronize());
          std::swap(dev_active_patches, dev_active_patches2);

          // copy number of improving
          checkCudaErrors(cudaMemcpyFromSymbol(&activePatches, dev_active_patch_count, sizeof(int)));
        } while (activePatches > 0);

        gradientStep << <divup(prevActivePatches, 512), 512 >> >(
          dev_recon_matrices, dev_recon_gradient, prevActivePatches, dev_patches, dev_active_patches_prev, -_StepSize);

        // copy matrices
        checkCudaErrors(cudaMemcpy(dev_recon_matrices_orig, dev_recon_matrices, sizeof(Matrix4<float>)*dev_patches, cudaMemcpyDeviceToDevice));

        // check for overall improvement and compact
        int h_active_patch_count = 0;
        checkCudaErrors(cudaMemcpyToSymbol(dev_active_patch_count, &h_active_patch_count, sizeof(int)));
        checkImprovement << <divup(prevActivePatches, 512), 512, dev_patches * sizeof(int) >> >(
          dev_active_patches, prevActivePatches, dev_patches, dev_active_patches_prev, dev_recon_similarities, 2, 1, m_Epsilon);
        checkCudaErrors(cudaMemcpyFromSymbol(&activePatches, dev_active_patch_count, sizeof(int)));

        //printf("still active: %d/%d\n", activePatches, dev_patches);
        //printSim(dev_recon_similarities[dev], dev_active_patches[dev], dev_patches, activePatches);
        cout.rdbuf(strm_buffer);
        cerr.rdbuf(strm_buffer_e);
        advance_cursor();
        cerr.rdbuf(file_e.rdbuf());
        cout.rdbuf(file.rdbuf());

        if (activePatches == 0)
          break;
      }
      _StepSize /= 2.0f;
    }
  }

  resetOriginDevReconMatrices<T> << <divup(dev_patches, 512), 512 >> >(*m_inputStack, dev_recon_matrices);

  cout.rdbuf(strm_buffer);
  cerr.rdbuf(strm_buffer_e);

  //copy matrices back to patches.Transform
  //TODO inverse on device?
  Matrix4<float>* h_InvRecon_matrices = new Matrix4<float>[dev_patches];
  checkCudaErrors(cudaMemcpy(h_InvRecon_matrices, dev_recon_matrices, sizeof(Matrix4<float>)*dev_patches, cudaMemcpyDeviceToHost));
  for (int i = 0; i < dev_patches; i++)
  {
    irtkMatrix Tmat = fromMatrix4<float>(h_InvRecon_matrices[i]);
    Tmat.Invert();
    h_InvRecon_matrices[i] = toMatrix4<float>(Tmat);
    if (i > 100 && i < 110)
    {
      Tmat.Print();
    }
  }
 // std::cin.get();
  Matrix4<float>* d_InvRecon_matrices;
  checkCudaErrors(cudaMalloc(&d_InvRecon_matrices, sizeof(Matrix4<float>)*dev_patches));
  checkCudaErrors(cudaMemcpy(d_InvRecon_matrices, h_InvRecon_matrices, sizeof(Matrix4<float>)*dev_patches, cudaMemcpyHostToDevice));

  finishDevReconMatrices<T> << <divup(dev_patches, 512), 512 >> >(*m_inputStack, dev_recon_matrices, d_InvRecon_matrices);
  CHECK_ERROR(finishDevReconMatrices);

  checkCudaErrors(cudaFree(d_InvRecon_matrices));
  delete[] h_InvRecon_matrices;

  printf("\n");

}


template <typename T>
PatchBased2D3DRegistration_gpu<T>::PatchBased2D3DRegistration_gpu(int cuda_device, 
  PatchBasedVolume<T>* inputStack, ReconVolume<T>* reconstruction) :
  m_cuda_device(cuda_device), m_inputStack(inputStack), m_reconstruction(reconstruction)
{

  strm_buffer = cout.rdbuf();
  strm_buffer_e = cerr.rdbuf();
  file.open("log-2D3DregGPU.txt");
  file_e.open("log-2D3Dreg-errorGPU.txt");

  m_NumberOfLevels = 3;
  m_NumberOfSteps = 4;
  m_NumberOfIterations = 20;
  m_Epsilon = 0.0001f;

  m_N = m_inputStack->getXYZPatchGridSize().x * m_inputStack->getXYZPatchGridSize().y * m_inputStack->getXYZPatchGridSize().z;
  checkCudaErrors(cudaMalloc(&dev_active_patches, sizeof(int)*m_inputStack->getXYZPatchGridSize().z));
  checkCudaErrors(cudaMalloc(&dev_active_patches2, sizeof(int)*m_inputStack->getXYZPatchGridSize().z));
  checkCudaErrors(cudaMalloc(&dev_active_patches_prev, sizeof(int)*m_inputStack->getXYZPatchGridSize().z));

  checkCudaErrors(cudaMalloc(&dev_temp_float, sizeof(T) * 6 * m_inputStack->getXYZPatchGridSize().z));
  checkCudaErrors(cudaMalloc(&dev_temp_int, sizeof(int) * 2 * m_inputStack->getXYZPatchGridSize().z));

  checkCudaErrors(cudaMalloc(&dev_recon_similarities, sizeof(T) * 5 * m_inputStack->getXYZPatchGridSize().z));
  checkCudaErrors(cudaMalloc(&dev_recon_gradient, sizeof(T) * 7 * m_inputStack->getXYZPatchGridSize().z));
  checkCudaErrors(cudaMalloc(&dev_recon_matrices_orig, sizeof(Matrix4<float>)* m_inputStack->getXYZPatchGridSize().z));
  checkCudaErrors(cudaMalloc(&dev_recon_matrices, sizeof(Matrix4<float>)* m_inputStack->getXYZPatchGridSize().z));
  
  m_Blurring = new float[m_NumberOfLevels];
  m_LengthOfSteps = new float[m_NumberOfLevels];
  m_Blurring[0] = (m_reconstruction->m_dim.x) / 2.0f;
  for (int i = 0; i < m_NumberOfLevels; i++) {
    m_LengthOfSteps[i] = 2.0f * pow(2.0f, i);
  }
  for (int i = 1; i < m_NumberOfLevels; i++) {
    m_Blurring[i] = m_Blurring[i - 1] * 2;
  }

  prepareReconTex(m_cuda_device);
}

template <typename T>
PatchBased2D3DRegistration_gpu<T>::~PatchBased2D3DRegistration_gpu()
{
  checkCudaErrors(cudaFreeArray(m_d_reconstructed_array));

  checkCudaErrors(cudaFree(dev_active_patches));
  checkCudaErrors(cudaFree(dev_active_patches2));
  checkCudaErrors(cudaFree(dev_active_patches_prev));

  checkCudaErrors(cudaFree(dev_temp_float));
  checkCudaErrors(cudaFree(dev_temp_int));
  checkCudaErrors(cudaFree(dev_recon_similarities));
  checkCudaErrors(cudaFree(dev_recon_gradient));

  checkCudaErrors(cudaFree(dev_recon_matrices_orig));
  checkCudaErrors(cudaFree(dev_recon_matrices));
}

template class PatchBased2D3DRegistration_gpu < float >;
//template patchBased2D3DRegistration_gpu < double >;
