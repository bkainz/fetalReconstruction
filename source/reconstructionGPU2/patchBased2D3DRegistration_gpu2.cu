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

#include "patchBased2D3DRegistration_gpu2.cuh"
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

//this is only possible in float and not thread safe!
//texture<float, 3, cudaReadModeElementType > reconstructedTex2_;

void advance_cursor2() {
  static int pos = 0;
  char cursor[4] = { '/', '-', '\\', '|' };
  printf("%c\b", cursor[pos]);
  fflush(stdout);
  pos = (pos + 1) % 4;
}

/*template <typename T>
void PatchBased2D3DRegistration_gpu2<T>::prepareReconTex(int dev)
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

  reconstructedTex2_.addressMode[0] = cudaAddressModeBorder;
  reconstructedTex2_.addressMode[1] = cudaAddressModeBorder;
  reconstructedTex2_.addressMode[2] = cudaAddressModeBorder;
  reconstructedTex2_.filterMode = cudaFilterModeLinear;
  reconstructedTex2_.normalized = true;

  checkCudaErrors(cudaBindTextureToArray(reconstructedTex2_, m_d_reconstructed_array, channelDesc));
  CHECK_ERROR(cudaBindTextureToArray);
}*/

template <typename T>
__global__ void initDevReconMatrices2(PatchBasedVolume<T> inputStack, Matrix4<T>* origTransfMats, Matrix4<T>* TransfMats)
{
  const unsigned int idx = blockIdx.x* blockDim.x + threadIdx.x;

  uint3 vSize = inputStack.getXYZPatchGridSize();
  if (idx >= inputStack.getXYZPatchGridSize().z)
    return;

  ImagePatch2D<T> patch = inputStack.getImagePatch2D(idx);
  Matrix4<T> resetMat = patch.Transformation*patch.Mo;
  origTransfMats[idx] = resetMat;
  TransfMats[idx] = resetMat;
}

template <typename T>
__global__ void resetOriginDevReconMatrices2(PatchBasedVolume<T> inputStack, Matrix4<T>* TransfMats)
{
  const unsigned int idx = blockIdx.x* blockDim.x + threadIdx.x;

  uint3 vSize = inputStack.getXYZPatchGridSize();
  if (idx >= inputStack.getXYZPatchGridSize().z)
    return;

  ImagePatch2D<T> patch = inputStack.getImagePatch2D(idx);
  Matrix4<T> resetMat = TransfMats[idx] * patch.InvMo;
  TransfMats[idx] = resetMat;

}

template <typename T>
__global__ void finishDevReconMatrices2(PatchBasedVolume<T> inputStack, Matrix4<T>* TransfMats, Matrix4<T>* InvTransfMats)
{
  const unsigned int idx = blockIdx.x* blockDim.x + threadIdx.x;

  uint3 vSize = inputStack.getXYZPatchGridSize();
  if (idx >= inputStack.getXYZPatchGridSize().z)
    return;

  inputStack.setTransformationMatrix(idx, TransfMats[idx]);
  inputStack.setInvTransformationMatrix(idx, InvTransfMats[idx]);
}

//////////////////////////////////////////////////////////////
//TODO implement this as per block reduction in shared memory
template <typename T>
__device__ T computeCCpatch(PatchBasedVolume<T>& inputStack, ReconVolume<T>& reconstruction, const Matrix4<T>& Tmat, unsigned int patchIdx, uint3& psize, int level)
{

  ImagePatch2D<T>& patch = inputStack.getImagePatch2D(patchIdx);

  T _xy = 0;
  T _y = 0;
  T _x = 0;
  T _y2 = 0;
  T _x2 = 0;
  unsigned int _n = 0;

  for (int y = 0; y < psize.y; y += (level + 1))
  {
    for (int x = 0; x < psize.x; x += (level + 1))
    {
#pragma unroll
      for (int z = (int)(-(psize.z + 0.5f)) / 2; z < (int)((psize.z + 0.5f) / 2 + 1); z++ /*+= (level + 1)*/)
      {
        T a = inputStack.getBufferValue(make_uint3(x, y, patchIdx)); //TODO three gaussian levels!

        //TODO make double3 with T
        float3 patchPos = make_float3(x, y, z);
        float3 wpos = Tmat*patch.RI2W*patchPos;//Tr*patch.I2W*patchPos;  
        float3 volumePos = reconstruction.reconstructedW2I * wpos;

        T b = reconstruction.getReconValueInterp(volumePos);

        if (a >= 0.0f && b >= 0.0f && a == a && b == b)
        {
          _xy += a*b;
          _x += a;
          _y += b;
          _x2 += a*a;
          _y2 += b*b;
          _n++;
        }
        //TODO why are a few NANs coming from the filtered a -- TODO check
        /*if (a != a)
        {
          printf(" a \n");
        }
        if (b != b)
        {
          printf(" b \n");
        }*/
      }
    }
  }

  if (_n > 0)
  {
    return (_xy - (_x * _y) / _n) / (sqrt(_x2 - _x * _x / _n) * sqrt(_y2 - _y *_y / _n));
  }
  else
  {
    return 0.0f;
  }

}


#define _EPS 0.0001f
//I am testing now a different approach, which is to make one thread to fully optimize one (small!!) patch
//also integrate a pre-computed Gaussian pyramid of the patches and the recon
//TODO this works but would be more efficent if deistributed on blocks instead of threads, whith CC reduction in shared memory
template <typename T>
__global__ void parallelPatchRegOptimization(PatchBasedVolume<T> inputStack, ReconVolume<T> reconstruction, Matrix4<T>* transfMats, /*T* testResults,*/ int level, /*T* gradientSpace, T* paramSpace,*/ T _step)
{
  const unsigned int patchIdx = blockIdx.x* blockDim.x + threadIdx.x;

  if (patchIdx >= inputStack.getXYZPatchGridSize().z)
    return;

  uint3 psize = make_uint3(inputStack.getXYZPatchGridSize().x, inputStack.getXYZPatchGridSize().y, 3);

  //actually I want that to be in a register... -> If done so I get NANs
  Matrix4<T>& currentMat = transfMats[patchIdx];

  //TODO estimating breaking point
  T parameterValue = 0;
  T params[6];
  Matrix2Parameters(currentMat, params);
  T dxt[6];

  //evaluate gradient
#pragma unroll
  for (int i = 0; i < 6; i++) {
    //dx[i] = 0;
    parameterValue = params[i];
    params[i] = parameterValue + _step;
    Parameters2Matrix(params, currentMat);
    T s1 = computeCCpatch<T>(inputStack, reconstruction, currentMat, patchIdx, psize, level);
    params[i] = parameterValue - _step;
    Parameters2Matrix(params, currentMat);
    T s2 = computeCCpatch<T>(inputStack, reconstruction, currentMat, patchIdx, psize, level);
    // float diff = s1 - s2;
    dxt[i] = s1 - s2; //diff;
    params[i] = parameterValue;
    Parameters2Matrix(params, currentMat);
  }

  T norm = 0;
#pragma unroll
  for (int i = 0; i < 6; i++) {
    norm += dxt[i] * dxt[i];
  }

  norm = sqrt(norm);
  if (norm > 0.0f) {
#pragma unroll
    for (int i = 0; i < 6; i++) {
      dxt[i] /= norm;
    }
  }
  else {
#pragma unroll
    for (int i = 0; i < 6; i++) {
      dxt[i] = 0;
    }
  }
  //evaluate gradient end

  T similarity = computeCCpatch<T>(inputStack, reconstruction, currentMat, patchIdx, psize, level);
  T new_similarity = similarity;
  int count = 0;

  // printf("similarity before = %f \n", similarity);

  //This is usually 1-2
  do {
    new_similarity = similarity;
    //manual unroll because of ptxas compiler bug
    params[0] = params[0] + _step * dxt[0];
    params[1] = params[1] + _step * dxt[1];
    params[2] = params[2] + _step * dxt[2];
    params[3] = params[3] + _step * dxt[3];
    params[4] = params[4] + _step * dxt[4];
    params[5] = params[5] + _step * dxt[5];
    Parameters2Matrix(params, currentMat);
    similarity = computeCCpatch<T>(inputStack, reconstruction, currentMat, patchIdx, psize, level);//_Registration->Evaluate();
    count++;
  } while (similarity > new_similarity + _EPS && count < 50);

  // printf("patchIdx %i - similarity %f - count %i \n", patchIdx, similarity, count);
  /*if (threadIdx.x == 0)
  {
    printf(" %f  \n", similarity);
  }*/

  // Last step was no improvement, so back track
#pragma unroll
  for (int i = 0; i < 6; i++) {
    params[i] = params[i] - _step * dxt[i];
  }
  Parameters2Matrix(params, currentMat);

  //testResults[patchIdx] = count;
}

////////////////////////////////////////////////////////
//construction and testing site
template <typename T>
__device__ void computeCCpatchRed(PatchBasedVolume<T>& inputStack, ReconVolume<T>& reconstruction, const Matrix4<T>& Tmat, unsigned int patchIdx/*, uint3& psize*/, int level,
  const unsigned int threads, const unsigned int localid, float* reduction, const uint3& patchPos)
{

  ImagePatch2D<T>& patch = inputStack.getImagePatch2D(patchIdx);

  reduction[localid] = 0;
  reduction[localid + threads] = 0;
  reduction[localid + 2 * threads] = 0;
  reduction[localid + 3 * threads] = 0;
  reduction[localid + 4 * threads] = 0;
  reduction[localid + 5 * threads] = 0;

  //for (int y = 0; y < psize.y; y += (level + 1))
  // {
  //   for (int x = 0; x < psize.x; x += (level + 1))
  //   {
#pragma unroll
  //for (int z = (int)(-(psize.z + 0.5f)) / 2; z < (int)((psize.z + 0.5f) / 2 + 1); z++ /*+= (level + 1)*/)
  for (int z = -1; z <= 1; z++ /*+= (level + 1)*/)
  {
    //a = -1.0f;
    T a = inputStack.getBufferValue(make_uint3(patchPos.x, patchPos.y, patchIdx)); //three gaussian levels!

    //TODO make double3 with T
    float3 patchPos = make_float3(patchPos.x, patchPos.y, z);
    float3 wpos = Tmat*patch.RI2W*patchPos;//Tr*patch.I2W*patchPos;  
    float3 volumePos = reconstruction.reconstructedW2I * wpos;

    //b = -1.0f;
    T b = reconstruction.getReconValueInterp(volumePos);
    //uint3 apos = make_uint3(volumePos.x, volumePos.y, volumePos.z);
    // b = reconstruction.getReconValue(apos);

    if (a >= 0.0f && b >= 0.0f && a == a && b == b) //TODO a few NANs coming from the filter...
    {
      reduction[localid] += a*b;
      reduction[localid + threads] += a;
      reduction[localid + 2 * threads] += b;
      reduction[localid + 3 * threads] += a*a;
      reduction[localid + 4 * threads] += b*b;
      reduction[localid + 5 * threads] += 1.0f;
    }
    //TODO why are there a few NANs coming from a?? -- Gaussian??
    /*if (a != a)
    {
      printf(" a \n");
    }
    if (b != b)
    {
      printf(" b \n");
    }*/
  }
  //   }
  // }
  __syncthreads();


  /*
  for (int n = threads / 2; n > 1; n /= 2)
  {
    if (localid < n){
      T a = reduction[localid + threads];
      T b = reduction[localid + 2 * threads];
      if (a >= 0.0f && b >= 0.0f)
      {
        reduction[localid] = reduction[localid] + reduction[localid + n],
          reduction[localid + threads] = reduction[localid + threads] + reduction[localid + threads + n],
          reduction[localid + 2 * threads] = reduction[localid + 2 * threads] + reduction[localid + 2 * threads + n],
          reduction[localid + 3 * threads] = reduction[localid + 3 * threads] + reduction[localid + 3 * threads + n],
          reduction[localid + 4 * threads] = reduction[localid + 4 * threads] + reduction[localid + 4 * threads + n],
          reduction[localid + 5 * threads] = reduction[localid + 5 * threads] + reduction[localid + 5 * threads + n];
      }
    }
    __syncthreads();
  }

  __syncthreads();
  if (localid == 0)
  {
    T _xy = reduction[0] + reduction[1];
    T _x = reduction[threads] + reduction[threads + 1];
    T _y = reduction[2 * threads] + reduction[2 * threads + 1];
    T _x2 = reduction[3 * threads] + reduction[3 * threads + 1];
    T _y2 = reduction[4 * threads] + reduction[4 * threads + 1];
    T _n = reduction[5 * threads] + reduction[5 * threads + 1];
    if (_n > 0 && _x >= 0.0f && _y >= 0.0f)
    {
      reduction[0] = (_xy - (_x * _y) / _n) / (sqrt(_x2 - _x * _x / _n) * sqrt(_y2 - _y *_y / _n));
      //printf(" %f %f %f %f %f %f %f \n", _xy, _x, _y, _x2, _y2, _n, reduction[0]);
    }
    else
    {
      // printf(" no samples \n");
      reduction[0] = 0.0f;
    }

    //reduction[0] = 1.0f;

  }
  __syncthreads();
  */
}

template <typename T>
__global__ void parallelPatchRegOptimizationRed(PatchBasedVolume<T> inputStack, ReconVolume<T> reconstruction, Matrix4<T>* transfMats, /*T* testResults,*/ int level, /*T* gradientSpace, T* paramSpace,*/ T _step)
{
  //note the different grid. One patch is processed by one block
  //const unsigned int patchIdx = blockIdx.x + blockIdx.y * blocks.x + blockIdx.z * blocks.x * blocks.y;
  // if (patchIdx >= inputStack.getXYZPatchGridSize().z)
  // return;

  const unsigned int threads = blockDim.x*blockDim.y*blockDim.z;
  const unsigned int localid = threadIdx.x + threadIdx.y*blockDim.x;// +blockDim.x*blockDim.y*threadIdx.z;
  const uint3 threadPatchPos = make_uint3((threadIdx.x) * (level + 1), (threadIdx.y) * (level + 1), 0);
  const unsigned int patchIdx = threadIdx.z + blockIdx.z* blockDim.z;

  if (patchIdx >= inputStack.getXYZPatchGridSize().z)
    return;

  if (threadPatchPos.x >= inputStack.getXYZPatchGridSize().x || threadPatchPos.y >= inputStack.getXYZPatchGridSize().y)
    return;

  extern __shared__ float reductionSpace[]; //blockDim.x * blockDim.y

 // uint3 psize = make_uint3(inputStack.getXYZPatchGridSize().x, inputStack.getXYZPatchGridSize().y, 3);

  //actually I want that to be in a register... -> If done, I get NANs
  Matrix4<T>& currentMat = transfMats[patchIdx];

  //TODO estimating breaking point
 // if (threadIdx.x == 0)
//  {
 /*   T parameterValue = 0;
    T params[6];
    Matrix2Parameters(currentMat, params);
    T dxt[6];
    T s1 = 0;
    T s2 = 0;*/
 // }

    //Test
    computeCCpatchRed<T>(inputStack, reconstruction, currentMat, patchIdx/*, psize*/, level, threads, localid, reductionSpace, threadPatchPos);

    if (localid == 0)
    {
      float res0 = reductionSpace[0];
      printf(" %f  \n", res0);
    }

}

//construction site end
///////////////////////////////////////////////////////


template <typename T>
void PatchBased2D3DRegistration_gpu2<T>::run()
{
  printf("PatchBased2D3DRegistration_gpu2 ");
  cerr.rdbuf(file_e.rdbuf());
  cout.rdbuf(file.rdbuf());

  checkCudaErrors(cudaSetDevice(m_cuda_device));

  int num_patches = m_inputStack->getXYZPatchGridSize().z;

  //Reset origin to avoid drift!
  initDevReconMatrices2<T> << <divup(num_patches, 512), 512 >> >(*m_inputStack, dev_recon_matrices_orig, dev_recon_matrices);
  CHECK_ERROR(initDevReconMatrices2);

 /* T* d_testoutput;
  checkCudaErrors(cudaMalloc(&d_testoutput, sizeof(T) * num_patches));

  T* d_gradientSpace;
  checkCudaErrors(cudaMalloc(&d_gradientSpace, sizeof(T) * num_patches * 6));

  T* d_paramSpace;
  checkCudaErrors(cudaMalloc(&d_paramSpace, sizeof(T) * num_patches * 6));*/

  int _NumberOfLevels = 4;
  for (int level = _NumberOfLevels - 1; level >= 0; level--) {
    T _TargetBlurring = m_Blurring[level];
    T _StepSize = m_LengthOfSteps[level];

    dim3 blockSize3 = dim3(m_inputStack->getXYZPatchGridSize().x, m_inputStack->getXYZPatchGridSize().y, 1); //max 1024 threads / direction
    dim3 gridSize3 = divup(dim3(m_inputStack->getXYZPatchGridSize().x / (level + 1), m_inputStack->getXYZPatchGridSize().y / (level + 1),
      m_inputStack->getXYZPatchGridSize().z), blockSize3);
    dim3 redgrid = divup(dim3(m_inputStack->getXYZPatchGridSize().x / (level + 1), m_inputStack->getXYZPatchGridSize().y / (level + 1),
      m_inputStack->getXYZPatchGridSize().z), dim3(blockSize3.x, blockSize3.y, blockSize3.z));

    //TODO precompute these if necessary.
    // printf("m_inputStack->getXYZPatchGridSize()= %d %d %d \n",m_inputStack->getXYZPatchGridSize().x,m_inputStack->getXYZPatchGridSize().y,m_inputStack->getXYZPatchGridSize().z);
    patchBasedFilterGaussStack(m_cuda_device, m_inputStack->getPatchesPtr(), m_inputStack->getBufferPtr(),
      m_inputStack->getXYZPatchGridSize(), num_patches, _TargetBlurring);

    cout.rdbuf(strm_buffer);
    cerr.rdbuf(strm_buffer_e);
    T _step = _StepSize;
    for (int st = 0; st < 4; st++)
    {
      for (int iter = 0; iter < 20; iter++)
      {
#if 1
       parallelPatchRegOptimization<T> << <divup(num_patches, 256), 256 >> >(*m_inputStack, *m_reconstruction, dev_recon_matrices, /*d_testoutput,*/ level, /*d_gradientSpace, d_paramSpace,*/ _step);
        CHECK_ERROR(parallelPatchRegOptimization);
#else
        //TODO start kernel in patchsize.x x patchsize.y x numpatches grid. Do reduction of CC per block in shared memory 
        //Test
        parallelPatchRegOptimizationRed<T> << <redgrid, blockSize3, blockSize3.x*blockSize3.y*blockSize3.z*sizeof(float) * 6 + sizeof(float) >> >(*m_inputStack, *m_reconstruction, dev_recon_matrices, /*d_testoutput,*/ level, /*d_gradientSpace, d_paramSpace,*/ _step);
        CHECK_ERROR(parallelPatchRegOptimizationRed);
#endif
      }
      advance_cursor2();
      _step /= 2.0f;
    }
    cerr.rdbuf(file_e.rdbuf());
    cout.rdbuf(file.rdbuf());
  }
 // T* h_testoutput = new T[num_patches];
 // checkCudaErrors(cudaMemcpy(h_testoutput, d_testoutput, num_patches*sizeof(T), cudaMemcpyDeviceToHost));
 /* checkCudaErrors(cudaFree(d_testoutput));
  checkCudaErrors(cudaFree(d_gradientSpace));
  checkCudaErrors(cudaFree(d_paramSpace));*/

 // printf("similarities: ");
 /* for (int i = 0; i < num_patches; i++)
  {
    printf(" %f ", h_testoutput[i]);
  }
  printf("\n");

  delete[] h_testoutput;*/

  resetOriginDevReconMatrices2<T> << <divup(num_patches, 512), 512 >> >(*m_inputStack, dev_recon_matrices);

  cout.rdbuf(strm_buffer);
  cerr.rdbuf(strm_buffer_e);

  Matrix4<T>* h_InvRecon_matrices_ = new Matrix4<T>[num_patches];
  checkCudaErrors(cudaMemcpy(h_InvRecon_matrices_, dev_recon_matrices, sizeof(Matrix4<T>)*num_patches, cudaMemcpyDeviceToHost));
  for (int i = 0; i < num_patches; i++)
  {
    irtkMatrix Tmat = fromMatrix4<T>(h_InvRecon_matrices_[i]);
    if (i > 100 && i < 110)
    {
      Tmat.Print();
    }
  }

  //copy matrices back to patches.Transform
  //TODO inverse on device
  Matrix4<T>* h_InvRecon_matrices = new Matrix4<T>[num_patches];
  checkCudaErrors(cudaMemcpy(h_InvRecon_matrices, dev_recon_matrices, sizeof(Matrix4<T>)*num_patches, cudaMemcpyDeviceToHost));
  for (int i = 0; i < num_patches; i++)
  {
    irtkMatrix Tmat = fromMatrix4<T>(h_InvRecon_matrices[i]);
    Tmat.Invert();
    h_InvRecon_matrices[i] = toMatrix4<T>(Tmat);
  }
  //std::cin.get();

  Matrix4<T>* d_InvRecon_matrices;
  checkCudaErrors(cudaMalloc(&d_InvRecon_matrices, sizeof(Matrix4<T>)*num_patches));
  checkCudaErrors(cudaMemcpy(d_InvRecon_matrices, h_InvRecon_matrices, sizeof(Matrix4<T>)*num_patches, cudaMemcpyHostToDevice));

  finishDevReconMatrices2<T> << <divup(num_patches, 512), 512 >> >(*m_inputStack, dev_recon_matrices, d_InvRecon_matrices);
  CHECK_ERROR(finishDevReconMatrices);

  checkCudaErrors(cudaFree(d_InvRecon_matrices));
  delete[] h_InvRecon_matrices;

  printf("\n");

}

template <typename T>
PatchBased2D3DRegistration_gpu2<T>::PatchBased2D3DRegistration_gpu2(int cuda_device, 
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

  checkCudaErrors(cudaMalloc(&dev_recon_matrices_orig, sizeof(Matrix4<T>)* m_inputStack->getXYZPatchGridSize().z));
  checkCudaErrors(cudaMalloc(&dev_recon_matrices, sizeof(Matrix4<T>)* m_inputStack->getXYZPatchGridSize().z));
  
  m_Blurring = new T[m_NumberOfLevels];
  m_LengthOfSteps = new T[m_NumberOfLevels];
  m_Blurring[0] = (m_reconstruction->m_dim.x) / 2.0f;
  for (int i = 0; i < m_NumberOfLevels; i++) {
    m_LengthOfSteps[i] = 2.0f * pow(2.0f, i);
  }
  for (int i = 1; i < m_NumberOfLevels; i++) {
    m_Blurring[i] = m_Blurring[i - 1] * 2;
  }

  //not required at the moment
  //prepareReconTex(cuda_device);
}

template <typename T>
PatchBased2D3DRegistration_gpu2<T>::~PatchBased2D3DRegistration_gpu2()
{
//  checkCudaErrors(cudaFreeArray(m_d_reconstructed_array));
  checkCudaErrors(cudaFree(dev_recon_matrices_orig));
  checkCudaErrors(cudaFree(dev_recon_matrices));
}

template class PatchBased2D3DRegistration_gpu2 < float >;
template class PatchBased2D3DRegistration_gpu2 < double >;
