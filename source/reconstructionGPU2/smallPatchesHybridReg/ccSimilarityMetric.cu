
#include "ccSimilarityMetric.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

//#include "../interpFunctions.cuh"

//TODO texture is unfortunately not thread safe
texture<float, 3, cudaReadModeElementType > reconstructedTexFloat_;



inline __host__ __device__ float fracf(float v)
{
  return v - floorf(v);
}
inline __host__ __device__ float3 fracf(float3 v)
{
  return make_float3(fracf(v.x), fracf(v.y), fracf(v.z));
}
inline __host__ __device__ int3 operator-(int3 a, int3 b)
{
  return make_int3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ int3 operator+(int3 a, int3 b)
{
  return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ int3 make_int3(float3 a)
{
  return make_int3(int(a.x), int(a.y), int(a.z));
}
inline __host__ __device__ int3 make_int3(uint3 a)
{
  return make_int3(int(a.x), int(a.y), int(a.z));
}
inline __host__ __device__ float3 floorf(float3 v)
{
  return make_float3(floorf(v.x), floorf(v.y), floorf(v.z));
}
inline __host__ __device__ int3 max(int3 a, int3 b)
{
  return make_int3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}
inline __host__ __device__ int3 min(int3 a, int3 b)
{
  return make_int3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}
inline __host__ __device__ int3 make_int3(int s)
{
  return make_int3(s, s, s);
}

__device__ float v_(const uint3 & pos, float* data, const uint3 size) {
  float val = 0.0f;
  if (pos.x < size.x && pos.y < size.y && pos.z < size.z)
    val = data[pos.x + pos.y * size.x + pos.z * size.x * size.y]; 
 
  return val;
}

__device__ float interp_(const float3 & pos, float* data, const uint3 size, const float3 dim) {
  //const float3 scaled_pos = make_float3((pos.x * size.x / dim.x) - 0.5f, (pos.y * size.y / dim.y) - 0.5f, (pos.z * size.z / dim.z) - 0.5f);
  const float3 scaled_pos = make_float3((pos.x), (pos.y), (pos.z));

  const int3 base = make_int3(floorf(scaled_pos));
  const float3 factor = fracf(scaled_pos);
  const int3 lower = max(base, make_int3(0));
  const int3 upper = min(base + make_int3(1), make_int3(size) - make_int3(1));
  return v_(make_uint3(lower.x, lower.y, lower.z), data, size) * (1 - factor.x) * (1 - factor.y) * (1 - factor.z)
    + v_(make_uint3(upper.x, lower.y, lower.z), data, size) * factor.x * (1 - factor.y) * (1 - factor.z)
    + v_(make_uint3(lower.x, upper.y, lower.z), data, size) * (1 - factor.x) * factor.y * (1 - factor.z)
    + v_(make_uint3(upper.x, upper.y, lower.z), data, size) * factor.x * factor.y * (1 - factor.z)
    + v_(make_uint3(lower.x, lower.y, upper.z), data, size) * (1 - factor.x) * (1 - factor.y) * factor.z
    + v_(make_uint3(upper.x, lower.y, upper.z), data, size) * factor.x * (1 - factor.y) * factor.z
    + v_(make_uint3(lower.x, upper.y, upper.z), data, size) * (1 - factor.x) * factor.y * factor.z
    + v_(make_uint3(upper.x, upper.y, upper.z), data, size) * factor.x * factor.y * factor.z;
}


__global__ void computeCCAndReduce(float* target, float* source,
  transformations transf_, /*float *d_f,
  float* sums, int* counts, float* averages,*/ float* results, uint3 tsize, uint3 ssize, float3 sdim)
{

  int threads = blockDim.x*blockDim.y*blockDim.z;
  int localid = threadIdx.x + blockDim.x*threadIdx.y + blockDim.x*blockDim.y*threadIdx.z;

  if (localid == 0)
  {
    results[0] = 0;
    results[1] = 0;
    results[2] = 0;
    results[3] = 0;
    results[4] = 0;
    results[5] = 0;
   /* counts[0] = 0;
    counts[1] = 0;
    sums[0] = 0;
    sums[1] = 0;
    averages[0] = 0;
    averages[1] = 0;*/
    //*d_f = FLT_MAX;
  }

  //average computation
  int width = tsize.x;// inputStack.getXYZPatchGridSize().x;
  int height = tsize.y;// inputStack.getXYZPatchGridSize().y;
  int depth = tsize.z;
  /*
  int myActive = 0;
  float myCount = 0;
  int myActive1 = 0;
  float myCount1 = 0;
  for (int z = blockIdx.z*blockDim.z + threadIdx.z; z < depth; z += blockDim.z*gridDim.z)
    for (int y = blockIdx.y*blockDim.y + threadIdx.y; y < height; y += blockDim.y*gridDim.y)
      for (int x = blockIdx.x*blockDim.x + threadIdx.x; x < width; x += blockDim.x*gridDim.x)
      {
        float val = -1.0f;
        if (x < tsize.x && y < tsize.y && z < tsize.z)
        val = target[x + y * tsize.x + z * tsize.x * tsize.y];//inputStack.getRegPatchValue(make_uint3(x, y, patchN));
        if (val >= 0.0f)
          ++myActive, myCount += val;

        float3 patchPos = make_float3(x, y, z);
        float3 wpos = transf_.Tmat*transf_.targetI2W*patchPos;
        //Tr*patch.RI2W*patchPos;//Tr*patch.I2W*patchPos;  
        float3 volumePos = transf_.sourceW2I * wpos;
        // reconstruction.reconstructedW2I * wpos;
        // uint3 apos = make_uint3(volumePos.x + 0.5f, volumePos.y + 0.5f, volumePos.z + 0.5f);
        float val1 = -1.0f;
        //if (apos.x < ssize.x && apos.y < ssize.y && apos.z < ssize.z)
        val1 = interp(volumePos, source, ssize, sdim);
        //val1 = source[apos.x + apos.y*ssize.x + apos.z*ssize.x*ssize.y];
        //val1 = tex3D(reconstructedTexFloat_, volumePos.x / ssize.x, volumePos.y / ssize.y, volumePos.z / ssize.z);

        if (val1 >= 0.0f && val >= 0.0f)
          ++myActive1, myCount1 += val1;
      }
  
  extern __shared__ float reductionSpace[];
  float* f_reduction = reinterpret_cast<float*>(reductionSpace + threads);
  float* f_reduction1 = reinterpret_cast<float*>(reductionSpace + threads * 2);
  float* reductionSpace1 = reinterpret_cast<float*>(reductionSpace + threads * 3);
  reductionSpace[localid] = myActive;
  f_reduction[localid] = myCount;
  reductionSpace1[localid] = myActive1;
  f_reduction1[localid] = myCount1;

  __syncthreads();

  for (int n = threads / 2; n > 1; n /= 2)
  {
    if (localid < n)
      reductionSpace[localid] += reductionSpace[localid + n],
      f_reduction[localid] += f_reduction[localid + n],
      reductionSpace1[localid] += reductionSpace1[localid + n],
      f_reduction1[localid] += f_reduction1[localid + n];
    __syncthreads();
  }

  if (threadIdx.x == 0)
  {
    atomicAdd(&sums[0], f_reduction[0] + f_reduction[1]),
      atomicAdd(&counts[0], reductionSpace[0] + reductionSpace[1]),
      atomicAdd(&sums[1], f_reduction1[0] + f_reduction1[1]),
      atomicAdd(&counts[1], reductionSpace1[0] + reductionSpace1[1]);
  }
  __syncthreads();

  if (localid == 0)
  {
    if (counts[0] > 0)
      averages[0] = sums[0] / counts[0];
    if (counts[1] > 0)
      averages[1] = sums[1] / counts[1];

    //printf("%f %f \n", averages[0], averages[1]);
  }
  */
  //CC computation
  float3 values = make_float3(0, 0, 0);
  float3 values1 = make_float3(0, 0, 0);

  for (int z = blockIdx.z*blockDim.z + threadIdx.z; z < depth; z += blockDim.z*gridDim.z)
    for (int y = blockIdx.y*blockDim.y + threadIdx.y; y < height; y += blockDim.y*gridDim.y)
      for (int x = blockIdx.x*blockDim.x + threadIdx.x; x < width; x += blockDim.x*gridDim.x)
      {
        //float a = inputStack.getRegPatchValue(make_uint3(x, y, patchN));
        // float b = inputStack.getBufferValue(make_uint3(x, y, patchN));
        float _x = -1.0f;
        if (x < tsize.x && y < tsize.y && z < tsize.z)
          _x = target[x + y * tsize.x + z * tsize.x * tsize.y];//inputStack.getRegPatchValue(make_uint3(x, y, patchN));
       // float a = target[x + y * tsize.x + z * tsize.x * tsize.y];//inputStack.getRegPatchValue(make_uint3(x, y, patchN));
        //TODO transform to source space and interpolate via texture
        //float b = source[x + y * ssize.x + z * ssize.x * ssize.y];//inputStack.getBufferValue(make_uint3(x, y, patchN));
        float3 patchPos = make_float3(x, y, z);
        float3 wpos = transf_.Tmat*transf_.targetI2W*patchPos;
        //Tr*patch.RI2W*patchPos;//Tr*patch.I2W*patchPos;  
        float3 volumePos = transf_.sourceW2I * wpos;
        // reconstruction.reconstructedW2I * wpos;

        //uint3 apos = make_uint3(volumePos.x + 0.5f, volumePos.y + 0.5f, volumePos.z + 0.5f);
        float _y = -1.0f;
        // if (apos.x < ssize.x && apos.y < ssize.y && apos.z < ssize.z)
        //b = source[apos.x + apos.y*ssize.x + apos.z*ssize.x*ssize.y];
        _y = interp_(volumePos, source, ssize, sdim);
        // b = tex3D(reconstructedTexFloat_, volumePos.x / ssize.x, volumePos.y / ssize.y, volumePos.z / ssize.z);

        if (_x >= 0.0f && _y >= 0.0f)
        {
          //float _x = a;//a - averages[0];
          //float _y = b;//b - averages[1];
          values = values + make_float3(_x*_y, _x, _x*_x);
          values1 = values1 + make_float3(_y, _y*_y, 1.0f);
        }
      }

  //reuse reduction space
  //float* reduction = reinterpret_cast<float*>(reductionSpace);
  extern __shared__ float reduction[];
  reduction[localid] = values.x;
  reduction[localid + threads] = values.y;
  reduction[localid + 2 * threads] = values.z;
  reduction[localid + 3 * threads] = values1.x;
  reduction[localid + 4 * threads] = values1.y;
  reduction[localid + 5 * threads] = values1.z;

  __syncthreads();

  for (int n = threads / 2; n > 1; n /= 2)
  {
    if (localid < n){
      reduction[localid] = reduction[localid] + reduction[localid + n],
        reduction[localid + threads] = reduction[localid + threads] + reduction[localid + threads + n],
        reduction[localid + 2 * threads] = reduction[localid + 2 * threads] + reduction[localid + 2 * threads + n],
        reduction[localid + 3 * threads] = reduction[localid + 3 * threads] + reduction[localid + 3 * threads + n],
        reduction[localid + 4 * threads] = reduction[localid + 4 * threads] + reduction[localid + 4 * threads + n],
        reduction[localid + 5 * threads] = reduction[localid + 5 * threads] + reduction[localid + 5 * threads + n];
    }
    __syncthreads();
  }

  __syncthreads();

  if (threadIdx.x == 0)
  {
    atomicAdd(results + 0, reduction[0] + reduction[1]);
    atomicAdd(results + 1, reduction[threads] + reduction[threads + 1]);
    atomicAdd(results + 2, reduction[2 * threads] + reduction[2 * threads + 1]);
    atomicAdd(results + 3, reduction[3 * threads] + reduction[3 * threads + 1]);
    atomicAdd(results + 4, reduction[4 * threads] + reduction[4 * threads + 1]);
    atomicAdd(results + 5, reduction[5 * threads] + reduction[5 * threads + 1]);
  }

/*  __syncthreads();

  if (localid == 0)
  {
    //
    float _xy = results[0];
    float _x = results[1];
    float _x2 = results[2];
    float _y = results[3];
    float _y2 = results[4];
    float _n = results[5];

    float no = (sqrt(_x2 - _x * _x / _n) * sqrt(_y2 - _y *_y / _n));

    if (_n > 0 && no > 0)
    {
      *d_f = (_xy - (_x * _y) / _n) / no ;
    }
    else
    {
      //no samples
      *d_f = 0.0;
    }
   // printf("%f \n", res);
  }*/

}

ccSimilarityMetric::ccSimilarityMetric(int cuda_device) : device(cuda_device), 
m_d_reconstructed_array(NULL), d_source(NULL), d_target(NULL)
{
 // checkCudaErrors(cudaMalloc(&d_m_f, sizeof(float)));
  //checkCudaErrors(cudaMalloc(&d_sums, sizeof(float) * 2));
 // checkCudaErrors(cudaMalloc(&d_counts, sizeof(int) * 2));
 // checkCudaErrors(cudaMalloc(&d_averages, sizeof(float) * 2));
 // checkCudaErrors(cudaMalloc(&d_results, sizeof(float) * 6));
}

ccSimilarityMetric::~ccSimilarityMetric()
{
  freeGPU();
  //checkCudaErrors(cudaFree(d_sums));
  //checkCudaErrors(cudaFree(d_counts));
  //checkCudaErrors(cudaFree(d_averages));
 // checkCudaErrors(cudaFree(d_results));
}

void ccSimilarityMetric::freeGPU()
{
  //checkCudaErrors(cudaFreeArray(m_d_reconstructed_array));
  checkCudaErrors(cudaFree(d_source));
  checkCudaErrors(cudaFree(d_target));
  //m_d_reconstructed_array = NULL;
  d_source = NULL;
  d_target = NULL;
}

void ccSimilarityMetric::prepareReconTex()
{
  checkCudaErrors(cudaSetDevice(device));

  //works only for float interpolation -- will fail for double!!
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

  cudaExtent asize;
  asize.width = source_size.x;
  asize.height = source_size.y;
  asize.depth = source_size.z;

  if (m_d_reconstructed_array != NULL)
    checkCudaErrors(cudaFreeArray(m_d_reconstructed_array));
  checkCudaErrors(cudaMalloc3DArray(&m_d_reconstructed_array, &channelDesc, asize));

  cudaMemcpy3DParms copyParams = { 0 };
  copyParams.srcPtr = make_cudaPitchedPtr((void*)d_source, source_size.x*sizeof(float),
    source_size.x, source_size.y);
  copyParams.dstArray = m_d_reconstructed_array;
  copyParams.extent = asize;
  copyParams.kind = cudaMemcpyDeviceToDevice;
  checkCudaErrors(cudaMemcpy3D(&copyParams));

  reconstructedTexFloat_.addressMode[0] = cudaAddressModeBorder;
  reconstructedTexFloat_.addressMode[1] = cudaAddressModeBorder;
  reconstructedTexFloat_.addressMode[2] = cudaAddressModeBorder;
  reconstructedTexFloat_.filterMode = cudaFilterModeLinear;
  reconstructedTexFloat_.normalized = true;

  checkCudaErrors(cudaBindTextureToArray(reconstructedTexFloat_, m_d_reconstructed_array, channelDesc));
  CHECK_ERROR(cudaBindTextureToArray);
  checkCudaErrors(cudaDeviceSynchronize());
}



__global__ void castToFloat(short* __restrict d_in, float* d_out, unsigned int num)
{
  const int idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (idx < num)
    d_out[idx] = (float)d_in[idx];
}


void ccSimilarityMetric::setTarget(const short* h_in, uint3 target_size_)
{
  bool alloc = false;
  if (target_size.x != target_size_.x || target_size.y != target_size_.y || target_size.z != target_size_.z ||
    d_target == NULL)
    alloc = true;

  target_size = target_size_;
  unsigned int num = target_size.x*target_size.y*target_size.z;
  if (num <= 0)
  {
    printf("0 values\n");
    return;
  }
  //TODO TBB uses a lot of memory (many threads)
  //checkGPUMemory();

  short* d_in;
  checkCudaErrors(cudaMalloc(&d_in, sizeof(short) * num));
  checkCudaErrors(cudaMemcpy(d_in, h_in, num*sizeof(short), cudaMemcpyHostToDevice));

  if (alloc)
  {
    checkCudaErrors(cudaFree(d_target));
    checkCudaErrors(cudaMalloc(&d_target, sizeof(float) * num));
  }

  castToFloat << <divup(num, 512), 512 >> >(d_in, d_target, num);
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaFree(d_in));
}

void ccSimilarityMetric::setSource(const short* h_in, uint3 source_size_, float3 source_dim_)
{
  bool alloc = false;
  if (source_size.x != source_size_.x || source_size.y != source_size_.y || source_size.z != source_size_.z ||
    d_source == NULL)
    alloc = true;

  source_size = source_size_;
  source_dim = source_dim_;

  //printf("%f %f %f \n", source_dim.x, source_dim.y, source_dim.z);

  unsigned int num = source_size.x*source_size.y*source_size.z;
  if (num <= 0)
  {
    printf("0 values\n");
    return;
  }

  short* d_in;
  checkCudaErrors(cudaMalloc(&d_in, sizeof(short) * num));
  checkCudaErrors(cudaMemcpy(d_in, h_in, num*sizeof(short), cudaMemcpyHostToDevice));

  if (alloc)
  {
    checkCudaErrors(cudaFree(d_source));
    checkCudaErrors(cudaMalloc(&d_source, sizeof(float) * num));
  }

  castToFloat << <divup(num, 512), 512 >> >(d_in, d_source, num);
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaFree(d_in));

  //prepareReconTex();
}


void ccSimilarityMetric::checkGPUMemory()
{
  size_t free_byte;
  size_t total_byte;
  cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);

  if (cudaSuccess != cuda_status){
    printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status));
  }

  float free_db = (float)free_byte;
  float total_db = (float)total_byte;
  float used_db = total_db - free_db;
  printf("GPU memory usage: \nused = %f, free = %f MB, total = %f MB\n",
    used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);
}


float ccSimilarityMetric::evaluate()
{
  dim3 redblock = dim3(4, 4, 4);
  dim3 patchessize(target_size.x, target_size.y, target_size.z);
  dim3 redgrid = divup(patchessize, dim3(redblock.x * 2, redblock.y * 2, redblock.z * 2));

 // printf("shared: %f KB \n", (redblock.x*redblock.y*redblock.z*sizeof(float) * 6 + sizeof(float)) / 1024.0f);
//  float* d_f;
 // checkCudaErrors(cudaMalloc(&d_f, sizeof(float)));

  float* d_results_;
  checkCudaErrors(cudaMalloc(&d_results_, sizeof(float) * 6));

  computeCCAndReduce << <redgrid, redblock, redblock.x*redblock.y*redblock.z*sizeof(float) * 6 + sizeof(float) >> >(d_target,
    d_source, m_t, /*d_f, d_sums, d_counts, d_averages,*/ d_results_, target_size, source_size, source_dim);
  checkCudaErrors(cudaDeviceSynchronize());

  float h_results_[6];
  checkCudaErrors(cudaMemcpy(h_results_, d_results_, 6*sizeof(float), cudaMemcpyDeviceToHost));

  float _xy = h_results_[0];
  float _x = h_results_[1];
  float _x2 = h_results_[2];
  float _y = h_results_[3];
  float _y2 = h_results_[4];
  float _n = h_results_[5];

  float h_f = 0.0;
  if (_n > 0)
  {
    h_f = (_xy - (_x * _y) / _n) / (sqrt(_x2 - _x * _x / _n) * sqrt(_y2 - _y *_y / _n));
  }

  checkCudaErrors(cudaFree(d_results_));

  return h_f;
}