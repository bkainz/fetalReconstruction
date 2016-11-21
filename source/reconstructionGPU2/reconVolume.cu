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
#include "reconVolume.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/version.h>
#include <thrust/inner_product.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/count.h>
#include <thrust/system_error.h>

texture<float, 3, cudaReadModeElementType > reconTex_;

template< typename T >
class divS
{
public:
	T operator()(T a, T b)
	{
		return (b != 0) ? a / b : 0;
	}
};

template< typename T >
class divSame
{
public:
	T operator()(T a, T b)
	{
		return (b != 0) ? a / b : a;
	}
};

template <typename T>
__global__ void equalizeVol(T* recon, T* volWeights, uint3 m_size)
{
	const uint3 pos = make_uint3(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y,
		blockIdx.z * blockDim.z + threadIdx.z);

	if (pos.x >= m_size.x || pos.y >= m_size.y || pos.z >= m_size.z)
		return;

	unsigned int idx = pos.x + pos.y*m_size.x + pos.z * m_size.x*m_size.y;
	T a = recon[idx];
	T b = volWeights[idx];

	recon[idx] = (b != 0) ? a / b : a;
}

template <typename T>
void ReconVolume<T>::equalize(){
  unsigned int N = m_size.x*m_size.y*m_size.z;
  dim3 blockSize = dim3(8, 8, 8);
  dim3 gridSize = divup(dim3(m_size.x, m_size.y, m_size.z), blockSize);

  equalizeVol<T> << <gridSize, blockSize >> > (this->getDataPtr(), getReconstructed_volWeigthsPtr(), m_size);
  CHECK_ERROR(ReconVolume<T>::equalize());

  //this does not work with CUDA 7.5 -> fallback to classic kernel
 // try
 // {
	//  thrust::device_ptr<T> ptr_recons(getDataPtr());
	//  thrust::device_ptr<T> ptr_count(getReconstructed_volWeigthsPtr());
	//  thrust::transform(ptr_recons, ptr_recons + N, ptr_count, ptr_recons, divS<T>());
 /* }
  catch (thrust::system_error &e)
  {
	  // output an error message and exit
	  std::cerr << "Thrust error: " << e.what() << std::endl;
	  exit(-1);
  }*/
  //CHECK_ERROR(ReconVolume<T>::equalize());
  checkCudaErrors(cudaDeviceSynchronize());
}

template <typename T>
void ReconVolume<T>::updateReconTex(int dev)
{
  //this uses only float interpolation! careful with double
  checkCudaErrors(cudaSetDevice(dev));
  //works only for float interpolation -- will fail for double!!

  // ///////////////////////////////////////////////////////////////////////////////
  // // test code to fix memcheck error
  // const size_t SIZE_X = this->m_size.x;
  // const size_t SIZE_Y = this->m_size.y;
  // const size_t SIZE_Z = this->m_size.z;
  // const size_t width  = sizeof(float) * SIZE_X;

  // cudaExtent volumeSizeBytes = make_cudaExtent(width, SIZE_Y, SIZE_Z);
  // // cudaPitchedPtr d_volumeMem; 
  // // checkCudaErrors(cudaMalloc3D(&d_volumeMem, volumeSizeBytes));
  // // size_t size = d_volumeMem.pitch * SIZE_Y * SIZE_Z;

  // cudaChannelFormatDesc m_channelDesc = cudaCreateChannelDesc<float>();
  // cudaExtent volumeSize = make_cudaExtent(SIZE_X, SIZE_Y, SIZE_Z);
  
  // //initialize the 3d texture "tex" with a 3D array "d_volumeArray"
  // cudaArray* m_d_reconstructed_array;
  // checkCudaErrors( cudaMalloc3DArray(&m_d_reconstructed_array, &m_channelDesc, volumeSize) ); 

  // reconTex_.normalized     = true;                  // access with normalized texture coordinates      
  // reconTex_.filterMode     = cudaFilterModeLinear;  // linear interpolation   
  // reconTex_.addressMode[0] = cudaAddressModeClamp;   // clamp texture coordinates
  // reconTex_.addressMode[1] = cudaAddressModeClamp;
  // reconTex_.addressMode[2] = cudaAddressModeClamp;
  // // bind array to 3D texture
  // checkCudaErrors(cudaBindTextureToArray(reconTex_, m_d_reconstructed_array, m_channelDesc));

  // //get the real value for 3D texture "tex"
  // float *d_volumeMem;
  // float *f_m_d_data = (float*) m_d_data;
  // checkCudaErrors(cudaMalloc((void**)&d_volumeMem, SIZE_X*SIZE_Y*SIZE_Z*sizeof(float)));
  // checkCudaErrors(cudaMemcpy(d_volumeMem, f_m_d_data, SIZE_X*SIZE_Y*SIZE_Z*sizeof(float), cudaMemcpyHostToDevice));

  // //copy d_volumeMem to 3DArray
  // cudaMemcpy3DParms copyParams = {0};
  // copyParams.srcPtr   = make_cudaPitchedPtr((void*)d_volumeMem, SIZE_X*sizeof(float), SIZE_X, SIZE_Y);
  // copyParams.dstArray = m_d_reconstructed_array;
  // copyParams.extent   = volumeSize;
  // copyParams.kind     = cudaMemcpyDeviceToDevice;
  // checkCudaErrors( cudaMemcpy3D(&copyParams) ); 

  // ///////////////////////////////////////////////////////////////////////////////
  
  cudaMemcpy3DParms copyParams = { 0 };
  copyParams.srcPtr = make_cudaPitchedPtr((void*)this->m_d_data, this->m_size.x*sizeof(float),this->m_size.x, this->m_size.y);
  copyParams.dstArray = m_d_reconstructed_array;//a1;// dev_reconstructed_array[dev];
  copyParams.extent = m_asize;
  copyParams.kind = cudaMemcpyDeviceToDevice;
  checkCudaErrors(cudaMemcpy3D(&copyParams));

  reconTex_.addressMode[0] = cudaAddressModeBorder;
  reconTex_.addressMode[1] = cudaAddressModeBorder;
  reconTex_.addressMode[2] = cudaAddressModeBorder;
  reconTex_.filterMode = cudaFilterModeLinear;
  reconTex_.normalized = true;

  checkCudaErrors(cudaBindTextureToArray(reconTex_, m_d_reconstructed_array, m_channelDesc));
  CHECK_ERROR(cudaBindTextureToArray);
}

template <typename T>
__device__ const T & ReconVolume<T>::getReconValueFromTexture(const uint3 & pos)
{
  if (pos.x >= m_size.x || pos.y >= m_size.y || pos.z >= m_size.z)
      return;

  // float x = (float)pos.x / m_size.x;
  // float y = (float)pos.y / m_size.y;
  // float z = (float)pos.z / m_size.z;

  // float x = float(pos.x)+0.5f;
  // float y = float(pos.y)+0.5f;
  // float z = float(pos.z)+0.5f;

  // unsigned int idx = pos.x + pos.y*m_size.x + pos.z*m_size.x*m_size.y;
  T val = (T)tex3D(reconTex_, (T)pos.x / m_size.x, (T)pos.y / m_size.y, (T)pos.z / m_size.z);

  return val;
}


template class ReconVolume < float >;
template class ReconVolume < double >;

//template void ReconVolume<float>::equalize();
//template void ReconVolume<double>::equalize();