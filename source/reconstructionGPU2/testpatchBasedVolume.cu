
#include "patchBasedVolume.cuh"
#include "patchBasedLayeredSurface3D.cuh"
#include <irtkImage.h>
#include "reconVolume.cuh"

///////////////////////////////test

__global__ void testKernel(PatchBasedVolume<float> testVol)
{
  const uint3 pos = make_uint3(blockIdx.x* blockDim.x + threadIdx.x,
    blockIdx.y* blockDim.y + threadIdx.y,
    blockIdx.z* blockDim.z + threadIdx.z);

  //set every voxel of testVol to pos
//  testVol.setValueFromPatchCoords(pos, pos.x*pos.y*pos.z / (float)(testVol.getSize().x*testVol.getSize().y*testVol.getSize().z));
}


__global__ void testDaemonKernel(PatchBasedVolume<float> testVol)
{
  const uint3 pos = make_uint3(blockIdx.x* blockDim.x + threadIdx.x,
    blockIdx.y* blockDim.y + threadIdx.y,
    blockIdx.z* blockDim.z + threadIdx.z);

  float v = testVol.getValueFromPatchCoords(pos);
  testVol.setWeightValue(pos, v);
}

__global__ void testCopyKernel(PatchBasedVolume<float> testVol, PatchBasedVolume<float> testVol2)
{
  const uint3 pos = make_uint3(blockIdx.x* blockDim.x + threadIdx.x,
    blockIdx.y* blockDim.y + threadIdx.y,
    blockIdx.z* blockDim.z + threadIdx.z);

  //set every voxel of testVol to pos
  float v = testVol.getValueFromPatchCoords(pos);
//  testVol2.setValueFromPatchCoords(pos, v);
}

__global__ void testSurfaceKernel(PatchBasedLayeredSurface3D<float> testVol)
{
  const uint3 pos = make_uint3(blockIdx.x* blockDim.x + threadIdx.x,
    blockIdx.y* blockDim.y + threadIdx.y,
    blockIdx.z* blockDim.z + threadIdx.z);
  
  //read value
  float v = testVol.getValueFromPatchCoords(pos);
  v += pos.x*pos.y*pos.z / (float)(testVol.getSize().x*testVol.getSize().y*testVol.getSize().z);
  //set every voxel of testVol to pos
  testVol.setValueFromPatchCoords(pos, v);
}


/*__global__ void testContainerKernel(PatchBasedReconContainer<float> testContainer, uint3 MaxSize)
{
  const uint3 pos = make_uint3(blockIdx.x* blockDim.x + threadIdx.x,
    blockIdx.y* blockDim.y + threadIdx.y,
    blockIdx.z* blockDim.z + threadIdx.z);

  if (pos.x >= MaxSize.x || pos.y >= MaxSize.y || pos.z >= MaxSize.z)
    return;

  PatchBasedVolume<float> vol = testContainer.getPatchBasedVolumeD(0);
  float v = testContainer.getSliceValueFromPatchCoords(0, pos);
  testContainer.setWeightValue(0, pos, v);

}*/

void runTest(int cuda_device)
{
  cudaSetDevice(cuda_device);
  cudaDeviceReset();

  irtkGenericImage<float> img;
  irtkImageAttributes attr;
  attr._x = 150;
  attr._y = 110;
  attr._z = 82;
  img.Initialize(attr);
  img = 0;

  uint2 patchSize = make_uint2(32, 32);
  uint2 patchStride = make_uint2(16, 16);

  irtkRigidTransformation *rigidTransf = new irtkRigidTransformation;

  PatchBasedVolume<float> testVol;
  testVol.init(img/*make_uint3(img.GetX(), img.GetY(), img.GetZ()), make_float3(1, 1, 1)*/, *rigidTransf, patchSize, patchStride);

  checkCudaErrors(cudaMemcpy(testVol.getDataPointer(), img.GetPointerToVoxels(), img.GetX()*img.GetY()*img.GetZ()*sizeof(float), cudaMemcpyHostToDevice));

  dim3 blockSize3 = dim3(8, 8, 8); //max 1024 threads
  dim3 gridSize3 = divup(dim3(testVol.getXYZPatchGridSize().x, testVol.getXYZPatchGridSize().y, testVol.getXYZPatchGridSize().z), blockSize3);
  printf("grid %d %d %d\n", gridSize3.x, gridSize3.y, gridSize3.z);
  testKernel <<<gridSize3, blockSize3>>>(testVol);
  CHECK_ERROR(testKernel);
  checkCudaErrors(cudaDeviceSynchronize());
  CHECK_ERROR(testKernel);
  checkCudaErrors(cudaMemcpy(img.GetPointerToVoxels(), testVol.getDataPointer(), img.GetX()*img.GetY()*img.GetZ()*sizeof(float), cudaMemcpyDeviceToHost));

  img.Write("testtestKernel.nii");

  PatchBasedVolume<float> testVol2;
  testVol2.init(img/*make_uint3(img.GetX(), img.GetY(), img.GetZ()), make_float3(1, 1, 1)*/, *rigidTransf, patchSize, patchStride);

  gridSize3 = divup(dim3(testVol.getXYZPatchGridSize().x, testVol.getXYZPatchGridSize().y, testVol.getXYZPatchGridSize().z), blockSize3);
  printf("grid %d %d %d\n", gridSize3.x, gridSize3.y, gridSize3.z);
  testCopyKernel << <gridSize3, blockSize3 >> >(testVol, testVol2);
  CHECK_ERROR(testCopyKernel);
  checkCudaErrors(cudaDeviceSynchronize());
  CHECK_ERROR(testCopyKernel);
  checkCudaErrors(cudaMemcpy(img.GetPointerToVoxels(), testVol2.getDataPointer(), img.GetX()*img.GetY()*img.GetZ()*sizeof(float), cudaMemcpyDeviceToHost));

  img.Write("testtestCopyKernel.nii");

  gridSize3 = divup(dim3(testVol.getXYZPatchGridSize().x, testVol.getXYZPatchGridSize().y, testVol.getXYZPatchGridSize().z), blockSize3);
  testDaemonKernel << <gridSize3, blockSize3 >> >(testVol);
  CHECK_ERROR(testDaemonKernel);
  checkCudaErrors(cudaDeviceSynchronize());
  CHECK_ERROR(testDaemonKernel);
  irtkGenericImage<float> imgW;
  // irtkImageAttributes attr;
  attr._x = testVol.getXYZPatchGridSize().x;
  attr._y = testVol.getXYZPatchGridSize().y;
  attr._z = testVol.getXYZPatchGridSize().z;
  imgW.Initialize(attr);
  imgW = 0;
  checkCudaErrors(cudaMemcpy(imgW.GetPointerToVoxels(), testVol.getWeigthDataPtr(), testVol.getXYZPatchGridSize().x*testVol.getXYZPatchGridSize().y*
    testVol.getXYZPatchGridSize().z*sizeof(float), cudaMemcpyDeviceToHost));
  imgW.Write("testtestDaemonKernel.nii");

#if 0 
  PatchBasedLayeredSurface3D<float> test_layerd_surface;
  test_layerd_surface.init(make_uint3(img.GetX(), img.GetY(), img.GetZ()), make_float3(1, 1, 1));
  test_layerd_surface.copyFromHost(img.GetPointerToVoxels());
  test_layerd_surface.generate2DPatches(make_uint2(64, 64), make_uint2(32, 32));

  blockSize3 = dim3(4, 4, 64); //max 1024 threads
  gridSize3 = divup(dim3(test_layerd_surface.getXYZPatchGridSize().x, test_layerd_surface.getXYZPatchGridSize().y, test_layerd_surface.getXYZPatchGridSize().z), blockSize3);
  printf("grid %d %d %d\n", gridSize3.x, gridSize3.y, gridSize3.z);
  testSurfaceKernel<<<gridSize3, blockSize3>>>(test_layerd_surface);
  CHECK_ERROR(testSurfaceKernel);
  checkCudaErrors(cudaDeviceSynchronize());
  CHECK_ERROR(testSurfaceKernel);

  test_layerd_surface.copyToHost(img.GetPointerToVoxels());
  img.Write("test2.nii");
#endif

 // PatchBasedVolume<float> testVol2;
 // testVol2.init(make_uint3(img.GetX(), img.GetY(), img.GetZ()), make_float3(1, 1, 1));
 // testVol2.generate2DPatches(make_uint2(64, 64), make_uint2(32, 32));

 /* PatchBasedReconContainer<float> container;
  std::vector<PatchBasedVolume<float>> stacks;
  stacks.push_back(testVol);
  container.init(stacks, make_uint3(120, 120, 120), make_float3(0.5, 0.5, 0.5));
  CHECK_ERROR(container.init);

  blockSize3 = dim3(4, 4, 64); //max 1024 threads
  gridSize3 = divup(dim3(testVol.getXYZPatchGridSize().x, testVol.getXYZPatchGridSize().y, testVol.getXYZPatchGridSize().z), blockSize3);
  printf("grid %d %d %d\n", gridSize3.x, gridSize3.y, gridSize3.z);
  testContainerKernel << <gridSize3, blockSize3 >> >(container, testVol.getXYZPatchGridSize());
  CHECK_ERROR(testContainerKernel);
  checkCudaErrors(cudaDeviceSynchronize());
  
  irtkGenericImage<float> imgW;
 // irtkImageAttributes attr;
  attr._x = testVol.getXYZPatchGridSize().x;
  attr._y = testVol.getXYZPatchGridSize().y;
  attr._z = testVol.getXYZPatchGridSize().z;
  imgW.Initialize(attr);
  imgW = 0;
  checkCudaErrors(cudaMemcpy(imgW.GetPointerToVoxels(), container.getWeigthDataPtrH(0), testVol.getXYZPatchGridSize().x*testVol.getXYZPatchGridSize().y*
    testVol.getXYZPatchGridSize().z*sizeof(float), cudaMemcpyDeviceToHost));
  imgW.Write("test3.nii");*/


  testVol.release();


  ReconVolume<float> reconVolume;

}