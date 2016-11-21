#include <cuda_runtime_api.h>
#include <iostream>


void runTest(int device);
int main(int argc, char** argv)
{
  int cuda_device = argc > 1 ? atoi(argv[1]) : 0;
  runTest(cuda_device);
}