# check if we have c++11
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
	message("USING -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
	message("USING -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()



# CUDA
option(CUDA_BUILD_EMULATION "enable emulation mode" OFF)
find_package(CUDA REQUIRED COMPONENTS sdk npp thrust)
# FindCUDA.cmake of CMake 2.8.7 does not look for NPP
if (NOT DEFINED CUDA_npp_LIBRARY AND COMMAND find_cuda_helper_libs AND NOT CUDA_VERSION VERSION_LESS "4.0")
  find_cuda_helper_libs(npp)
endif ()
include_directories(${CUDA_INCLUDE_DIRS})
#message("CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
# hide variable as SDK now (usually) part of CUDA Toolkit
# mark_as_advanced(FORCE CUDA_SDK_ROOT_DIR)
# additional directories where to look for SDK which are
# not yet added in FindCUDA.cmake of CMake
list(APPEND INTERNAL_CUDA_SDK_SEARCH_PATH "$ENV{CUDA_SDK_INCLUDE_DIR}") 
list(APPEND INTERNAL_CUDA_SDK_SEARCH_PATH "/usr/local/cuda-5.5-sdk/NVIDIA_CUDA-5.5_Samples/")
list(APPEND INTERNAL_CUDA_SDK_SEARCH_PATH "/usr/local/cuda-6.0-sdk/NVIDIA_CUDA-6.0_Samples/")
list(APPEND INTERNAL_CUDA_SDK_SEARCH_PATH "/opt/cuda-6.0/samples")
if(WIN32)
list(APPEND INTERNAL_CUDA_SDK_SEARCH_PATH "[HKEY_LOCAL_MACHINE\\SOFTWARE\\NVIDIA Corporation\\Installed Products\\NVIDIA SDK 10\\Compute;InstallDir]") 
endif()
list(APPEND INTERNAL_CUDA_SDK_SEARCH_PATH "C:/ProgramData/NVIDIA Corporation/CUDA Samples/v5.5/") 

find_path(CUDA_HELPER_INCLUDE_DIR
  helper_cuda.h
  PATHS          ${INTERNAL_CUDA_SDK_SEARCH_PATH}
  PATH_SUFFIXES "/common/inc"
  DOC           "Location of helper_cuda.h of the CUDA SDK."
  NO_DEFAULT_PATH
)

if (CUDA_HELPER_INCLUDE_DIR)
  include_directories(${CUDA_HELPER_INCLUDE_DIR})
  set(CUDA_SDK_ROOT_DIR ${CUDA_HELPER_INCLUDE_DIR})
 # message("CUDA_SDK_ROOT_DIR: ${CUDA_SDK_ROOT_DIR}")
else(CUDA_HELPER_INCLUDE_DIR)
   message("CUDA_SDK_ROOT_DIR: really not found")	
endif(CUDA_HELPER_INCLUDE_DIR) 
include_directories(${CUDA_SDK_ROOT_DIR}/common/inc)

if( ${CUDA_VERSION} VERSION_LESS "5")
  message(INFO "Building with CUDA < 5.0")
else()
  message(STATUS "Building with CUDA >= 5.0")
  # we build for all useful compute capabilities (C.P > 2.0)
  #set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode;arch=compute_12,code=sm_12;-gencode;arch=compute_20,code=sm_21;-gencode;arch=compute_20,code=sm_20;-gencode;arch=compute_30,code=sm_30;")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode;arch=compute_30,code=sm_30;-gencode;arch=compute_35,code=sm_35;-use_fast_math;")
  #-deviceemu -use_fast_math;
  #untested for CP < 2.0
  #set(CUDA_NVCC_FLAGS "-gencode;arch=compute_12,code=sm_12;--disable-warnings")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../IRTKSimple2/common++/include
	${CMAKE_CURRENT_SOURCE_DIR}/../IRTKSimple2/contrib++/include
	${CMAKE_CURRENT_SOURCE_DIR}/../IRTKSimple2/geometry++/include
	${CMAKE_CURRENT_SOURCE_DIR}/../IRTKSimple2/image++/include
	${CMAKE_CURRENT_SOURCE_DIR}/../IRTKSimple2/packages/transformation/include
	${CMAKE_CURRENT_SOURCE_DIR}/../IRTKSimple2/packages/registration/include
	${CMAKE_CURRENT_SOURCE_DIR}
	)

if(UNIX)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
endif(UNIX)

if(BUILD_WITH_CULA)
find_package(CULA REQUIRED)
include_directories(${CULA_INCLUDE_DIR})
add_definitions(-DHAVE_CULA)
endif(BUILD_WITH_CULA)

#need to check if Unix is sane
# add boost dependencies
if(UNIX)
find_package( Boost 1.46 REQUIRED COMPONENTS program_options filesystem system thread)
add_definitions(-DUSE_BOOST=1)
else(UNIX)
find_package( Boost 1.46 REQUIRED COMPONENTS program_options thread)
add_definitions(-DBOOST_NO_CXX11_ALLOCATOR)
if(MSVC11)
add_definitions(-DUSE_BOOST=0)
else(MSVC11)
add_definitions(-DUSE_BOOST=1)
endif(MSVC11)
endif(UNIX)

if ( NOT Boost_FOUND )
message( STATUS "Boost could not be found." )
   set( BOOST_ROOT ${BOOST_ROOT} CACHE PATH "Please enter path to Boost include folder." FORCE )
endif ()
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/GPUGauss)
cuda_add_library(reconstruction_cuda_lib2 
				reconstruction_cuda2.cu 
				reconstruction_cuda2.cuh 
				recon_volumeHelper.cuh 
				GPUGauss/gaussfilter.cu 
				GPUGauss/gaussFilterConvolution.cuh
				GPUWorker.cpp
				GPUWorker.h)

add_executable(reconstruction_GPU2 reconstruction.cc irtkReconstructionGPU.cc 
          irtkReconstructionGPU.h 
          perfstats.h
          stackMotionEstimator.cpp stackMotionEstimator.h)

target_link_libraries(reconstruction_GPU2 ${IRTK_LIBRARIES} ${TBB_LIBRARIES} ${GSL_LIBRARIES} reconstruction_cuda_lib2)

if(UNIX)
target_link_libraries(reconstruction_GPU2 ${Boost_LIBRARIES})
endif(UNIX)

if(BUILD_WITH_CULA)
target_link_libraries(reconstruction_GPU2 ${CULA_LIBRARIES} )
endif(BUILD_WITH_CULA)
