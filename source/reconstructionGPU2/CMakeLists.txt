if(CMAKE_COMPILER_IS_GNUCC)
    add_definitions(-Wall -Wextra -pedantic)
    add_definitions(-Wcast-align)
    add_definitions(-Wcast-qual)
    add_definitions(-Wdisabled-optimization)
    add_definitions(-Wstrict-aliasing)
    add_definitions(-Wunknown-pragmas)
endif(CMAKE_COMPILER_IS_GNUCC)

set(CONFIG_SUMMARY ${CMAKE_CURRENT_BINARY_DIR}/config.summary)
file(REMOVE ${CONFIG_SUMMARY})
get_filename_component(PROJECT_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/.. REALPATH)
set(CTEST_SOURCE_DIRECTORY ${PROJECT_SOURCE_DIR})
set(CTEST_BINARY_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
message(STATUS "CTEST_BINARY_DIRECTORY = ${CTEST_BINARY_DIRECTORY}.")
include(${PROJECT_SOURCE_DIR}/cmake/SciFuncsMacros.cmake)
find_package(SciCuda REQUIRED COMPONENTS sdk npp thrust)
include_directories(${CUDA_HELPER_INCLUDE_DIR})
include_directories(${CUDA_SDK_ROOT_DIR}/common/inc)
set(CUDA_SEPARABLE_COMPILATION ON)

list(APPEND CUDA_NVCC_FLAGS -keep --shared --compiler-options -fPIC -lineinfo )

if(NOT CUDA_SDK_ROOT_DIR)
message("I tried but I couldn't find the CUDA SDK. please set CUDA_SDK_ROOT_DIR manually to the samples directory of the CUDA toolkit!")
endif()

if(UNIX)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
endif(UNIX)


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../IRTKSimple2/common++/include
	${CMAKE_CURRENT_SOURCE_DIR}/../IRTKSimple2/contrib++/include
	${CMAKE_CURRENT_SOURCE_DIR}/../IRTKSimple2/geometry++/include
	${CMAKE_CURRENT_SOURCE_DIR}/../IRTKSimple2/image++/include
	${CMAKE_CURRENT_SOURCE_DIR}/../IRTKSimple2/packages/transformation/include
	${CMAKE_CURRENT_SOURCE_DIR}/../IRTKSimple2/packages/registration/include
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/include
	${CMAKE_CURRENT_SOURCE_DIR}/include/GPUGauss
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
find_package( Boost 1.58 REQUIRED COMPONENTS program_options filesystem system thread)
add_definitions(-DUSE_BOOST=1)
else(UNIX)
find_package( Boost 1.58 REQUIRED COMPONENTS program_options thread)
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

if( (Boost_VERSION GREATER 105999) AND
     (NOT CUDA_VERSION VERSION_EQUAL 7.5) AND
     (NOT CMAKE_CXX_STANDARD EQUAL 98) )
      # Boost Bug https://svn.boost.org/trac/boost/ticket/11897
	# message(STATUS "Boost: Disable variadic templates")
	# add_definitions(-DBOOST_NO_CXX11_VARIADIC_TEMPLATES)
	# message(STATUS "Boost: Disable template aliases")
    add_definitions(-DBOOST_NO_CXX11_TEMPLATE_ALIASES)
  endif()

#set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
#set(BUILD_SHARED_LIBS OFF)
set(CUDA_SEPARABLE_COMPILATION ON)

cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/GPUGauss ${CMAKE_CURRENT_SOURCE_DIR})

SET(RECON_MAIN_HDRS
	irtkReconstructionGPU.h
	perfstats.h
	stackMotionEstimator.h
	)

SET(RECON_MAIN_SRCS reconstruction.cc 
		irtkReconstructionGPU.cc 
        stackMotionEstimator.cpp )

SET(RECON_LIB_SRCS
	include/reconstruction_cuda2.cuh 
	include/recon_volumeHelper.cuh 
	include/GPUGauss/gaussFilterConvolution.cuh
	include/GPUWorker.h
)
SET(RECON_LIB_SRCS reconstruction_cuda2.cu 
				GPUGauss/gaussfilter.cu 
				GPUWorker.cpp
				)

#message(${CUDA_NVCC_FLAGS})
cuda_add_library(SVRreconstructionGPU_lib ${RECON_LIB_SRCS} ${RECON_LIB_HDRS})
cuda_add_executable(SVRreconstructionGPU ${RECON_MAIN_SRCS} ${RECON_MAIN_HDRSS})
target_link_libraries(SVRreconstructionGPU ${IRTK_LIBRARIES} ${TBB_LIBRARIES} ${GSL_LIBRARIES} ${CUDA_CUDADEVRT_LIBRARY} 
	${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} SVRreconstructionGPU_lib)

if(BUILD_WITH_CULA)
target_link_libraries(SVRreconstructionGPU ${CULA_LIBRARIES} )
endif(BUILD_WITH_CULA)

SET(PATCHRECON_MAIN_HDRS
	include/irtkPatchBasedReconstruction.h 
	include/nDRegistration.h
	include/irtkStack3D3DRegistration.h
	include/patchBased2D3DRegistration.h
	include/patchBasedPackageSplitter.h
	include/runStackSLIC.h
)

SET(PATCHRECON_MAIN_SRCS patchBasedReconMain.cpp 
		irtkPatchBasedReconstruction.cpp 	
		irtkStack3D3DRegistration.cpp	
		patchBased2D3DRegistration.cpp	
		patchBasedPackageSplitter.cpp	
		runStackSLIC.cpp)

SET(PATCHRECON_LIB_HDRS 
		include/patchBasedVolume.cuh 
		include/patchBasedObject.cuh 
		include/reconVolume.cuh
		include/matrix4.cuh
		include/volume.cuh
		include/reconConfig.cuh
		include/ImagePatch2D.cuh
		include/pointSpreadFunction.cuh
		include/patchBasedSuperresolution_gpu.cuh
		include/patchBasedRobustStatistics_gpu.cuh
		include/patchBased2D3DRegistration_gpu2.cuh
		include/interpFunctions.cuh
		include/GPUGauss/patchBasedGaussFilterConvolution.cuh
)

SET(PATCHRECON_LIB_SRCS 
		initPatchBasedRecon_gpu.cu
		patchBasedPSFReconstruction_gpu.cu
		volume.cu
		reconVolume.cu
		GPUGauss/patchBasedGaussfilter.cu
		patchBasedSimulatePatches_gpu.cu
		patchBasedSuperresolution_gpu.cu
		patchBasedRobustStatistics_gpu.cu
		patchBased2D3DRegistration_gpu2.cu
		runStackSLIC.cpp)

set(PVR_TOOL TRUE)
if(PVR_TOOL)
  cuda_add_library(patchBasedReconlib  ${PATCHRECON_LIB_SRCS} ${PATCHRECON_LIB_HDRS})
  cuda_add_executable(PVRreconstructionGPU ${PATCHRECON_MAIN_SRCS} ${PATCHRECON_MAIN_HDRS})
  target_link_libraries(PVRreconstructionGPU ${IRTK_LIBRARIES} ${TBB_LIBRARIES} 
	${GSL_LIBRARIES} ${CUDA_CUDADEVRT_LIBRARY} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} patchBasedReconlib)
endif(PVR_TOOL)

if(UNIX)
target_link_libraries(SVRreconstructionGPU ${Boost_LIBRARIES})
if(PVR_TOOL)
target_link_libraries(PVRreconstructionGPU ${Boost_LIBRARIES})
endif(PVR_TOOL)
endif(UNIX)


SET(EVALUATION_FUNCTIONS OFF CACHE BOOL "Turn on evaluation functions (for research)")
add_definitions( -DEVALUATE=${EVALUATION_FUNCTIONS} )
