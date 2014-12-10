project(FetalReconstruction)

cmake_minimum_required(VERSION 2.8)
cmake_policy(SET CMP0010 OLD)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH  ${CMAKE_SOURCE_DIR}/lib)

option(USE_SYSTEM_IRTK "use system IRTK version instead of simplified included version" OFF)
option(BUILD_WITH_CULA "build with CULA support, necessary for automatic motion measurement" OFF)

# Finding GNU scientific library GSL
FIND_PACKAGE(GSL REQUIRED)
set(GSL_LIBRARIES ${GSL_LIBRARIES} ${GSL_LIB})
#IF (GSL_FOUND)
  INCLUDE_DIRECTORIES(${GSL_INCLUDE_DIR})
  LINK_DIRECTORIES(${GSL_LINK_DIRECTORIES})
  LINK_LIBRARIES(${GSL_LIBRARIES})
#ENDIF (GSL_FOUND)

SET(BUILD_OUT_OF_IRTK_TREE TRUE)
if(USE_SYSTEM_IRTK)
#if IRTK_SOURCE_DIR is set, we are building within the IRTK source tree
  if(IRTK_SOURCE_DIR)
    SET(BUILD_OUT_OF_IRTK_TREE FALSE)
  else(IRTK_SOURCE_DIR)
    SET(BUILD_OUT_OF_IRTK_TREE TRUE)
  endif(IRTK_SOURCE_DIR)
INCLUDE_DIRECTORIES(${IRTK_DIR}/../packages/registration/include)
endif(USE_SYSTEM_IRTK)

if (WIN32)
  # for msvc 8 deprecated warning removal
  add_definitions(-D_CRT_SECURE_NO_DEPRECATE)
  add_definitions(-D_CRT_NONSTDC_NO_DEPRECATE)
  add_definitions(-D_CRT_SECURE_NO_DEPRECATE_GLOBALS)
  add_definitions(-DWIN32 -D_WIN32)
  add_definitions(-DNOMINMAX)

  #set(CMAKE_DEBUG_POSTFIX "D")

  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT.lib;libcmtd.lib")
endif(WIN32)

if (WIN32)
  add_definitions(-DGL_GLEXT_PROTOTYPES -bigobj)
endif()

if(BUILD_OUT_OF_IRTK_TREE AND USE_SYSTEM_IRTK)
# we do not need to search for IRTK for in source tree build
find_package(IRTK REQUIRED)
#message("${IRTK_LIBRARIES_DIR}")
include_directories(${IRTK_INCLUDE_DIRS})
link_directories(${IRTK_LIBRARIES_DIR})
endif(BUILD_OUT_OF_IRTK_TREE AND USE_SYSTEM_IRTK)


FIND_PACKAGE(TBB REQUIRED)
IF (TBB_FOUND)
    # Attention: DO NOT define TBB_DEPRECATED by default or before including the
    #            other TBB header files, in particular parallel_for. The deprecated
    #            behavior of parallel_for is to not choose the chunk size (grainsize)
    #            automatically!
    #
    # http://software.intel.com/sites/products/documentation/doclib/tbb_sa/help/tbb_userguide/Automatic_Chunking.htm
    ADD_DEFINITIONS(-DHAS_TBB)
    INCLUDE_DIRECTORIES(${TBB_INCLUDE_DIRS})
    LINK_DIRECTORIES(${TBB_LIBRARY_DIRS})
    LINK_LIBRARIES(${TBB_LIBRARIES})
ENDIF (TBB_FOUND)
#message("ignore the support TBB error. it works. Do not ignore the not found error!")

if(USE_SYSTEM_IRTK)
FIND_PACKAGE(OpenCV REQUIRED)
include_directories(${OpenCV_DIR}/include ${OpenCV_DIR}/include/opencv)

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})
else(USE_SYSTEM_IRTK)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/IRTKSimple2/)
set(IRTK_LIBRARIES common++ contrib++ geometry++ image++ registration++ transformation++ zlib)
endif(USE_SYSTEM_IRTK)

INCLUDE_DIRECTORIES(${TBB_INSTALL_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/reconstructionGPU2/)

## patch based preparations 
#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/patch-based/)



