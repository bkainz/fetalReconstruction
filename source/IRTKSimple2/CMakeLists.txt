cmake_minimum_required(VERSION 2.6)
if(COMMAND cmake_policy)
  # policy settings ...
  cmake_policy(SET CMP0003 NEW)
  if( POLICY CMP0038)
    cmake_policy(SET CMP0038 OLD)
  endif()
endif(COMMAND cmake_policy)

#PROJECT(IRTK)

#INCLUDE(CTest)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common++/include
	${CMAKE_CURRENT_SOURCE_DIR}/contrib++/include
	${CMAKE_CURRENT_SOURCE_DIR}/geometry++/include
	${CMAKE_CURRENT_SOURCE_DIR}/image++/include
	${CMAKE_CURRENT_SOURCE_DIR}/packages/transformation/include
	${CMAKE_CURRENT_SOURCE_DIR}/packages/registration/include
	${CMAKE_CURRENT_SOURCE_DIR}/nifti/niftilib
	${CMAKE_CURRENT_SOURCE_DIR}/nifti/zlib
	${CMAKE_CURRENT_SOURCE_DIR}/nifti/znzlib
	)

IF (BUILD_WITH_NIFTI)
   add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/nifti)
   include_directories(${CMAKE_CURRENT_BINARY_DIR}/nifti/zlib) #for up-to-date zlib.h
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/nifti/zlib)
ENDIF (BUILD_WITH_NIFTI)

# Finding GNU scientific library GSL
FIND_PACKAGE(GSL REQUIRED)
IF (GSL_FOUND)
  INCLUDE_DIRECTORIES(${GSL_INCLUDE_DIR})
  LINK_DIRECTORIES(${GSL_LINK_DIRECTORIES})
  LINK_LIBRARIES(${GSL_LIBRARIES})
ENDIF (GSL_FOUND)

# add boost dependencies
find_package( Boost 1.46 REQUIRED)

if ( NOT Boost_FOUND )
message( STATUS "Boost could not be found." )
   set( BOOST_ROOT ${BOOST_ROOT} CACHE PATH "Please enter path to Boost include folder." FORCE )
endif ()

#message( STATUS "Boost_INCLUDE_DIRS : '" ${Boost_INCLUDE_DIRS} "'" )

include_directories(${Boost_INCLUDE_DIRS})

OPTION(USE_VXL "Use VXL library instead of Numerical Recipes" OFF)
IF (USE_VXL)
  SUBDIRS(vxl)
  #  INCLUDE_DIRECTORIES(${VCL_INCLUDE_DIR} ${VXLCORE_INCLUDE_DIR})
  INCLUDE_DIRECTORIES(${IRTK_SOURCE_DIR}/vxl/core ${IRTK_SOURCE_DIR}/vxl/vcl)
  ADD_DEFINITIONS(-DUSE_VXL)
ELSE (USE_VXL)

ENDIF (USE_VXL)

OPTION(BUILD_WITH_TBB "Use TBB library for multithreading support" ON)

#SUBDIRS(common++/src geometry++/src image++/src contrib++/src applications packages wrapping)
SUBDIRS(common++/src geometry++/src image++/src contrib++/src packages)

INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/config/IRTKInclude.cmake)

SET(EXECUTABLE_OUTPUT_PATH ${IRTK_BINARY_DIR}/bin CACHE STRING
  "Where to put the executables")
SET(LIBRARY_OUTPUT_PATH ${IRTK_BINARY_DIR}/lib CACHE STRING
  "Where to put the libraries")

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/config/IRTKInclude.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/lib/IRTKInclude.cmake)	

#CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/InstallerScripts/irtkInstaller.sh
#               ${CMAKE_CURRENT_BINARY_DIR}/bin/irtkInstaller.sh)
               
#EXEC_PROGRAM(chmod ${CMAKE_CURRENT_BINARY_DIR}/bin ARGS +x;irtkInstaller.sh)

INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/config/IRTKGenerateConfigFile.cmake)
#INCLUDE(${CMAKE_CURRENT_BINARY_DIR}/IRTKConfig.cmake)

set(IRTK_LIBRARIES common++ contrib++ geometry++ image++ registration++ transformation++ ${ZLIB_LIBRARY} znz niftiio)