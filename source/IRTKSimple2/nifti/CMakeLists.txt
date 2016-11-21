#PROJECT(NIFTI)

if(WIN32)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/zlib) #for up-to-date zlib.h
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/zlib)
set(ZLIB_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/zlib)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/zlib)
set(ZLIB_LIBRARY zlib)
else(WIN32)
FIND_PACKAGE(ZLIB REQUIRED)
include_directories(${ZLIB_INCLUDE_DIRS})
set(ZLIB_LIBRARY ${ZLIB_LIBRARIES})
set(ZLIB_INCLUDE_DIR ${ZLIB_INCLUDE_DIRS})
endif(WIN32)

ADD_DEFINITIONS(-DHAVE_ZLIB)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/znzlib)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/niftilib)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/znzlib)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/niftilib)

#SUBDIRS(znzlib niftilib)
#FIND_PACKAGE(ZLIB)
