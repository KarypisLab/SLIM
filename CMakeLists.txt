cmake_minimum_required(VERSION 2.8)
project(SLIM)

set(SHARED FALSE CACHE BOOL "build a shared library")
set(WITH_MKL FALSE CACHE BOOL "build using MKL")

if(MSVC)
  set(SLIM_INSTALL FALSE)
else()
  set(SLIM_INSTALL TRUE)
endif()

# Configure libslim library.
if(SHARED)
  set(SLIM_LIBRARY_TYPE SHARED)
else()
  set(SLIM_LIBRARY_TYPE STATIC)
endif(SHARED)

if(WITH_MKL)
  message(STATUS "*** Building WITH MKL support!")
  set(USE_MKL true)
else()
  message(STATUS "*** Building WITHOUT MKL support!")
  set(USE_MKL false)
endif(WITH_MKL)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(MACOSX TRUE)
endif()

if (OPENMP)
  set(__OPENMP__ TRUE)
endif()

# Add GK's standard cmake settings
include(./conf/gkbuild.cmake)

# Include directories
include_directories(include)
include_directories(lib/GKlib)

# Link directories 
if (MACOSX)
  link_directories(${PROJECT_SOURCE_DIR}/lib/GKlib/build/Darwin-x86_64/)
else()  
  link_directories(${PROJECT_SOURCE_DIR}/lib/GKlib/build/Linux-x86_64/)
endif()


# Recursively look for CMakeLists.txt in subdirs.
add_subdirectory("include")
add_subdirectory("src/libslim")
# add_subdirectory("src/programs")

