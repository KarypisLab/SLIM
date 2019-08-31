# Add this directory for internal users.
include_directories(.)

# Find sources.
file(GLOB sources *.c)

# Build the library 
add_library(slim ${SLIM_LIBRARY_TYPE} ${sources})
if(APPLE)
  target_link_libraries(slim GKlib m)
elseif(UNIX)
  target_link_libraries(slim GKlib m)
endif()

if(USE_MKL)
  target_link_libraries(slim GKlib m mkl_rt)
  add_definitions(-DUSE_MKL=1)
endif()


if(SLIM_INSTALL)
  install(TARGETS slim
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib
    ARCHIVE DESTINATION lib)
endif()
