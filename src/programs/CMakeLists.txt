# These programs use internal slim data structures.
include_directories(../../include)

# Build program.
add_executable(slim_learn slim_learn.c cmdline_learn.c)
add_executable(slim_predict slim_predict.c cmdline_predict.c)
add_executable(slim_mselect slim_mselect.c cmdline_mselect.c)
foreach(prog slim_learn slim_predict slim_mselect)
  target_link_libraries(${prog} slim)
endforeach(prog)

if(SLIM_INSTALL)
  install(TARGETS slim_learn RUNTIME DESTINATION bin)
  install(TARGETS slim_predict RUNTIME DESTINATION bin)
  install(TARGETS slim_mselect RUNTIME DESTINATION bin)
endif()

