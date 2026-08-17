add_library(
  shadowspill_simulator
  SHARED
  src/admission.c
  src/diagnostics.c
  src/events.c
  src/memory.c
  src/simulator.c
  src/status.c
  src/tasks.c
  src/transfers.c
  src/validation.c
)
target_include_directories(
  shadowspill_simulator
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:shadowspill/include>
  PRIVATE src
)
target_link_libraries(shadowspill_simulator PRIVATE shadowspill_build_options)
set_target_properties(
  shadowspill_simulator
  PROPERTIES
    OUTPUT_NAME shadowspill_simulator
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
)

install(
  TARGETS shadowspill_simulator
  LIBRARY DESTINATION shadowspill/lib
  RUNTIME DESTINATION shadowspill/lib
)
install(
  FILES include/shadowspill/simulator.h
  DESTINATION shadowspill/include/shadowspill
)
