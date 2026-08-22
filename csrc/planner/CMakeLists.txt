add_library(
  shadowspill_planner
  SHARED
  src/admission.c
  src/context.c
  src/digest.c
  src/placement.c
  src/portfolio.c
  src/residency.c
  src/schedule.c
  src/selector.c
  src/status.c
)
target_include_directories(
  shadowspill_planner
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/csrc/simulator/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/csrc/runtime/include>
    $<INSTALL_INTERFACE:shadowspill/include>
  PRIVATE src
)
# Candidate admission reuses the production MemoryPool policy. Runtime does not
# depend on planner, so the compiled dependency graph remains acyclic.
target_link_libraries(
  shadowspill_planner
  PRIVATE
    shadowspill_runtime
    shadowspill_simulator
    shadowspill_build_options
)
set_target_properties(
  shadowspill_planner
  PROPERTIES
    OUTPUT_NAME shadowspill_planner
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
    INSTALL_RPATH "$ORIGIN"
)

install(
  TARGETS shadowspill_planner
  LIBRARY DESTINATION shadowspill/lib
  RUNTIME DESTINATION shadowspill/lib
)
install(
  FILES include/shadowspill/planner.h
  DESTINATION shadowspill/include/shadowspill
)
