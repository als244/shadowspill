cmake_minimum_required(VERSION 3.25)
project(ShadowSpill VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

include(CTest)

add_library(shadowspill_build_options INTERFACE)
target_compile_options(
  shadowspill_build_options
  INTERFACE
    $<$<COMPILE_LANG_AND_ID:C,GNU,Clang>:-Wall;-Wextra;-Werror;-Wpedantic>
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall;-Wextra;-Werror;-Wpedantic>
)

add_library(
  shadowspill_simulator
  SHARED
  simulator/src/diagnostics.c
  simulator/src/events.c
  simulator/src/memory.c
  simulator/src/simulator.c
  simulator/src/status.c
  simulator/src/tasks.c
  simulator/src/transfers.c
  simulator/src/validation.c
)
target_include_directories(
  shadowspill_simulator
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/simulator/include>
    $<INSTALL_INTERFACE:shadowspill/include>
  PRIVATE simulator/src
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
  FILES simulator/include/shadowspill/simulator.h
  DESTINATION shadowspill/include/shadowspill
)

if(BUILD_TESTING)
  add_executable(shadowspill_build_canary tests/native/build_canary.c)
  target_link_libraries(shadowspill_build_canary PRIVATE shadowspill_build_options)
  add_test(NAME shadowspill_build_canary COMMAND shadowspill_build_canary)

  add_executable(
    shadowspill_simulator_canary
    tests/simulator/simulator_canary.c
  )
  target_link_libraries(
    shadowspill_simulator_canary
    PRIVATE shadowspill_simulator shadowspill_build_options
  )
  add_test(NAME shadowspill_simulator_canary COMMAND shadowspill_simulator_canary)
endif()
