# The network extension library and the daemon that holds the far side.
#
# **Both are verbs-gated from Phase 4 on.** Until there was a lane, this
# library was a TCP client and linked libc alone, while only the daemon needed
# verbs; now the local side opens a queue pair too, so neither builds without
# them. `libshadowspill` still links libc and nothing else, and still loads
# nothing: this is loaded one layer up, the way a backend is.
#
# A build without verbs produces neither, and a remote pool then fails where it
# is configured -- which is the right place, since the absence is a property of
# the machine rather than of the plan.
find_path(SHADOWSPILL_VERBS_INCLUDE_DIR infiniband/verbs.h)
find_library(SHADOWSPILL_VERBS_LIBRARY NAMES ibverbs)

if(NOT (SHADOWSPILL_VERBS_INCLUDE_DIR AND SHADOWSPILL_VERBS_LIBRARY))
  message(
    STATUS
    "ShadowSpill network extension: skipped (no ibverbs headers or library)"
  )
  return()
endif()
message(STATUS "ShadowSpill network extension: building (ibverbs found)")

add_library(
  shadowspill_network
  SHARED
  src/control.c
  src/endpoint_verbs.c
  src/library.c
  src/remote_lane.c
  src/remote_lane_staging.c
  src/remote_lane_timeline.c
  src/remote_memory.c
  src/tuning.c
)
target_include_directories(
  shadowspill_network
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    ${SHADOWSPILL_VERBS_INCLUDE_DIR}
)
# Whether this verbs library can register a dma-buf, which is how device
# memory reaches the NIC on the open NVIDIA kernel modules. Older libraries
# cannot, and the lane then registers plainly or stages; nothing else differs.
include(CheckSymbolExists)
set(CMAKE_REQUIRED_INCLUDES ${SHADOWSPILL_VERBS_INCLUDE_DIR})
set(CMAKE_REQUIRED_LIBRARIES ${SHADOWSPILL_VERBS_LIBRARY})
check_symbol_exists(
  ibv_reg_dmabuf_mr "infiniband/verbs.h" SHADOWSPILL_HAVE_DMABUF_MR
)
unset(CMAKE_REQUIRED_INCLUDES)
unset(CMAKE_REQUIRED_LIBRARIES)
if(SHADOWSPILL_HAVE_DMABUF_MR)
  target_compile_definitions(
    shadowspill_network PRIVATE SHADOWSPILL_HAVE_DMABUF_MR=1
  )
  message(STATUS "ShadowSpill network extension: dma-buf registration available")
endif()
# Links the runtime as well as verbs, for one call: the way a lane reports a
# failure it found on a thread of its own.
#
# The dependency runs one way only -- `libshadowspill` neither links nor loads
# this, which is the property that matters. Leaving it implicit did not work:
# the adapter opens this with RTLD_LOCAL, so the runtime's symbols are not in
# the namespace it resolves against, and the load failed with an undefined
# symbol that every other path happened to satisfy.
target_link_libraries(
  shadowspill_network
  PRIVATE
    shadowspill
    ${SHADOWSPILL_VERBS_LIBRARY}
    shadowspill_build_options
)
set_target_properties(
  shadowspill_network
  PROPERTIES
    OUTPUT_NAME shadowspill_network
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
)
install(
  TARGETS shadowspill_network
  LIBRARY DESTINATION shadowspill/lib
  RUNTIME DESTINATION shadowspill/lib
)

# The daemon: the far side, linking verbs and libc and nothing of
# ShadowSpill's. That is what lets it be copied to a peer as one source file
# and built there with no repo, no CMake and no toolchain match.
add_executable(shadowspill_memory_daemon ../daemon/src/daemon.c)
target_include_directories(
  shadowspill_memory_daemon PRIVATE ${SHADOWSPILL_VERBS_INCLUDE_DIR}
)
target_link_libraries(
  shadowspill_memory_daemon
  PRIVATE ${SHADOWSPILL_VERBS_LIBRARY} shadowspill_build_options
)
install(TARGETS shadowspill_memory_daemon RUNTIME DESTINATION shadowspill/bin)
