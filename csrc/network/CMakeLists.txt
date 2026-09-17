# The network extension library and the daemon that holds the far side.
#
# The two have different dependencies on purpose. This library is a TCP client
# and a pool kind: it links libc, because nothing local touches a NIC until
# there is a lane. The daemon registers memory with its own NIC to hand back a
# real key, so it links verbs and is built only where verbs headers are found.
#
# Neither is linked into libshadowspill. This is loaded the way a backend is --
# one layer up, by whatever embeds the runtime -- and the runtime itself loads
# nothing.
add_library(
  shadowspill_network
  SHARED
  src/control.c
  src/library.c
  src/remote_memory.c
)
target_include_directories(
  shadowspill_network
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)
target_link_libraries(shadowspill_network PRIVATE shadowspill_build_options)
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

# The daemon. Verbs-gated: without them it cannot return a genuine key, and a
# daemon returning a made-up one would be worse than no daemon at all.
find_path(SHADOWSPILL_VERBS_INCLUDE_DIR infiniband/verbs.h)
find_library(SHADOWSPILL_VERBS_LIBRARY NAMES ibverbs)
if(SHADOWSPILL_VERBS_INCLUDE_DIR AND SHADOWSPILL_VERBS_LIBRARY)
  add_executable(shadowspill_memory_daemon ../daemon/src/daemon.c)
  target_include_directories(
    shadowspill_memory_daemon PRIVATE ${SHADOWSPILL_VERBS_INCLUDE_DIR}
  )
  target_link_libraries(
    shadowspill_memory_daemon
    PRIVATE ${SHADOWSPILL_VERBS_LIBRARY} shadowspill_build_options
  )
  install(TARGETS shadowspill_memory_daemon RUNTIME DESTINATION shadowspill/bin)
  message(STATUS "ShadowSpill memory daemon: building (ibverbs found)")
else()
  message(
    STATUS
    "ShadowSpill memory daemon: skipped (no ibverbs headers or library)"
  )
endif()
