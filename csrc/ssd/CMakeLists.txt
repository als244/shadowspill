if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(STATUS "ShadowSpill SSD extension: requires Linux direct I/O")
  return()
endif()

add_library(shadowspill_ssd SHARED pool.c lane.c lane_io.c library.c)
target_link_libraries(
  shadowspill_ssd PRIVATE shadowspill Threads::Threads shadowspill_build_options
)
set_target_properties(
  shadowspill_ssd PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION 0
)
install(TARGETS shadowspill_ssd LIBRARY DESTINATION shadowspill/lib)
