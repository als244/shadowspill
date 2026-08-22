if(NOT TARGET shadowspill_backend_cuda)
  return()
endif()

# Imported CMake targets are directory scoped.  The CUDA backend owns its own
# discovery, and this sibling adapter independently declares the driver API it
# includes and links.
find_package(CUDAToolkit REQUIRED)

add_library(
  shadowspill_pytorch
  SHARED
  allocator.c
  allocator_callback.cpp
)
target_include_directories(
  shadowspill_pytorch
  PRIVATE
    include
    ${PROJECT_SOURCE_DIR}/csrc/include
    ${PROJECT_SOURCE_DIR}/csrc/backends/cuda/include
)
target_link_libraries(
  shadowspill_pytorch
  PRIVATE
    shadowspill
    shadowspill_backend_cuda
    CUDA::cuda_driver
    Threads::Threads
    shadowspill_build_options
)
if(UNIX AND NOT APPLE)
  target_link_options(
    shadowspill_pytorch
    PRIVATE "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/shadowspill_pytorch.map"
  )
endif()

find_package(Torch QUIET)
if(Torch_FOUND)
  target_sources(shadowspill_pytorch PRIVATE storage.cpp)
  target_link_libraries(shadowspill_pytorch PRIVATE ${TORCH_LIBRARIES})
  target_compile_definitions(
    shadowspill_pytorch
    PRIVATE SHADOWSPILL_PYTORCH_STORAGE_ADAPTER=1
  )
  if(TORCH_CXX_FLAGS)
    separate_arguments(
      SHADOWSPILL_TORCH_CXX_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}"
    )
    target_compile_options(
      shadowspill_pytorch PRIVATE ${SHADOWSPILL_TORCH_CXX_FLAGS}
    )
  endif()
else()
  message(
    STATUS
    "PyTorch CMake package not found; building allocator connector without storage rebinding"
  )
endif()

set_target_properties(
  shadowspill_pytorch
  PROPERTIES
    OUTPUT_NAME shadowspill_pytorch
    INSTALL_RPATH "$ORIGIN"
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)
install(
  TARGETS shadowspill_pytorch
  LIBRARY DESTINATION shadowspill/lib
  RUNTIME DESTINATION shadowspill/lib
)
