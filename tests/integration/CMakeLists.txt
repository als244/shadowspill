if(NOT TARGET shadowspill_pytorch)
  return()
endif()

find_package(Python3 COMPONENTS Interpreter REQUIRED)

function(shadowspill_add_pytorch_canary name timeout)
  add_test(
    NAME shadowspill_pytorch_${name}_canary
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/pytorch/${name}_canary.py
      $<TARGET_FILE:shadowspill_pytorch>
  )
  set_tests_properties(
    shadowspill_pytorch_${name}_canary
    PROPERTIES
      LABELS "cuda;pytorch"
      ENVIRONMENT "PYTHONPATH=${PROJECT_SOURCE_DIR}/src:${PROJECT_SOURCE_DIR}"
      TIMEOUT ${timeout}
  )
endfunction()

shadowspill_add_pytorch_canary(allocator 60)
shadowspill_add_pytorch_canary(runtime_close 60)
shadowspill_add_pytorch_canary(overlap 20)
shadowspill_add_pytorch_canary(physical_failure 60)
shadowspill_add_pytorch_canary(allocator_oom 60)
shadowspill_add_pytorch_canary(allocation_contract_failure 20)
shadowspill_add_pytorch_canary(invalid_access 60)
shadowspill_add_pytorch_canary(planning_failure 120)
shadowspill_add_pytorch_canary(profiler 60)
shadowspill_add_pytorch_canary(forward 90)
shadowspill_add_pytorch_canary(state_import_export 90)
shadowspill_add_pytorch_canary(mutation 90)
shadowspill_add_pytorch_canary(training 120)

# Installing the allocator needs a process where CUDA has not been
# initialized, so exactly one of these can run per pytest process; in the
# ordinary suite the first one takes that slot and the rest skip. Running each
# in its own process is the only way they all execute.
function(shadowspill_add_public_api_test file name timeout)
  add_test(
    NAME shadowspill_pytorch_public_${name}
    COMMAND
      ${Python3_EXECUTABLE} -m pytest
      ${PROJECT_SOURCE_DIR}/tests/shadowspill/pytorch/api/${file}
      -k ${name}
      -m fresh_process
      --no-header
      -q
  )
  set_tests_properties(
    shadowspill_pytorch_public_${name}
    PROPERTIES
      LABELS "cuda;pytorch"
      ENVIRONMENT "PYTHONPATH=${PROJECT_SOURCE_DIR}/src:${PROJECT_SOURCE_DIR}"
      TIMEOUT ${timeout}
      # Each of these plans and runs a real model, so two at once contend for
      # the device and one of them loses.
      RESOURCE_LOCK shadowspill_device
  )
endfunction()

shadowspill_add_public_api_test(
  test_00_public_forward.py test_public_forward_executes_reloads_and_restores 300
)
shadowspill_add_public_api_test(
  test_01_public_training.py test_public_training_accumulates_replays_and_restores 300
)
shadowspill_add_public_api_test(
  test_01_public_training.py test_public_training_lazy_adamw_state_replays 300
)
shadowspill_add_public_api_test(
  test_01_public_training.py test_public_training_profiles_bounded_opaque_optimizer 300
)
shadowspill_add_public_api_test(
  test_01_public_training.py test_public_training_partitions_cuda_only_optimizer_and_replays 300
)
