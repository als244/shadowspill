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
