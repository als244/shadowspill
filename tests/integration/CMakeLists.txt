if(NOT TARGET shadowspill_pytorch)
  return()
endif()

find_package(Python3 COMPONENTS Interpreter REQUIRED)

function(shadowspill_add_pytorch_canary name timeout)
  add_test(
    NAME shadowspill_pytorch_${name}_canary
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/isolated_cache.py
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/pytorch/${name}_canary.py
      $<TARGET_FILE:shadowspill_pytorch>
  )
  set_tests_properties(
    shadowspill_pytorch_${name}_canary
    PROPERTIES
      LABELS "cuda;pytorch"
      # Three things this environment carries.
      #
      # SHADOWSPILL_CANARY_TIMEOUT arms the canary's watchdog from the same
      # number CTest kills on, so a wedged canary prints every thread's stack
      # instead of dying silently.
      #
      # SHADOWSPILL_LIBRARY_DIRECTORY says which build these run against.
      # Without it Python resolves libshadowspill.so by its own precedence and
      # can find a *different* build than the adapter is linked against; the
      # loader then maps both, and the process has two runtimes whose
      # structures are not interchangeable. Naming this build's directory makes
      # a ctest run self-contained, and `_refuse_a_second_runtime` catches
      # whatever still gets past it.
      #
      # PYTHONPATH is the source tree, so a canary runs what is checked out.
      # The test wrapper isolates compiler caches outside saved artifacts.
      ENVIRONMENT
        "PYTHONPATH=${PROJECT_SOURCE_DIR}/src:${PROJECT_SOURCE_DIR};SHADOWSPILL_CANARY_TIMEOUT=${timeout};SHADOWSPILL_LIBRARY_DIRECTORY=$<TARGET_FILE_DIR:shadowspill>"
      TIMEOUT ${timeout}
  )
endfunction()

shadowspill_add_pytorch_canary(allocator 60)
shadowspill_add_pytorch_canary(runtime_close 60)
shadowspill_add_pytorch_canary(overlap 60)
shadowspill_add_pytorch_canary(physical_failure 60)
shadowspill_add_pytorch_canary(physical_report_only 60)
shadowspill_add_pytorch_canary(pool_sizing_refusal 60)
shadowspill_add_pytorch_canary(allocator_oom 60)
shadowspill_add_pytorch_canary(allocation_contract_failure 20)
shadowspill_add_pytorch_canary(invalid_access 60)

# Saved-phase tests launch their own fresh execution workers and run in the
# shared suite. The cases named here assert what the allocator of their own
# process looks like after a failure or a measurement, which only a process no
# other test has touched can show. Each carries the fresh_process marker, so
# the shared suite deselects it and it runs once: here, in its own process,
# with its own timeout and its own line in the output.
function(shadowspill_add_public_api_test file name timeout)
  add_test(
    NAME shadowspill_pytorch_public_${name}
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/isolated_cache.py
      ${Python3_EXECUTABLE} -m pytest
      ${PROJECT_SOURCE_DIR}/tests/${file}
      -k ${name}
      -o addopts=
      --no-header
      -q
  )
  set_tests_properties(
    shadowspill_pytorch_public_${name}
    PROPERTIES
      LABELS "cuda;pytorch"
      # Public tests use temporary artifact stores; the wrapper also isolates
      # direct compiler calls and PyTorch reference work outside planning.
      ENVIRONMENT "PYTHONPATH=${PROJECT_SOURCE_DIR}/src:${PROJECT_SOURCE_DIR};SHADOWSPILL_LIBRARY_DIRECTORY=$<TARGET_FILE_DIR:shadowspill>"
      TIMEOUT ${timeout}
  )
endfunction()

shadowspill_add_public_api_test(
  shadowspill/build/test_failures.py
  test_phase_failures_leave_preparation_reusable 300
)
shadowspill_add_public_api_test(
  shadowspill/profiling/test_measure_program.py
  test_saved_training_program_measurement_without_runtime_pools 300
)
shadowspill_add_public_api_test(
  shadowspill/profiling/test_pair_lifetimes.py
  test_pair_inputs_are_released_between_measurements_and_on_failure 300
)
