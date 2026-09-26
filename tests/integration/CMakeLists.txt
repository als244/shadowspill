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
      # A cache root of this test's own. ShadowSpill stores compiled task
      # manifests under `cache_dir()`, which is **Inductor's** directory --
      # process-global, shared by every concurrent test and persistent across
      # runs, and nothing to do with the per-test `artifact_store`. Its key is
      # the FX graph cache key and the semantic contract digest, both fixed
      # before Inductor lowers, so two compiles that differ only in a decision
      # Inductor makes *during* lowering collide on one entry.
      #
      # `isolated_cache.py` wraps every command here: it makes a fresh cache
      # root, points Inductor and Triton at it, and deletes it afterwards, so
      # whatever a test sees it put there itself during this run.
      ENVIRONMENT
        "PYTHONPATH=${PROJECT_SOURCE_DIR}/src:${PROJECT_SOURCE_DIR};SHADOWSPILL_CANARY_TIMEOUT=${timeout};SHADOWSPILL_LIBRARY_DIRECTORY=$<TARGET_FILE_DIR:shadowspill>"
      TIMEOUT ${timeout}
  )
endfunction()

shadowspill_add_pytorch_canary(allocator 60)
shadowspill_add_pytorch_canary(runtime_close 60)
shadowspill_add_pytorch_canary(overlap 60)
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
shadowspill_add_pytorch_canary(step_archive 120)

# Installing the allocator needs a process where CUDA has not been
# initialized, so exactly one of these can run per pytest process; in the
# ordinary suite the first one takes that slot and the rest skip. Running each
# in its own process is the only way they all execute.
function(shadowspill_add_public_api_test file name timeout)
  add_test(
    NAME shadowspill_pytorch_public_${name}
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/isolated_cache.py
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
      # A cache root of this test's own. ShadowSpill stores compiled task
      # manifests under `cache_dir()`, which is **Inductor's** directory --
      # process-global, shared by every concurrent test and persistent across
      # runs, and nothing to do with the per-test `artifact_store`. Its key is
      # the FX graph cache key and the semantic contract digest, both fixed
      # before Inductor lowers, so two compiles that differ only in a decision
      # Inductor makes *during* lowering collide on one entry.
      #
      # `isolated_cache.py` wraps every command here: it makes a fresh cache
      # root, points Inductor and Triton at it, and deletes it afterwards, so
      # whatever a test sees it put there itself during this run.
      ENVIRONMENT "PYTHONPATH=${PROJECT_SOURCE_DIR}/src:${PROJECT_SOURCE_DIR};SHADOWSPILL_LIBRARY_DIRECTORY=$<TARGET_FILE_DIR:shadowspill>"
      TIMEOUT ${timeout}
  )
endfunction()

shadowspill_add_public_api_test(
  test_00_public_forward.py test_public_forward_executes_reloads_and_restores 300
)
shadowspill_add_public_api_test(
  test_01_public_training.py test_public_training_accumulates_replays_and_restores 300
)
shadowspill_add_public_api_test(
  test_01_public_training.py test_public_training_breadth_first_matches_the_eager_reference 300
)
shadowspill_add_public_api_test(
  test_01_public_training.py test_public_training_declared_adamw_state_replays 300
)
shadowspill_add_public_api_test(
  test_01_public_training.py test_public_training_fills_declared_state_in_the_spill_pool 300
)
shadowspill_add_public_api_test(
  test_01_public_training.py test_public_training_refuses_state_an_initialiser_replaced 300
)
shadowspill_add_public_api_test(
  test_01_public_training.py test_public_training_owns_the_model_state_it_imported 300
)
shadowspill_add_public_api_test(
  test_01_public_training.py test_public_training_saves_straight_from_the_pool 300
)
shadowspill_add_public_api_test(
  test_01_public_training.py test_public_training_saves_a_master_copy_once 300
)
shadowspill_add_public_api_test(
  test_01_public_training.py test_public_training_profiles_bounded_opaque_optimizer 300
)
shadowspill_add_public_api_test(
  test_01_public_training.py test_public_training_partitions_device_only_optimizer_and_replays 300
)
shadowspill_add_public_api_test(
  test_01_public_training.py test_public_training_follows_a_learning_rate_schedule 300
)
shadowspill_add_public_api_test(
  test_02_shared_model.py test_a_forward_planned_after_training_sees_every_update 300
)
shadowspill_add_public_api_test(
  test_02_shared_model.py test_a_training_step_planned_after_a_forward_shares_its_model 300
)
shadowspill_add_public_api_test(
  test_02_shared_model.py test_state_a_plan_imported_for_itself_is_not_shared 300
)
shadowspill_add_public_api_test(
  test_03_planning_host_memory.py test_a_plan_keeps_no_host_copy_of_what_its_forwards_saved 300
)
shadowspill_add_public_api_test(
  test_03_planning_host_memory.py test_planning_fails_when_the_spill_pool_has_no_room_for_saved_values 300
)
shadowspill_add_public_api_test(
  test_04_plans_side_by_side.py test_a_forward_is_planned_beside_a_step_holding_more_than_it_reserves 300
)
