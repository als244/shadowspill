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
      # Isolate compiler calls outside planning. Public planning calls also
      # supply a temporary artifact store, which owns their compiler caches.
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
      # Public tests use temporary artifact stores; the wrapper also isolates
      # direct compiler calls and PyTorch reference work outside planning.
      ENVIRONMENT "PYTHONPATH=${PROJECT_SOURCE_DIR}/src:${PROJECT_SOURCE_DIR};SHADOWSPILL_LIBRARY_DIRECTORY=$<TARGET_FILE_DIR:shadowspill>"
      TIMEOUT ${timeout}
  )
endfunction()

shadowspill_add_public_api_test(
  test_00_public_forward.py test_public_forward_executes_reloads_and_restores 300
)
shadowspill_add_public_api_test(
  test_00_public_forward.py test_public_forward_sets_a_model_buffer_each_call 300
)
shadowspill_add_public_api_test(
  test_01_public_training.py test_public_training_accumulates_replays_and_restores 300
)
shadowspill_add_public_api_test(
  test_01_public_training.py test_public_training_breadth_first_matches_the_eager_reference 300
)
shadowspill_add_public_api_test(
  test_parameter_metrics.py test_parameter_metrics_depth_first 300
)
shadowspill_add_public_api_test(
  test_parameter_metrics.py test_parameter_metrics_breadth_first 300
)
shadowspill_add_public_api_test(
  test_01_public_training.py test_public_training_declared_adamw_state_replays 300
)
shadowspill_add_public_api_test(
  test_01_public_training.py test_public_training_fills_declared_state_in_the_spill_pool 300
)
shadowspill_add_public_api_test(
  test_01_public_training.py test_public_training_refuses_state_with_no_value_before_its_first_step 300
)
shadowspill_add_public_api_test(
  test_01_public_training.py test_public_training_owns_the_model_state_it_imported 300
)
shadowspill_add_public_api_test(
  test_01_public_training.py test_public_training_saves_straight_from_the_pool 300
)
shadowspill_add_public_api_test(
  test_01_public_training.py test_public_training_steps_masters_of_its_weights 300
)
shadowspill_add_public_api_test(
  test_01_public_training.py test_public_training_steps_masters_beside_weights_that_have_none 300
)
shadowspill_add_public_api_test(
  test_01_public_training.py test_public_training_keeps_masters_gradients_and_moments_at_their_own_dtypes 600
)
shadowspill_add_public_api_test(
  test_01_public_training.py test_public_training_checkpoints_masters_in_place_of_their_weights 300
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
  test_01_public_training.py test_public_training_sets_a_model_buffer_each_step 300
)
shadowspill_add_public_api_test(
  test_01_public_training.py test_public_training_keeps_no_output_the_caller_dropped 300
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
shadowspill_add_public_api_test(
  test_04_plans_side_by_side.py test_a_forward_shares_the_slab_of_the_step_it_evaluates 300
)

shadowspill_add_public_api_test(
  test_05_generic_training.py test_shadowspill_generic_candidates_shared_forward_and_checkpoint 300
)
shadowspill_add_public_api_test(
  test_06_generic_quickstart.py test_generic_quickstart_search_run_plot 300
)
shadowspill_add_public_api_test(
  test_06_generic_quickstart.py test_generic_quickstart_reducer_failure_releases_outputs 300
)

shadowspill_add_public_api_test(
  test_07_distributed_training.py test_distributed_training_matches_combined_reference 1200
)

shadowspill_add_public_api_test(
  test_05_generic_training.py test_mutable_buffer_survives_diagnostics_and_repeated_training 300
)
shadowspill_add_public_api_test(
  test_08_tensor_representations.py test_planned_representation_save 300
)
shadowspill_add_public_api_test(
  test_08_tensor_representations.py test_planned_representation_recompute 300
)
shadowspill_add_public_api_test(
  ../state/test_pool_initialization.py
  test_meta_initialization_and_checkpoint_import_preserve_values_and_ties 180
)
shadowspill_add_public_api_test(
  test_08_tensor_representations.py test_planned_packed_parameter_gradients 300
)


shadowspill_add_public_api_test(
  test_05_generic_training.py test_first_stochastic_update_is_independent_of_preparation_and_diagnostics 300
)
shadowspill_add_public_api_test(
  test_00_public_forward.py test_public_forward_profiling_preserves_random_streams 300
)

shadowspill_add_public_api_test(
  test_cpu_controls.py test_cpu_outputs_and_mixed_device_controls_from_ssd 300
)

shadowspill_add_public_api_test(
  test_cpu_literals.py test_cpu_literal_forward_from_ssd 300
)
shadowspill_add_public_api_test(
  test_cpu_literals.py test_cpu_literal_training_from_ssd 300
)

shadowspill_add_public_api_test(
  test_partial_checkpoint.py test_partial_checkpoint_keeps_mutations_and_replays_from_ssd 300
)
