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
shadowspill_add_pytorch_canary(planning_failure 120)
shadowspill_add_pytorch_canary(profiler 60)
shadowspill_add_pytorch_canary(forward 90)
shadowspill_add_pytorch_canary(state_import_export 90)
shadowspill_add_pytorch_canary(training 120)
shadowspill_add_pytorch_canary(planning_residue 120)

# Older tests install the allocator in pytest; saved-phase tests launch their
# own fresh execution workers. Select the named case explicitly so both forms
# run here, including tests excluded by the ordinary suite's default markers.
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
  shadowspill/execution/test_saved_executor.py
  test_saved_plan_executes_without_build_or_measurement[forward-None-False] 300
)
shadowspill_add_public_api_test(
  shadowspill/execution/test_model_controls.py
  test_model_buffer_controls_in_saved_workers 300
)
shadowspill_add_public_api_test(
  shadowspill/execution/optimizers/test_starts.py
  test_constant_and_parameter_initializers_survive_cold_execution 300
)
shadowspill_add_public_api_test(
  shadowspill/execution/optimizers/test_precision_matrix.py
  test_saved_adam_precision_matrix 600
)
shadowspill_add_public_api_test(
  shadowspill/execution/optimizers/test_precision_matrix.py
  test_mixed_parameters_only_create_masters_where_needed 300
)
shadowspill_add_public_api_test(
  shadowspill/execution/optimizers/test_masters.py
  test_mixed_precision_master_update_and_checkpoint 300
)
shadowspill_add_public_api_test(
  shadowspill/execution/optimizers/test_mlops.py
  test_mlops_adamw_cold_worker_matches_independent_torch_updates 300
)
shadowspill_add_public_api_test(
  shadowspill/build/test_optimizer_initialization.py
  test_gradient_initialized_state_has_no_guess_before_first_step 120
)
shadowspill_add_public_api_test(
  shadowspill/build/test_optimizer_initialization.py
  test_portable_build_refuses_python_optimizer_fallback 120
)
shadowspill_add_public_api_test(
  shadowspill/execution/test_saved_executor.py
  test_saved_plan_executes_without_build_or_measurement[step-save-True] 300
)
shadowspill_add_public_api_test(
  shadowspill/training/test_saved_trainer.py
  test_trainer_updates_and_cold_resume_match_independent_eager_loop 300
)
shadowspill_add_public_api_test(
  shadowspill/training/test_mixed_inputs.py
  test_mixed_calls_keep_weights_metrics_and_group_controls[depth_first] 300
)
shadowspill_add_public_api_test(
  shadowspill/training/test_mixed_inputs.py
  test_mixed_calls_keep_weights_metrics_and_group_controls[breadth_first] 300
)
shadowspill_add_public_api_test(
  shadowspill/execution/test_shared_plans.py
  test_shared_evaluation_and_checkpoint_in_cold_workers 300
)
shadowspill_add_public_api_test(
  shadowspill/profiling/test_pair_lifetimes.py
  test_pair_inputs_are_released_between_measurements_and_on_failure 300
)
shadowspill_add_public_api_test(
  shadowspill/build/test_step.py
  test_adam_updates_and_moments_match_pytorch_without_rebuilding 300
)

shadowspill_add_public_api_test(
  benchmarking/test_generic_quickstart.py
  test_generic_quickstart_search_run_plot 300
)
shadowspill_add_public_api_test(
  benchmarking/test_generic_quickstart.py
  test_generic_quickstart_reducer_failure_releases_outputs 300
)

shadowspill_add_public_api_test(
  shadowspill/pytorch/api/test_07_distributed_training.py
  test_distributed_training_matches_combined_reference 1200
)

shadowspill_add_public_api_test(
  shadowspill/training/test_diagnostics.py
  test_saved_startup_diagnostics_preserve_state_and_rng[False] 300
)
shadowspill_add_public_api_test(
  shadowspill/execution/test_tensor_representations.py
  test_quantized_training_evaluation_and_cold_checkpoint[save] 300
)
shadowspill_add_public_api_test(
  shadowspill/execution/test_tensor_representations.py
  test_quantized_training_evaluation_and_cold_checkpoint[recompute] 300
)
shadowspill_add_public_api_test(
  shadowspill/pytorch/state/test_pool_initialization.py
  test_meta_initialization_and_checkpoint_import_preserve_values_and_ties 180
)
shadowspill_add_public_api_test(
  shadowspill/pytorch/state/test_pool_initialization.py
  test_a_wrapped_weight_in_an_unreadable_pool_is_read_component_by_component 180
)
shadowspill_add_public_api_test(
  shadowspill/execution/test_packed_gradients.py
  test_packed_parameter_gradients_in_cold_worker 300
)

shadowspill_add_public_api_test(
  shadowspill/training/test_diagnostics.py
  test_saved_startup_diagnostics_preserve_state_and_rng[True] 300
)
shadowspill_add_public_api_test(
  shadowspill/execution/test_random.py
  test_build_and_profile_preserve_rng_and_cold_execution_replays_it 300
)

shadowspill_add_public_api_test(
  shadowspill/pytorch/api/test_cpu_controls.py
  test_cpu_outputs_and_mixed_device_controls_from_ssd 300
)

shadowspill_add_public_api_test(
  shadowspill/execution/test_host_metrics.py
  test_host_metrics_and_mixed_shapes_in_saved_training 300
)

shadowspill_add_public_api_test(
  shadowspill/execution/test_mutations.py
  test_saved_buffer_replacement_preserves_state_and_workspace 300
)

shadowspill_add_public_api_test(
  shadowspill/execution/test_resources.py
  test_saved_provider_resource_lifecycle_and_memory[step] 300
)

shadowspill_add_public_api_test(
  shadowspill/training/test_partial_trainer.py
  test_partial_trainer_preserves_masters_mutations_evaluation_and_resume 300
)

shadowspill_add_public_api_test(
  shadowspill/pytorch/api/test_pool_image.py
  test_an_image_is_written_once_and_adopted_without_a_copy 300
)
shadowspill_add_public_api_test(
  shadowspill/pytorch/api/test_pool_image.py
  test_an_image_refuses_state_it_does_not_describe 300
)
shadowspill_add_public_api_test(
  shadowspill/pytorch/api/test_pool_image.py
  test_an_image_pool_needs_an_image_to_adopt 300
)
