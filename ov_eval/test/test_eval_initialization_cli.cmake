if(NOT DEFINED EVAL_EXECUTABLE OR NOT DEFINED TEST_DATA_DIR OR NOT DEFINED TEST_WORK_DIR)
  message(FATAL_ERROR "EVAL_EXECUTABLE, TEST_DATA_DIR, and TEST_WORK_DIR are required")
endif()

set(TEST_DIR "${TEST_WORK_DIR}/eval_initialization_cli")
file(REMOVE_RECURSE "${TEST_DIR}")
file(MAKE_DIRECTORY "${TEST_DIR}")
file(COPY "${TEST_DATA_DIR}/initialization_result_v2.tum" DESTINATION "${TEST_DIR}")
file(COPY "${TEST_DATA_DIR}/groundtruth.tum" DESTINATION "${TEST_DIR}")
file(RENAME "${TEST_DIR}/initialization_result_v2.tum" "${TEST_DIR}/trajectory.tum")

execute_process(
  COMMAND "${EVAL_EXECUTABLE}" "${TEST_DIR}/trajectory.tum" "${TEST_DIR}/groundtruth.tum"
  RESULT_VARIABLE DEFAULT_RESULT
  OUTPUT_VARIABLE DEFAULT_STDOUT
  ERROR_VARIABLE DEFAULT_STDERR
)
if(NOT DEFAULT_RESULT EQUAL 0)
  message(FATAL_ERROR "default evaluator invocation failed: ${DEFAULT_STDERR}")
endif()
if(NOT EXISTS "${TEST_DIR}/trajectory_init_eval.md")
  message(FATAL_ERROR "default report was not created")
endif()
file(READ "${TEST_DIR}/trajectory_init_eval.md" DEFAULT_REPORT)
if(NOT DEFAULT_REPORT STREQUAL DEFAULT_STDOUT)
  message(FATAL_ERROR "default report differs from stdout")
endif()
if(NOT DEFAULT_REPORT MATCHES "^\\| Sequence \\|" OR
   NOT DEFAULT_REPORT MATCHES "\\| 0 \\|.*\\| 2\\.000 \\|" OR
   NOT DEFAULT_REPORT MATCHES "\\| 1 \\|.*\\| 1\\.500 \\|" OR
   NOT DEFAULT_REPORT MATCHES "\\| Mean \\|.*\\| 1\\.750 \\|" OR
   DEFAULT_REPORT MATCHES "Diagnostics|startup|periodic_reset")
  message(FATAL_ERROR "default report is not the compact sequence-plus-mean table")
endif()

execute_process(
  COMMAND "${EVAL_EXECUTABLE}" "${TEST_DIR}/trajectory.tum" "${TEST_DIR}/groundtruth.tum"
          --output "${TEST_DIR}/custom.txt" se3
  RESULT_VARIABLE CUSTOM_RESULT
  OUTPUT_VARIABLE CUSTOM_STDOUT
  ERROR_VARIABLE CUSTOM_STDERR
)
if(NOT CUSTOM_RESULT EQUAL 0 OR NOT EXISTS "${TEST_DIR}/custom.txt")
  message(FATAL_ERROR "custom evaluator invocation failed: ${CUSTOM_STDERR}")
endif()
file(READ "${TEST_DIR}/custom.txt" CUSTOM_REPORT)
if(NOT CUSTOM_REPORT STREQUAL CUSTOM_STDOUT)
  message(FATAL_ERROR "custom report differs from stdout")
endif()

file(SHA256 "${TEST_DIR}/trajectory.tum" TRAJECTORY_HASH_BEFORE)
execute_process(
  COMMAND "${EVAL_EXECUTABLE}" "${TEST_DIR}/trajectory.tum" "${TEST_DIR}/groundtruth.tum"
          --output "${TEST_DIR}/trajectory.tum"
  RESULT_VARIABLE TRAJECTORY_OVERWRITE_RESULT
  OUTPUT_QUIET
  ERROR_QUIET
)
file(SHA256 "${TEST_DIR}/trajectory.tum" TRAJECTORY_HASH_AFTER)
if(TRAJECTORY_OVERWRITE_RESULT EQUAL 0 OR NOT TRAJECTORY_HASH_BEFORE STREQUAL TRAJECTORY_HASH_AFTER)
  message(FATAL_ERROR "evaluator accepted or modified a trajectory used as its report path")
endif()

file(SHA256 "${TEST_DIR}/groundtruth.tum" GROUND_TRUTH_HASH_BEFORE)
execute_process(
  COMMAND "${EVAL_EXECUTABLE}" "${TEST_DIR}/trajectory.tum" "${TEST_DIR}/groundtruth.tum"
          --output "${TEST_DIR}/groundtruth.tum"
  RESULT_VARIABLE GROUND_TRUTH_OVERWRITE_RESULT
  OUTPUT_QUIET
  ERROR_QUIET
)
file(SHA256 "${TEST_DIR}/groundtruth.tum" GROUND_TRUTH_HASH_AFTER)
if(GROUND_TRUTH_OVERWRITE_RESULT EQUAL 0 OR NOT GROUND_TRUTH_HASH_BEFORE STREQUAL GROUND_TRUTH_HASH_AFTER)
  message(FATAL_ERROR "evaluator accepted or modified ground truth used as its report path")
endif()

file(REMOVE_RECURSE "${TEST_DIR}")
