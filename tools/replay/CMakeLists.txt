# tools/replay - v1.14 parity-sweep harness and tests.

add_executable(dirtybird-miner-cpu-replay
    replay_harness.cpp
    report.cpp
    replay_globals.cpp
)
target_include_directories(dirtybird-miner-cpu-replay PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(dirtybird-miner-cpu-replay PRIVATE dluna_stages)
dluna_copy_runtime_dlls(dirtybird-miner-cpu-replay)

add_executable(bench_stage5 bench_stage5.cpp replay_globals.cpp)
target_include_directories(bench_stage5 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(bench_stage5 PRIVATE dluna_stages)
dluna_copy_runtime_dlls(bench_stage5)

add_executable(test_cap_format test_cap_format.cpp)
target_include_directories(test_cap_format PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
dluna_copy_runtime_dlls(test_cap_format)

add_executable(test_self_diff test_self_diff.cpp replay_globals.cpp)
target_include_directories(test_self_diff PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(test_self_diff PRIVATE dluna_stages)
dluna_copy_runtime_dlls(test_self_diff)

add_executable(test_v114_encode test_v114_encode.cpp replay_globals.cpp)
target_include_directories(test_v114_encode PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(test_v114_encode PRIVATE dluna_stages)
dluna_copy_runtime_dlls(test_v114_encode)

add_executable(test_stage3_tail test_stage3_tail.cpp replay_globals.cpp)
target_include_directories(test_stage3_tail PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(test_stage3_tail PRIVATE dluna_stages)
dluna_copy_runtime_dlls(test_stage3_tail)

add_executable(test_stage5_descriptor test_stage5_descriptor.cpp replay_globals.cpp)
target_include_directories(test_stage5_descriptor PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(test_stage5_descriptor PRIVATE dluna_stages)
dluna_copy_runtime_dlls(test_stage5_descriptor)

add_executable(test_runtime_tune test_runtime_tune.cpp)
target_include_directories(test_runtime_tune PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(test_runtime_tune PRIVATE dluna_runtime)
dluna_copy_runtime_dlls(test_runtime_tune)

add_executable(test_cpu_topology test_cpu_topology.cpp)
target_include_directories(test_cpu_topology PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(test_cpu_topology PRIVATE dluna_runtime)
dluna_copy_runtime_dlls(test_cpu_topology)

add_executable(test_statusline test_statusline.cpp)
target_include_directories(test_statusline PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(test_statusline PRIVATE dluna_runtime)
dluna_copy_runtime_dlls(test_statusline)

add_executable(test_sleep_ms test_sleep_ms.cpp)
target_include_directories(test_sleep_ms PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(test_sleep_ms PRIVATE dluna_runtime)
dluna_copy_runtime_dlls(test_sleep_ms)

add_executable(test_share_queue test_share_queue.cpp)
target_include_directories(test_share_queue PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(test_share_queue PRIVATE OpenSSL::Crypto)
dluna_copy_runtime_dlls(test_share_queue)

add_test(NAME cap_format COMMAND test_cap_format)
# Runs from the binary output dir, not this target's build dir: the test shells
# out to "./dirtybird-miner-cpu-replay" by relative path, and every executable
# lands in ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}.
add_test(NAME self_diff
    COMMAND test_self_diff
    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)
add_test(NAME v114_encode COMMAND test_v114_encode)
add_test(NAME v114_encode_count1_singletons COMMAND test_v114_encode)
set_tests_properties(v114_encode_count1_singletons PROPERTIES
    ENVIRONMENT "DLUNA_STAGE5_COUNT1_SINGLETONS=1"
)
add_test(NAME v114_encode_count1_false_is_off COMMAND test_v114_encode)
set_tests_properties(v114_encode_count1_false_is_off PROPERTIES
    ENVIRONMENT "DLUNA_STAGE5_COUNT1_SINGLETONS=false"
)
add_test(NAME stage3_tail COMMAND test_stage3_tail)
add_test(NAME stage5_descriptor COMMAND test_stage5_descriptor)
add_test(NAME runtime_tune_policy COMMAND test_runtime_tune)
add_test(NAME cpu_topology_policy COMMAND test_cpu_topology)
add_test(NAME statusline COMMAND test_statusline)
add_test(NAME statusline_env_cols COMMAND test_statusline)
set_tests_properties(statusline_env_cols PROPERTIES
    ENVIRONMENT "DIRTYBIRD_COLS=56"
)
add_test(NAME sleep_ms COMMAND test_sleep_ms)
add_test(NAME share_queue COMMAND test_share_queue)
# Stage5 contract tests need a PowerShell driver plus a capture corpus, neither
# of which ships in this repo. Registering them unconditionally made `ctest` fail
# out of the box on a fresh clone with an error about a missing -File argument,
# which is indistinguishable from a real regression. Register only when the
# fixtures are actually present.
set(STAGE5_FUSED_DRIVER
    ${CMAKE_SOURCE_DIR}/tools/test_stage5_fused_profile_contract.ps1)
set(STAGE5_FUSED_CAPTURES
    ${CMAKE_SOURCE_DIR}/captures/v1.14/869d2bd2_20260501_1810_1000)
if(EXISTS ${STAGE5_FUSED_DRIVER} AND EXISTS ${STAGE5_FUSED_CAPTURES})
    add_test(NAME stage5_fused_profile_contract
        COMMAND powershell -NoProfile -ExecutionPolicy Bypass
            -File ${STAGE5_FUSED_DRIVER}
            -BenchExe $<TARGET_FILE:bench_stage5>
            -Captures ${STAGE5_FUSED_CAPTURES}
    )
else()
    message(STATUS "stage5_fused_profile_contract: fixtures absent, test skipped")
endif()

set(STAGE5_GATE_DRIVER
    ${CMAKE_SOURCE_DIR}/tools/test_stage5_candidate_gate_contract.ps1)
if(EXISTS ${STAGE5_GATE_DRIVER})
    add_test(NAME stage5_candidate_gate_contract
        COMMAND powershell -NoProfile -ExecutionPolicy Bypass
            -File ${STAGE5_GATE_DRIVER}
    )
else()
    message(STATUS "stage5_candidate_gate_contract: driver absent, test skipped")
endif()
