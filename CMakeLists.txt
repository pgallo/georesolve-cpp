cmake_minimum_required(VERSION 3.16)
project(georesolve VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(GEORESOLVE_BUILD_TESTS "Build unit tests" ON)

add_library(georesolve
  src/json.cpp
  src/tokenizer.cpp
  src/gazetteer.cpp
  src/trie.cpp
  src/matcher.cpp
  src/geo.cpp
  src/tfidf.cpp
  src/knn.cpp
  src/features.cpp
  src/hmm.cpp
  src/transitions.cpp
  src/corpus.cpp
  src/resolver.cpp
  src/evaluation.cpp
)
target_include_directories(georesolve PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(georesolve PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion)
endif()

set(GEORESOLVE_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data")

add_executable(georesolve_cli app/main.cpp)
set_target_properties(georesolve_cli PROPERTIES OUTPUT_NAME georesolve)
target_link_libraries(georesolve_cli PRIVATE georesolve)
target_compile_definitions(georesolve_cli PRIVATE GEORESOLVE_DATA_DIR="${GEORESOLVE_DATA_DIR}")

if(GEORESOLVE_BUILD_TESTS)
  enable_testing()
  find_package(GTest REQUIRED)
  add_executable(georesolve_tests
    tests/test_tokenizer_trie.cpp
    tests/test_geo_tfidf.cpp
    tests/test_hmm.cpp
    tests/test_features.cpp
    tests/test_end_to_end.cpp
  )
  target_link_libraries(georesolve_tests PRIVATE georesolve GTest::gtest GTest::gtest_main)
  target_compile_definitions(georesolve_tests PRIVATE GEORESOLVE_DATA_DIR="${GEORESOLVE_DATA_DIR}")
  include(GoogleTest)
  gtest_discover_tests(georesolve_tests)

  # The CLI regression gate also runs under ctest.
  add_test(NAME cli_eval_gate COMMAND georesolve_cli --eval)
endif()
