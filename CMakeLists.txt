cmake_minimum_required(VERSION 3.9)

project(gaussian_process_regression VERSION 0.0.0)

# Build options
option(BUILD_TESTING "Build all tests." OFF)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(Eigen3 REQUIRED)
include_directories(${Eigen3_INCLUDE_DIRS})

if(BUILD_TESTING)
  enable_testing()
  find_package(GTest REQUIRED)
  if (APPLE)
    add_definitions(-DGTEST_USE_OWN_TR1_TUPLE)
    add_definitions(-D__GLIBCXX__)
  endif (APPLE)
else()
  find_package(GTest QUIET)
endif()

include_directories(include)

install(DIRECTORY include/
  DESTINATION include
)

if (BUILD_TESTING)
  add_executable(${PROJECT_NAME}_test test/test_gpr.cpp)
  target_link_libraries(${PROJECT_NAME}_test
          ${GTEST_LIBRARIES}
          pthread
          )
  add_test(NAME ${PROJECT_NAME}_test COMMAND ${PROJECT_NAME}_test)
endif ()