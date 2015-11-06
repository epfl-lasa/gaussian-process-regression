cmake_minimum_required(VERSION 2.8.3)
project(gaussian_process_regression)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(CMAKE_BUILD_TYPE Release)
## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages

## Boost.Numpy is not yet part of the official boost
## I have forked it into a catkin package that is found here
# find_package(catkin REQUIRED COMPONENTS
#   boost_numpy)

## for python wrapper
FIND_PACKAGE(PythonInterp)
FIND_PACKAGE(PythonLibs 2.7 REQUIRED)
FIND_PACKAGE(Boost COMPONENTS python)

#catkin_python_setup()

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  INCLUDE_DIRS include
#  LIBRARIES gaussian_process_regression
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(include
  ${catkin_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  )

## Declare a cpp library
# add_library(gaussian_process_regression
#   src/${PROJECT_NAME}/gaussian_process_regression.cpp
# )

## PYTHON INTERFACE ##
PYTHON_ADD_MODULE(gaussian_process_regression src/gpr_python_interface.cpp)
target_link_libraries(gaussian_process_regression ${Boost_LIBRARIES}  ${catkin_LIBRARIES} ${Python_LIBRARIES})
set_target_properties(gaussian_process_regression PROPERTIES
        PREFIX ""
        LIBRARY_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_PYTHON_DESTINATION}
        )
#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
catkin_add_gtest(${PROJECT_NAME}-test test/test_gpr.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/test)
if(TARGET ${PROJECT_NAME}-test)
#  target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
endif()

## Add folders to be run by python nosetests
catkin_add_nosetests(test_python)
# export inofrmation for autocompletion tools 
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
