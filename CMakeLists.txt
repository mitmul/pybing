cmake_minimum_required(VERSION 2.8)
project(bing)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3")

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindTBB.cmake)

find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS system filesystem python REQUIRED)
find_package(PythonLibs REQUIRED)

if (${Boost_NumPy_INCLUDE_DIR} STREQUAL "")
  message(FATAL_ERROR "Boost_NumPy_INCLUDE_DIR is not set")
endif()

if (${Boost_NumPy_LIBRARY_DIR} STREQUAL "")
  message(FATAL_ERROR "Boost_NumPy_LIBRARY_DIR is not set")
endif()

set(inc_dirs
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${Boost_NumPy_INCLUDE_DIR}
  ${PYTHON_INCLUDE_DIRS}
)

set(lib_dirs
  ${TBB_LIBRARY_DIRS}
  ${Boost_NumPy_LIBRARY_DIR}
)

set(libs
  boost_numpy
  ${Boost_LIBRARIES}
  ${PYTHON_LIBRARY}
  ${OpenCV_LIBS}
)

include_directories(${inc_dirs})

link_directories(${lib_dirs})

add_library(bing SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/bing.cpp)

target_link_libraries(bing ${libs})

set_target_properties(bing PROPERTIES PREFIX "")
