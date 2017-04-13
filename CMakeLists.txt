cmake_minimum_required(VERSION 3.6)
project(machine_learning)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp)
add_executable(machine_learning ${SOURCE_FILES})