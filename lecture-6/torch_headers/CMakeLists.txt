cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(torch_test)

set(Torch_DIR ~/Downloads/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)

add_executable(test main.cpp)
target_link_libraries(test "${TORCH_LIBRARIES}")
set_property(TARGET test PROPERTY CXX_STANDARD 17)

