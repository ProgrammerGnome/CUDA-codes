cmake_minimum_required(VERSION 3.8)
project(CudaProject LANGUAGES CUDA CXX)

add_executable(cuda2_m KoMaL_S_141.cu)

install(TARGETS cuda2_m RUNTIME DESTINATION bin)
