# CMakeList.txt : Core slgenerics project.
#
cmake_minimum_required (VERSION 3.0.2)
project (openfst_dll)

# openfst include directory.
include_directories(../../src/include)

set(dlfcn-win32_DIR "../../../vcpkg/installed/x86-windows/share/dlfcn-win32")
set(mman_DIR "../../../vcpkg/installed/x86-windows/share/mman")

include_directories("../../../vcpkg/installed/x86-windows/include/mman")

#find_path(MMAN_DIR NAMES sys/mman.h PATH_SUFFIXES mman)

link_directories("../../../vcpkg/packages/mman_x86-windows/lib")

#if (WIN32)
  find_package(dlfcn-win32 REQUIRED)
  set(CMAKE_DL_LIBS dlfcn-win32::dl)
  include_directories("../../../vcpkg/installed/x86-windows/include")
#endif ()

# sources.
set(SRCS 
)

# dll library.
add_library(${PROJECT_NAME} SHARED "${SRCS}")

target_link_libraries(${PROJECT_NAME}
	openfst_lib
)