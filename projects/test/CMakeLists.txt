# CMakeList.txt : Core slgenerics project.
#
cmake_minimum_required (VERSION 3.0.2)
project (openfst_test)

#set(dlfcn-win32_DIR "../../../vcpkg/installed/x86-windows/share/dlfcn-win32")
#set(mman_DIR "../../../vcpkg/installed/x86-windows/share/mman")

# openfst include directory.
include_directories(../../src/include)
include_directories("../../../vcpkg/installed/x64-windows/include")
include_directories("../../../vcpkg/installed/x64-windows/include/mman")

set(dlfcn-win32_DIR "../../../vcpkg/installed/x64-windows/share/dlfcn-win32")
set(mman_DIR "../../../vcpkg/installed/x64-windows/share/mman")

link_directories("../../../vcpkg/installed/x64-windows/lib")

#if (WIN32)
  find_package(dlfcn-win32 REQUIRED)
  set(CMAKE_DL_LIBS dlfcn-win32::dl)
  include_directories("../../../vcpkg/installed/x64-windows/include")
#endif ()

#target_link_libraries(${PROJECT_NAME} ${CMAKE_DL_LIBS})

add_compile_options(-bigobj)

# sources.
set(SRCS 
	../../src/test/algo_test.cc
)

# executable.
add_executable(${PROJECT_NAME} "${SRCS}")

target_link_libraries(${PROJECT_NAME}
	openfst_lib  # static openfst lib.
	mman.lib
	dl.lib
)