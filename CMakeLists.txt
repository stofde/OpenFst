# CMakeList.txt : Core slgenerics project.
#
cmake_minimum_required (VERSION 3.0.2)
project (openfst)

# You can tweak some common (for all subprojects) stuff here. For example:
#set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
#set(CMAKE_DISABLE_SOURCE_CHANGES  ON)

if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  message(SEND_ERROR "In-source builds are not allowed.")
endif ()

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_COLOR_MAKEFILE   ON)

# Remove 'lib' prefix for shared libraries on Windows
if (WIN32)
  set(CMAKE_SHARED_LIBRARY_PREFIX "")

endif ()

##############################################
# projects

# core openfst library.
add_subdirectory(projects/openfst)

# openfst dll.
#add_subdirectory(projects/openfstdll)

# test library.
add_subdirectory(projects/test)