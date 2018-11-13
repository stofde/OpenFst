# CMakeList.txt : Core slgenerics project.
#
cmake_minimum_required (VERSION 3.0.2)
project (openfst)

# You can tweak some common (for all subprojects) stuff here. For example:
#set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
#set(CMAKE_DISABLE_SOURCE_CHANGES  ON)

# Make the scripts available in the 'cmake' directory available for the
# 'include()' command, 'find_package()' command.
set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake )

if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  message(SEND_ERROR "In-source builds are not allowed.")
endif ()

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_COLOR_MAKEFILE   ON)

set( USE_VCPKG ON )

# vcpkg root directory in case vcpkg is used.
set( VCPKG_ROOT_DIR "C:/users/user/source/repos/vcpkg" )

###############################################
# check if 32 bit or 64 bit.
set ( IS_64BIT_BUILD ON )

if("${CMAKE_SIZEOF_VOID_P}" STREQUAL "4")
	set ( IS_64BIT_BUILD OFF )
endif()

if(NOT CMAKE_CL_64)
	set ( IS_64BIT_BUILD OFF )
endif()

if ( ${IS_64BIT_BUILD} )
	set ( WIN_BUILD_PREFIX "x64" )
else()
	set ( WIN_BUILD_PREFIX "x86" )
endif()

###############################################


# Remove 'lib' prefix for shared libraries on Windows
if (WIN32)
	set(CMAKE_SHARED_LIBRARY_PREFIX "")
	# check if vcpkg can be found.
	if (VCPKG_ROOT_DIR)
		# check if valid directory.
		if( IS_DIRECTORY ${VCPKG_ROOT_DIR} )
			if( EXISTS "${VCPKG_ROOT_DIR}/.vcpkg-root" )
				# found.
			else()
				message( FATAL_ERROR "path '${VCPKG_ROOT_DIR}' not valid vcpkg root directory. Missing '.vcpkg-root' file" )
			endif()
		else()
			message( FATAL_ERROR "path '${VCPKG_ROOT_DIR}' not valid vcpkg root directory" )
		endif()
	else()
		message( FATAL_ERROR "vcpkg root directory not set. set(VCPKG_ROOT_DIR <path>)" )
	endif()
endif ()



##############################################
# projects

# TODO: remove
#add_subdirectory(projects/openfst_tt)

# core openfst library.
add_subdirectory(projects/openfst)

# openfst dll.
#add_subdirectory(projects/openfstdll)

# test library.
add_subdirectory(projects/test)

# examples.
add_subdirectory(projects/examples)

# Python fst wrapper.
add_subdirectory(src/python)