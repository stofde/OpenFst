# CMakeList.txt : Python extension.
#
cmake_minimum_required (VERSION 3.0.2)
project (openfst_python)

# TODO: Constrain python version.
# This library requires python 3.4+
find_package( PythonInterp 3.6 REQUIRED )
find_package( PythonLibs 3.6 REQUIRED )


# With CMake, a clean separation can be made between the source tree and the
# build tree.  When all source is compiled, as with pure C/C++, the source is
# no-longer needed in the build tree.  However, with pure *.py source, the
# source is processed directly.  To handle this, we reproduce the availability
# of the source files in the build tree.
add_custom_target( ReplicatePythonSourceTree ALL ${CMAKE_COMMAND} -P
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ReplicatePythonSourceTree.cmake
  ${CMAKE_CURRENT_BINARY_DIR}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )

# Include the CMake script UseCython.cmake. 
include( UseCython )

# openfst include directory.
include_directories( ../../src/include )

if (WIN32)

	#################################################
	# windows instructions.
	#################################################

	# static mman include.
	include_directories( "${VCPKG_ROOT_DIR}/installed/${WIN_BUILD_PREFIX}-windows-static/include/mman" )

	# global vcpkg includes.
	include_directories( "${VCPKG_ROOT_DIR}/installed/${WIN_BUILD_PREFIX}-windows-static/include" )

	# global vcpkg static lib directory.
	link_directories( "${VCPKG_ROOT_DIR}/installed/${WIN_BUILD_PREFIX}-windows-static/lib" )

	# openfst core lib.
	link_directories( "${CMAKE_CURRENT_BINARY_DIR}/../../projects/openfst" )

	# find psapi; required by dlfcn-win32.
	find_library (PSAPI Psapi)

	# set cython libraries.
	set ( CYTHON_LIBS  
		openfst_lib
		dl
		mman
		psapi
	)

	# big objects expected.
	add_compile_options(-bigobj)

	# ensure static (MT) build of library.
	set ( CYTHON_CXX_FLAGS /MT )

else ()
	#################################################
	# linux instructions.
	#################################################
	message( FATAL_ERROR "specific linux/mac build instructions" )
endif ()


# If the pyx file is a C++ file, we should specify that here.
set_source_files_properties(
	openpyfst.pyx
	PROPERTIES 
		CYTHON_IS_CXX TRUE
)

# add cython module.
# Multi-file cython modules do not appear to be working at the moment.
cython_add_module( 
	${PROJECT_NAME} 
	openpyfst.pyx
)