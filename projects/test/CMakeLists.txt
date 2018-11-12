# CMakeList.txt : Core slgenerics project.
#
cmake_minimum_required (VERSION 3.0.2)
project (openfst_test)

# openfst include directory.
include_directories(../../src/include)

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

	# libs.
	set (LIBS 
		mman.lib
		dl.lib
	)

	# check if libs exist.
	if ( EXISTS "${VCPKG_ROOT_DIR}/installed/${WIN_BUILD_PREFIX}-windows-static/lib/mman.lib"
	AND EXISTS "${VCPKG_ROOT_DIR}/installed/${WIN_BUILD_PREFIX}-windows-static/lib/dl.lib")
	else()
		message( FATAL_ERROR "Libraries'${LIBS}' not found" )
	endif()

else ()
	#################################################
	# linux instructions.
	#################################################
	message( FATAL_ERROR "specific linux/mac build instructions" )
endif ()

# large objects expected.
add_compile_options(-bigobj)

# sources.
set(SRCS 
	../../src/test/algo_test.cc
)

# executable.
add_executable(${PROJECT_NAME} "${SRCS}")

target_link_libraries(${PROJECT_NAME}
	openfst_lib  # static openfst lib.
)