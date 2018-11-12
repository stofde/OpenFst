# CMakeList.txt : Root CMake

cmake_minimum_required (VERSION 3.0.2)
project (openfst_lib)

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

	../../src/script/arcsort.cc
	../../src/script/closure.cc
	../../src/script/compile.cc
	../../src/script/compose.cc
	../../src/script/concat.cc
	../../src/script/connect.cc
	../../src/script/convert.cc
	../../src/script/decode.cc
	../../src/script/determinize.cc
	../../src/script/difference.cc
	../../src/script/draw.cc
	../../src/script/encode.cc
	../../src/script/epsnormalize.cc
	../../src/script/equal.cc
	../../src/script/equivalent.cc
	../../src/script/fst-class.cc
	../../src/script/info.cc
	../../src/script/intersect.cc
	../../src/script/invert.cc
	../../src/script/map.cc
	../../src/script/minimize.cc
	../../src/script/print.cc
	../../src/script/project.cc
	../../src/script/prune.cc
	../../src/script/push.cc
	../../src/script/randequivalent.cc
	../../src/script/randgen.cc
	../../src/script/relabel.cc
	../../src/script/replace.cc
	../../src/script/reverse.cc
	../../src/script/reweight.cc
	../../src/script/rmepsilon.cc
	../../src/script/script-impl.cc
	../../src/script/shortest-distance.cc
	../../src/script/shortest-path.cc
	../../src/script/synchronize.cc
	../../src/script/text-io.cc
	../../src/script/topsort.cc
	../../src/script/union.cc
	../../src/script/verify.cc
	../../src/script/weight-class.cc

	../../src/extensions/compact/compact16_acceptor-fst.cc
	../../src/extensions/compact/compact16_string-fst.cc
	../../src/extensions/compact/compact16_unweighted_acceptor-fst.cc
	../../src/extensions/compact/compact16_unweighted-fst.cc
	../../src/extensions/compact/compact16_weighted_string-fst.cc
	../../src/extensions/compact/compact64_acceptor-fst.cc
	../../src/extensions/compact/compact64_string-fst.cc
	../../src/extensions/compact/compact64_unweighted_acceptor-fst.cc
	../../src/extensions/compact/compact64_unweighted-fst.cc
	../../src/extensions/compact/compact64_weighted_string-fst.cc
	../../src/extensions/compact/compact8_acceptor-fst.cc
	../../src/extensions/compact/compact8_string-fst.cc
	../../src/extensions/compact/compact8_unweighted_acceptor-fst.cc
	../../src/extensions/compact/compact8_unweighted-fst.cc
	../../src/extensions/compact/compact8_weighted_string-fst.cc

	../../src/extensions/const/const16-fst.cc
	../../src/extensions/const/const64-fst.cc
	../../src/extensions/const/const8-fst.cc

	../../src/lib/compat.cc
	../../src/lib/flags.cc
	../../src/lib/fst.cc
	../../src/lib/mapped-file.cc
	../../src/lib/properties.cc
	../../src/lib/symbol-table.cc
	../../src/lib/symbol-table-ops.cc
	../../src/lib/util.cc
)

# static library.
add_library(${PROJECT_NAME} STATIC "${SRCS}")

target_link_libraries( ${PROJECT_NAME} ${LIBS} )