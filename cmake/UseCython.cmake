# Define a function to create Cython modules.
#
# For more information on the Cython project, see http://cython.org/.
# "Cython is a language that makes writing C extensions for the Python language
# as easy as Python itself."
#
# This file defines a CMake function to build a Cython Python module.
# To use it, first include this file.
#
#   include( UseCython )
#
# Then call cython_add_module to create a module.
#
#   cython_add_module( <module_name> <src1> <src2> ... <srcN> )
#
# To create a standalone executable, the function
#
#   cython_add_standalone_executable( <executable_name> [MAIN_MODULE src1] <src1> <src2> ... <srcN> )
#
# To avoid dependence on Python, set the PYTHON_LIBRARY cache variable to point
# to a static library.  If a MAIN_MODULE source is specified,
# the "if __name__ == '__main__':" from that module is used as the C main() method
# for the executable.  If MAIN_MODULE, the source with the same basename as
# <executable_name> is assumed to be the MAIN_MODULE.
#
# Where <module_name> is the name of the resulting Python module and
# <src1> <src2> ... are source files to be compiled into the module, e.g. *.pyx,
# *.py, *.c, *.cxx, etc.  A CMake target is created with name <module_name>.  This can
# be used for target_link_libraries(), etc.
#
# The sample paths set with the CMake include_directories() command will be used
# for include directories to search for *.pxd when running the Cython complire.
#
# Cache variables that effect the behavior include:
#
#  CYTHON_ANNOTATE
#  CYTHON_NO_DOCSTRINGS
#  CYTHON_FLAGS
#
# Source file properties that effect the build process are
#
#  CYTHON_IS_CXX
#
# If this is set of a *.pyx file with CMake set_source_files_properties()
# command, the file will be compiled as a C++ file.
#
# See also FindCython.cmake

#=============================================================================
# Copyright 2011 Kitware, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

# Configuration options.
set( CYTHON_ANNOTATE OFF
  CACHE BOOL "Create an annotated .html file when compiling *.pyx." )
set( CYTHON_NO_DOCSTRINGS OFF
  CACHE BOOL "Strip docstrings from the compiled module." )
set( CYTHON_FLAGS "" CACHE STRING
  "Extra flags to the cython compiler." )
mark_as_advanced( CYTHON_ANNOTATE CYTHON_NO_DOCSTRINGS CYTHON_FLAGS )

find_package( Cython REQUIRED )
find_package( PythonLibs REQUIRED )

set( CYTHON_CXX_EXTENSION "cxx" )
set( CYTHON_C_EXTENSION "c" )

# Create a *.c or *.cxx file from a *.pyx file.
# Input the generated file basename.  The generate file will put into the variable
# placed in the "generated_file" argument. Finally all the *.py and *.pyx files.
function( cythonize_pyx _name generated_file )
  # Default to assuming all files are C.
  set( cxx_arg "" )
  set( extension ${CYTHON_C_EXTENSION} )
  set( pyx_lang "C" )
  set( comment "Compiling Cython C source for ${_name}..." )

  set( cython_include_directories "" )
  set( pxd_dependencies "" )
  set( pxi_dependencies "" )
  set( c_header_dependencies "" )
  set( pyx_locations "" )

  # append cythonize include directories.
  foreach( _include_dir ${CYTHONIZE_INCLUDE_DIRS} )
    set( include_directory_arg ${include_directory_arg} "-I" "${_include_dir}" )
  endforeach()
 
  # append cythonize library directories.
  set( library_directory_arg "" )
  foreach( _lib_dir ${CYTHONIZE_LIBRARY_DIRS} )
    set( library_directory_arg ${library_directory_arg} "-L" "${_lib_dir}" )
  endforeach()

  foreach( pyx_file ${ARGN} )
    get_filename_component( pyx_file_basename "${pyx_file}" NAME_WE )

    # Determine if it is a C or C++ file.
    get_source_file_property( property_is_cxx ${pyx_file} CYTHON_IS_CXX )
    if( ${property_is_cxx} )
      set( cxx_arg "--cplus" )
      set( extension ${CYTHON_CXX_EXTENSION} )
      set( pyx_lang "CXX" )
      set( comment "Compiling Cython CXX source for ${_name}..." )
    endif()

    # Get the include directories.
    get_source_file_property( pyx_location ${pyx_file} LOCATION )
    get_filename_component( pyx_path ${pyx_location} PATH )
    get_directory_property( cmake_include_directories DIRECTORY ${pyx_path} INCLUDE_DIRECTORIES )
    list( APPEND cython_include_directories ${cmake_include_directories} )
    list( APPEND pyx_locations "${pyx_location}" )

    # Determine dependencies.
    # Add the pxd file will the same name as the given pyx file.
    unset( corresponding_pxd_file CACHE )
    find_file( corresponding_pxd_file ${pyx_file_basename}.pxd
      PATHS "${pyx_path}" ${cmake_include_directories}
      NO_DEFAULT_PATH )
    if( corresponding_pxd_file )
      list( APPEND pxd_dependencies "${corresponding_pxd_file}" )
    endif()

    # Look for included pxi files
    file(STRINGS "${pyx_file}" include_statements REGEX "include +['\"]([^'\"]+).*")
    foreach(statement ${include_statements})
      string(REGEX REPLACE "include +['\"]([^'\"]+).*" "\\1" pxi_file "${statement}")
      unset(pxi_location CACHE)
      find_file(pxi_location ${pxi_file}
        PATHS "${pyx_path}" ${cmake_include_directories} NO_DEFAULT_PATH)
      if (pxi_location)
        list(APPEND pxi_dependencies ${pxi_location})
        get_filename_component( found_pyi_file_basename "${pxi_file}" NAME_WE )
        get_filename_component( found_pyi_path ${pxi_location} PATH )
        unset( found_pyi_pxd_file CACHE )
        find_file( found_pyi_pxd_file ${found_pyi_file_basename}.pxd
          PATHS "${found_pyi_path}" ${cmake_include_directories} NO_DEFAULT_PATH )
        if (found_pyi_pxd_file)
            list( APPEND pxd_dependencies "${found_pyi_pxd_file}" )
        endif()
      endif()
    endforeach() # for each include statement found

    # pxd files to check for additional dependencies.
    set( pxds_to_check "${pyx_file}" "${pxd_dependencies}" )
    set( pxds_checked "" )
    set( number_pxds_to_check 1 )
    while( ${number_pxds_to_check} GREATER 0 )
      foreach( pxd ${pxds_to_check} )
        list( APPEND pxds_checked "${pxd}" )
        list( REMOVE_ITEM pxds_to_check "${pxd}" )

        # check for C header dependencies
        file( STRINGS "${pxd}" extern_from_statements
          REGEX "cdef[ ]+extern[ ]+from.*$" )
        foreach( statement ${extern_from_statements} )
          # Had trouble getting the quote in the regex
          string( REGEX REPLACE "cdef[ ]+extern[ ]+from[ ]+[\"]([^\"]+)[\"].*" "\\1" header "${statement}" )
          unset( header_location CACHE )
          find_file( header_location ${header} PATHS ${cmake_include_directories} )
          if( header_location )
            list( FIND c_header_dependencies "${header_location}" header_idx )
            if( ${header_idx} LESS 0 )
              list( APPEND c_header_dependencies "${header_location}" )
            endif()
          endif()
        endforeach()

        # check for pxd dependencies

        # Look for cimport statements.
        set( module_dependencies "" )
        file( STRINGS "${pxd}" cimport_statements REGEX cimport )
        foreach( statement ${cimport_statements} )
          if( ${statement} MATCHES from )
            string( REGEX REPLACE "from[ ]+([^ ]+).*" "\\1" module "${statement}" )
          else()
            string( REGEX REPLACE "cimport[ ]+([^ ]+).*" "\\1" module "${statement}" )
          endif()
          list( APPEND module_dependencies ${module} )
        endforeach()
        list( REMOVE_DUPLICATES module_dependencies )
        # Add the module to the files to check, if appropriate.
        foreach( module ${module_dependencies} )
          unset( pxd_location CACHE )
          find_file( pxd_location ${module}.pxd
            PATHS "${pyx_path}" ${cmake_include_directories} NO_DEFAULT_PATH )
          if( pxd_location )
            list( FIND pxds_checked ${pxd_location} pxd_idx )
            if( ${pxd_idx} LESS 0 )
              list( FIND pxds_to_check ${pxd_location} pxd_idx )
              if( ${pxd_idx} LESS 0 )
                list( APPEND pxds_to_check ${pxd_location} )
                list( APPEND pxd_dependencies ${pxd_location} )
              endif() # if it is not already going to be checked
            endif() # if it has not already been checked
          endif() # if pxd file can be found
        endforeach() # for each module dependency discovered
      endforeach() # for each pxd file to check
      list( LENGTH pxds_to_check number_pxds_to_check )
    endwhile()



  endforeach() # pyx_file

  # Set additional flags.
  if( CYTHON_ANNOTATE )
    set( annotate_arg "--annotate" )
  endif()

  if( CYTHON_NO_DOCSTRINGS )
    set( no_docstrings_arg "--no-docstrings" )
  endif()

  if( "${CMAKE_BUILD_TYPE}" STREQUAL "Debug" OR
        "${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo" )
      set( cython_debug_arg "--gdb" )
  endif()

  if( "${PYTHONLIBS_VERSION_STRING}" MATCHES "^2." )
    set( version_arg "-2" )
  elseif( "${PYTHONLIBS_VERSION_STRING}" MATCHES "^3." )
    set( version_arg "-3" )
  else()
    set( version_arg )
  endif()

  # Include directory arguments.
  list( REMOVE_DUPLICATES cython_include_directories )
  set( include_directory_arg "" )
  foreach( _include_dir ${cython_include_directories} )
    #set( include_directory_arg ${include_directory_arg} "-I" "${_include_dir}" )
  endforeach()


  # Determining generated file name.
  set( _generated_file "${CMAKE_CURRENT_BINARY_DIR}/${_name}.${extension}" )
  set_source_files_properties( ${_generated_file} PROPERTIES GENERATED TRUE )
  set( ${generated_file} ${_generated_file} PARENT_SCOPE )

  list( REMOVE_DUPLICATES pxd_dependencies )
  list( REMOVE_DUPLICATES c_header_dependencies )


  # Remove their visibility to the user.
  set( corresponding_pxd_file "" CACHE INTERNAL "" )
  set( header_location "" CACHE INTERNAL "" )
  set( pxd_location "" CACHE INTERNAL "" )
endfunction()

set ( include_directory_arg "" )
set ( library_directory_arg "" )
set ( cython_build_libs "" )
set ( pyx_sources "" )

function( sample _name )

	if( "${CMAKE_BUILD_TYPE}" STREQUAL "Debug" OR
	"${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo" )
		set( cython_debug_arg "--gdb" )
	endif()

	if( "${PYTHONLIBS_VERSION_STRING}" MATCHES "^2." )
		set( version_arg "-2" )
	elseif( "${PYTHONLIBS_VERSION_STRING}" MATCHES "^3." )
		set( version_arg "-3" )
	else()
		set( version_arg )
	endif()

	# get includes from parent.
	get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
	foreach(_include_dir ${dirs})
		set( include_directory_arg ${include_directory_arg} "-I" "${_include_dir}" )
	endforeach()

	# get libraries from parent.
	get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY LINK_DIRECTORIES)
	foreach(_link_dir ${dirs})
		set( library_directory_arg ${library_directory_arg} "-L" "${_link_dir}" )
	endforeach()

	# get libs.
	foreach(_lib ${CYTHON_LIBS})
		set( cython_build_libs ${cython_build_libs} "-l" "${_lib}" )
	endforeach()

	# get cython compiler flags.

	# build sources.
	foreach(_pyx_file ${ARGN})
		set( pyx_sources ${pyx_sources} "" "${CMAKE_CURRENT_SOURCE_DIR}/${_pyx_file}" )
	endforeach()
	# cythonize compilation command.
	add_custom_command(OUTPUT "_CYT${_name}"
		COMMAND (${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/setup.py
		${include_directory_arg} # include directories.
		${library_directory_arg} # library directories.
		${cython_build_libs} # libraries.
		-o ${CMAKE_CURRENT_BINARY_DIR} # output directory.
		-c ${CYTHON_CXX_FLAGS}
		${version_arg} # python version.
		${pyx_sources} # pyx files.
		)
	)

endfunction()

include( CMakeParseArguments )

#  LIBS libs INCLUDES incs SRCS srcs

# cython_add_module( <name> src1 src2 ... srcN )
# Build the Cython Python module.
function( cython_add_module _name )

	foreach( _file ${ARGN} )
		if( ${_file} MATCHES ".*\\.py[x]?$" )
			list( APPEND pyx_locations ${_file} )
		else()
			list( APPEND other_module_sources ${_file} )
		endif()
	endforeach()

	# setup.
	sample( ${_name} ${pyx_locations} )

	# add custom target = cython module.
	add_custom_target( ${_name}
		DEPENDS "_CYT${_name}"
	)

	include_directories( ${PYTHON_INCLUDE_DIRS} )

endfunction()
