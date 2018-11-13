# OpenFst - Release 1.3.6

## Python Wrapper

TODO:

## Windows installation

Use vcpkg to install dependencies under windows. In order to setup vcpkg for system
wide usage, execute the following command in the root directory of vcpkg (requires 
admin on first use):

	.\vcpkg integrate install

Use 

	"-DCMAKE_TOOLCHAIN_FILE=[vcpkg-root]/scripts/buildsystems/vcpkg.cmake" 

when building the project with CMake.

### Dependencies:

* dlfcn-win32:x64-windows-static
* mman:x64-windows-static
* psapi

# windows stuff

## how to determine whether build release or debug

dumpbin /directives

/MT		=> LIBCMT.LIB, LIBCPMT.LIB
/MTd	=> LIBCMTD.LIB, LIBCPMTD.LIB
/MD		=> MSVCRT.LIB, MSVCPRT.LIB
/MDd	=> MSVCRTD.LIB, MSVCPRTD.LIB

mman.lib|dl.lib 

static relase 

	/DEFAULTLIB LIBCMT => MT

static debug 

	/DEFAULTLIB LIBCMTD => MTd

openfst.lib

?  
	/DEFAULTLIB:msvcprt
	/FAILIFMISMATCH:_CRT_STDIO_ISO_WIDE_SPECIFIERS=0
	/DEFAULTLIB:MSVCRT