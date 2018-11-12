# OpenFst - Release 1.3

## Windows installation instruction.

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
