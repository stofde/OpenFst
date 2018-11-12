""" Special Cython extension builder.

Unfortunately, the inbuilt cython and cythonize command line utilities do no 
support the generation of python extension modules out of the box. More
precisely, the cython command is able to produce a pyd/so file but does not 
export a python module initializer. cythonize, on the other hand, cannot be 
passed additional includes and is therefore not able to compile more complex
projects.

This file requires python 2.7+ and cython to be installed.
"""

# todo: check python version.

# check if cython is installed.
try:
	import Cython
except:
	raise ValueError('Cython not installed')

from Cython.Build.Dependencies import cythonize
from distutils.core import Extension
from Cython.Utils import get_cython_cache_dir
import os
import argparse

# CLI arguments.
parser = argparse.ArgumentParser(description='Compile cython extension module')
parser.add_argument('scripts', metavar='S', type=str, nargs='+',
                    help='pyx file(s) to compile')
parser.add_argument('-I', dest='include_dirs', action='append',
                    help='include directory')

def get_build_extension():
    from distutils.core import Distribution, Extension
    from distutils.command.build_ext import build_ext

    dist = Distribution()
    # Ensure the build respects distutils configuration by parsing
    # the configuration files
    config_files = dist.find_config_files()
    dist.parse_config_files(config_files)
    build_extension = build_ext(dist)
    build_extension.finalize_options()
    return build_extension

def create_context(cython_include_dirs):
    from Cython.Compiler.Main import Context, default_options
    return Context(list(cython_include_dirs), default_options)

def cythonize_new(include_dirs, pyx_files):

	print(include_dirs)
	print(pyx_files)
	lib_dir = os.path.join(get_cython_cache_dir(), 'inline')

	build_extension = get_build_extension()
	if build_extension is None:
		raise ValueError('fail')
	mod_ext = build_extension.get_ext_filename('')
	pyx_file = '%s.pyx' % (module_name,)
	full_module_name = '%s%s' % (module_name, mod_ext)
	pxy_file_path = os.path.join(lib_dir, pyx_file)
	module_path = os.path.join(lib_dir, full_module_name)
	quiet = True

	cython_compiler_directives = []
	_cython_inline_default_context = create_context(('.',))
	cython_include_dirs = None

	ctx = create_context(tuple(cython_include_dirs)) if cython_include_dirs else _cython_inline_default_context

	# build include dirs and cflags.
	c_include_dirs = []
	cflags = []

	fh = open(pxy_file_path, 'w')
	try:
		fh.write(module_code)
	finally:
		fh.close()
	extension = Extension(
		name=module_name,
		sources=[pxy_file_path],
		include_dirs=c_include_dirs,
		extra_compile_args=cflags)

	build_extension.extensions = cythonize(
		[extension],
		include_path=cython_include_dirs or ['.'],
		compiler_directives=cython_compiler_directives,
		quiet=quiet)
	build_extension.build_temp = os.path.dirname(pyx_file)
	build_extension.build_lib  = lib_dir
	build_extension.verbose = not quiet
	build_extension.run()

def main():
	# parse arguments.
	args = parser.parse_args()
	cythonize_new(args.include_dirs, args.scripts)

if __name__ == '__main__':
    main()
