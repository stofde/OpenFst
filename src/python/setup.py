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
import platform
import sysconfig

import distutils.log
distutils.log.set_verbosity(distutils.log.DEBUG) # Set DEBUG level

# CLI arguments.
parser = argparse.ArgumentParser(description='Compile cython extension module')
parser.add_argument('scripts', metavar='scripts', type=str, nargs='+',
                    help='pyx file(s) to compile')
parser.add_argument('-I', dest='include_directories', action='append',
                    help='include directory')
parser.add_argument('-L', dest='library_directories', action='append',
                    help='library directory')
parser.add_argument('-l', dest='libraries', action='append',
                    help='library')
parser.add_argument('-c', dest='c_flags', action='append',
                    help='compiler flags')
parser.add_argument('-u', dest='l_flags', action='append',
                    help='linker flags')
parser.add_argument('-o', dest='out_dir', action='store',
                    help='output directory')
parser.add_argument('-2', dest='py2', action='store_true', default=False)  # python 2 flag.
parser.add_argument('-3', dest='py3', action='store_true', default=False)  # python 3 flag.

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

def cythonize_new(include_dirs, libraries_dirs, libraries, c_flags, l_flags, o_dir, py_version, pyx_files):

    # inline directory.
    lib_dir = o_dir
    if not lib_dir:
        lib_dir = os.path.join(get_cython_cache_dir(), 'inline')
    if pyx_files is None or len(pyx_files) != 1:
        raise ValueError('expected single file but got "%s"'
                         % (pyx_files,))
    pyx_file = pyx_files[0]

    if not os.path.isfile(pyx_file):
        raise ValueError('"%s" not a valid file'
                         % (pyx_file,))
    f_name = os.path.split(pyx_file)[1]
    module_name = os.path.splitext(f_name)[0]
    # todo: check if module_name is valid.
    build_extension = get_build_extension()
    if build_extension is None:
        raise ValueError('fail')
    mod_ext = build_extension.get_ext_filename('')
    full_module_name = '%s%s' % (module_name, mod_ext)
    module_path = os.path.join(lib_dir, full_module_name)
    quiet = True

    print('cythonize {')
    print('   include_dirs: %s' % (include_dirs,))
    print('   lib_dirs: %s' % (libraries_dirs,))
    print('   libs: %s' % (libraries,))
    print('   cxx_flags: %s' % (c_flags,))
    print('   l_flags: %s' % (l_flags,))
    print('   file: %s' % (pyx_file,))
    print('   out_file: %s' % (module_path,))
    print('   py_version: %s' % (py_version,))
    print('}')

    #pyx_file = "C:/Users/user/source/repos/openfst/OpenFst/src/python/openpyfst.pyx"
    #module_path = r'C:\Users\user\.cython\inline\openpyfst.cp36 - win_amd64.pyd'
    cython_compiler_directives = []
    _cython_inline_default_context = create_context(('.',))
    cython_include_dirs = None

    ctx = create_context(tuple(cython_include_dirs)) if cython_include_dirs else _cython_inline_default_context

    # build include dirs and cflags.
    #cflags = c_flags #['/MT'] # no debug symbols.
    #lflags = l_flags # ['/verbose:lib']
    lang = "c++" # c++
    extension = Extension(
        name=module_name,
        sources=[pyx_file],
        include_dirs=include_dirs,
        library_dirs=libraries_dirs,
        libraries=libraries,
        extra_compile_args=c_flags,
        extra_link_args=l_flags,
        language=lang)
    build_extension.extensions = cythonize(
        [extension],
        include_path=cython_include_dirs or ['.'],
        compiler_directives=cython_compiler_directives,
        quiet=quiet)

    build_extension.build_temp = lib_dir
    build_extension.build_lib  = lib_dir
    build_extension.verbose = not quiet
    build_extension.run()
    print('done', build_extension)

def main():
    # parse arguments.
    args = parser.parse_args()
    py_version = 0
    if args.py2 and args.py3:
        raise ValueError('-2 and -3 cannot be set at same time.')
    elif args.py2:
        py_version = 2
    elif args.py3:
        py_version = 3
    else:
        print('-2 or -3 set. defaulting to -3')
        py_version = 3
    cythonize_new(args.include_directories, args.library_directories,
                  args.libraries, args.c_flags, args.l_flags, args.out_dir, py_version, args.scripts)

if __name__ == '__main__':
    main()