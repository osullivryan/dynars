# Build the dynars C and Fortran FFI examples.
#
#   make            # build the Rust lib (--features ffi) + both examples
#   make c_example  # just the C example
#   make f_example  # just the Fortran example
#   make clean
#
# Requires: cargo, a C compiler (CC), and gfortran (FC) for the Fortran example.

ROOT   := ../..
LIBDIR := $(ROOT)/target/release

CC     ?= cc
# Plain `=`, not `?=`: make predefines FC=f77, which `?=` would not override.
# A command-line `make FC=ifort` still wins over this.
FC     = gfortran
CFLAGS ?= -I. -Wall -O2

# Link the shared lib and bake in an rpath so the examples find it at runtime
# without setting (DY)LD_LIBRARY_PATH.
LDLIBS := -L$(LIBDIR) -ldynars
RPATH  := -Wl,-rpath,$(LIBDIR)

.PHONY: all lib clean
all: c_example f_example

# The C-linkable artifacts (libdynars.dylib/.so + libdynars.a) only exist when
# the `ffi` feature is on.
lib:
	cd $(ROOT) && cargo build --release --features ffi

c_example: lib example.c dynars.h
	$(CC) $(CFLAGS) example.c -o c_example $(LDLIBS) $(RPATH)

f_example: lib example.f90
	$(FC) -O2 example.f90 -o f_example $(LDLIBS) $(RPATH)

clean:
	rm -f c_example f_example dynars.mod
