# ffc

ffc is a compiler for Fortran and Lazy Fortran (`.lf`) source. It parses and
type-checks programs with [fortfront](https://github.com/lazy-fortran/fortfront),
lowers the typed AST in Fortran, and emits native object files and executables
through the C API of the [LIRIC](https://github.com/lazy-fortran/liric) code
generator.

Status: research compiler under active development. Only the constructs listed
in the support contract are accepted; everything else is refused with a
diagnostic.

```text
Fortran / Lazy Fortran source
  -> fortfront typed AST and diagnostics
  -> ffc lowering and runtime ABI
  -> LIRIC C API (via ISO_C_BINDING)
  -> object file / executable
```

fortfront stays backend-neutral; ffc owns lowering, the runtime ABI, the LIRIC
bindings and object and executable emission.

## Features

- Separate compilation of programs and modules (`-c`, `-I`, linking `.o`
  inputs), with module interfaces exported in `.fmod` files.
- Lazy Fortran: `.lf` files, and bare fragments in `.f90` files, are compiled
  with type inference.
- A growing subset of Fortran 2003-2018: intrinsic and derived types,
  allocatables, pointers, array intrinsics, modules and submodules, limited
  polymorphism, formatted and file I/O.
- Diagnostics as text or JSON (`--json`).
- A conformance runner that compiles external test corpora and compares their
  output with gfortran.

The authoritative list of supported constructs and tracked gaps is
[docs/SUPPORT_CONTRACT.md](docs/SUPPORT_CONTRACT.md).

## Build

ffc needs a Fortran compiler, [fpm](https://fpm.fortran-lang.org) (or
[fo](https://github.com/lazy-fortran/fo)), the LIRIC static library and a
fortfront checkout next to the ffc checkout (`fpm.toml` refers to
`../fortfront`):

```bash
git clone https://github.com/lazy-fortran/fortfront
git clone https://github.com/lazy-fortran/liric
git clone https://github.com/lazy-fortran/ffc
cmake -S liric -B liric/build -G Ninja && cmake --build liric/build
cd ffc
export LIBRARY_PATH=../liric/build
fpm build
fpm test
```

## Usage

```bash
printf 'program main\n  print *, 6*7\nend program main\n' > hello.f90
fpm run ffc -- hello.f90 -o hello
./hello
```

Options of the `ffc` binary:

```text
ffc <input.f90> [options]
  -o <file>      output file
  -c             emit an object file
  -I <dir>       add a module/include search directory
  <file>.o       link an object file (its directory is searched for .fmod)
  --backend <b>  code generation backend: isel, copy-patch or llvm
  --json         emit diagnostics as JSON
```

## Testing and conformance

`fpm test` (or `fo test`) runs the behavioural test suite. External corpora
(LFortran and the gfortran testsuite) are fetched with
`scripts/fetch_corpora.sh` and checked with `scripts/conformance_check.sh`; see
[docs/CONFORMANCE.md](docs/CONFORMANCE.md).

## Documentation

- [docs/SUPPORT_CONTRACT.md](docs/SUPPORT_CONTRACT.md): supported constructs and gaps
- [docs/FEATURE_NOTES.md](docs/FEATURE_NOTES.md): detailed implementation notes by topic
- [docs/RUNTIME_ABI.md](docs/RUNTIME_ABI.md),
  [docs/ARRAY_DESCRIPTOR_ABI.md](docs/ARRAY_DESCRIPTOR_ABI.md),
  [docs/CHARACTER_DESCRIPTOR_ABI.md](docs/CHARACTER_DESCRIPTOR_ABI.md): runtime and data layout
- [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md): source layout, conventions and adding features
- [docs/CONFORMANCE.md](docs/CONFORMANCE.md): conformance runner
- [docs/F2023_DELTA.md](docs/F2023_DELTA.md): Fortran 2023 coverage audit

## Related projects

- [fortfront](https://github.com/lazy-fortran/fortfront): frontend and typed AST
- [LIRIC](https://github.com/lazy-fortran/liric): code generation backend
- [standard](https://github.com/lazy-fortran/standard): the Lazy Fortran language definition
- [fo](https://github.com/lazy-fortran/fo): build driver; `fo build --native` uses ffc

## License

MIT, see [LICENSE](LICENSE).
