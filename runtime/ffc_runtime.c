/* ffc runtime support library.
 *
 * Single source of truth for the ffc runtime. Two consumers
 * read this file:
 *
 *   - src/ffc_runtime_source.f90 embeds it verbatim in the
 *     compiler, and ffc links it into every executable it
 *     emits (issue #565). Regenerate that module with
 *     scripts/generate_runtime_source.sh after every edit;
 *     test_runtime_link_compiler checks the two agree.
 *   - runtime/CMakeLists.txt packages it into the
 *     backend-qualified LIRIC runtime archives (#374), used
 *     by sessions that resolve runtime calls without a
 *     system linker.
 *
 * Every entry point defined here must also be listed in
 * ffc_runtime_link's FFC_RUNTIME_SYMBOLS, and documented in
 * docs/RUNTIME_ABI.md. Lines stay at or below 66 columns so
 * the generated Fortran embedding fits in 88.
 */

/* Returns 42. The sole purpose is to give a consumer a
 * cheap, unambiguous end-to-end check that the runtime it
 * linked is really present and callable. */
int _ffc_runtime_probe(void) {
    return 42;
}
