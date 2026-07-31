/* ffc runtime support library.
 *
 * This translation unit is compiled to LLVM bitcode and packaged into one
 * backend-qualified LIRIC runtime archive per target/backend pair
 * (see runtime/CMakeLists.txt and docs/RUNTIME_ABI.md).
 *
 * It currently carries only the probe symbol that lets a consumer confirm it
 * loaded a real archive. Runtime entry points for file units, formatted
 * output, IOSTAT/IOMSG, and descriptor allocation are added by their own
 * issues; this issue only establishes the artifact.
 */

/* Returns 42. The sole purpose is to give a loader a cheap, unambiguous
 * end-to-end check that the archive it selected actually resolves. */
int _ffc_runtime_probe(void) {
    return 42;
}
