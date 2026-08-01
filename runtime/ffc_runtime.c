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

/* ---- File units (issue #396) ----------------------------- */

/* The runtime owns file-unit state: which units are connected,
 * the FILE* behind each, and the status of the last operation.
 * Before #396 the compiler emitted one stack slot per unit inside
 * the function that opened it, so unit state was scoped to a
 * lowered function and keyed by a compile-time constant. Here it
 * is per process and keyed by the unit number the program
 * computes at run time, which is what Fortran describes.
 *
 * Status codes are stable and are the values IOSTAT= reports:
 *
 *   0                  success
 *   FFC_IOSTAT_BADUNIT unit number outside the supported range
 *   FFC_IOSTAT_NOUNIT  operation on an unconnected unit
 *   FFC_IOSTAT_INUSE   OPEN on an already connected unit
 *   FFC_IOSTAT_OPEN    the file could not be opened
 *   FFC_IOSTAT_NOSPACE no free unit left for NEWUNIT
 */

#include <stdio.h>

#define FFC_UNIT_MIN 0
#define FFC_UNIT_MAX 2048
#define FFC_NEWUNIT_FIRST 1000

#define FFC_IOSTAT_BADUNIT 5001
#define FFC_IOSTAT_NOUNIT 5002
#define FFC_IOSTAT_INUSE 5003
#define FFC_IOSTAT_OPEN 5004
#define FFC_IOSTAT_NOSPACE 5005

struct ffc_unit {
    FILE *fp;
    int connected;
};

static struct ffc_unit ffc_units[FFC_UNIT_MAX + 1];
static int ffc_unit_last_status = 0;

static int ffc_unit_valid(int unit) {
    return unit >= FFC_UNIT_MIN && unit <= FFC_UNIT_MAX;
}

/* Case-sensitive compare; the compiler lowers STATUS= values in
 * lower case. */
static int ffc_streq(const char *a, const char *b) {
    if (a == NULL || b == NULL) {
        return 0;
    }
    while (*a != '\0' && *a == *b) {
        a++;
        b++;
    }
    return *a == *b;
}

static int ffc_unit_fail(int status) {
    ffc_unit_last_status = status;
    return status;
}

/* Status of the most recent unit operation. */
int _ffc_unit_status(void) {
    return ffc_unit_last_status;
}

/* Lowest free unit at or above FFC_NEWUNIT_FIRST, which is above
 * anything a program is expected to name explicitly. Returns -1
 * on exhaustion, so a caller can store it and let the operation
 * that follows fail on a bad unit number. */
int _ffc_unit_newunit(void) {
    int unit;
    for (unit = FFC_NEWUNIT_FIRST; unit <= FFC_UNIT_MAX; unit++) {
        if (!ffc_units[unit].connected) {
            ffc_unit_last_status = 0;
            return unit;
        }
    }
    ffc_unit_fail(FFC_IOSTAT_NOSPACE);
    return -1;
}

/* Opens the file for a Fortran STATUS= value. Units are readable
 * and writable, so every mode is an update mode: a unit can be
 * written, rewound, and read back. STATUS='UNKNOWN' and an absent
 * STATUS keep an existing file's contents and create the file
 * otherwise, which is what gfortran does; that is why it probes
 * "r+" before falling back to "w+" rather than truncating. */
static FILE *ffc_unit_fopen(const char *path,
                            const char *status) {
    FILE *fp;
    if (ffc_streq(status, "old")) {
        return fopen(path, "r+");
    }
    if (ffc_streq(status, "new") ||
        ffc_streq(status, "replace")) {
        return fopen(path, "w+");
    }
    fp = fopen(path, "r+");
    if (fp != NULL) {
        return fp;
    }
    return fopen(path, "w+");
}

/* Connects unit to path with the given Fortran STATUS= value. A
 * null or empty path, or STATUS='SCRATCH', connects a temporary
 * file that disappears when the unit is closed.
 *
 * Connecting a unit that is already connected is an error rather
 * than a silent reconnection, so a leaked unit surfaces where it
 * happens. */
int _ffc_unit_open(int unit, const char *path,
                   const char *status) {
    FILE *fp;
    if (!ffc_unit_valid(unit)) {
        return ffc_unit_fail(FFC_IOSTAT_BADUNIT);
    }
    if (ffc_units[unit].connected) {
        return ffc_unit_fail(FFC_IOSTAT_INUSE);
    }
    if (path == NULL || path[0] == '\0' ||
        ffc_streq(status, "scratch")) {
        fp = tmpfile();
    } else {
        fp = ffc_unit_fopen(path, status);
    }
    if (fp == NULL) {
        return ffc_unit_fail(FFC_IOSTAT_OPEN);
    }
    ffc_units[unit].fp = fp;
    ffc_units[unit].connected = 1;
    ffc_unit_last_status = 0;
    return 0;
}

/* Whether the unit is currently connected. */
int _ffc_unit_is_open(int unit) {
    if (!ffc_unit_valid(unit)) {
        return 0;
    }
    return ffc_units[unit].connected ? 1 : 0;
}

/* The FILE* behind a unit, connecting a numeric unit to fort.<N>
 * on first use the way an unopened preconnected unit behaves.
 * Returns NULL only when the unit is unusable, recording why. */
FILE *_ffc_unit_file(int unit) {
    char name[32];
    FILE *fp;
    if (!ffc_unit_valid(unit)) {
        ffc_unit_fail(FFC_IOSTAT_BADUNIT);
        return NULL;
    }
    if (ffc_units[unit].connected) {
        ffc_unit_last_status = 0;
        return ffc_units[unit].fp;
    }
    snprintf(name, sizeof name, "fort.%d", unit);
    fp = fopen(name, "w+");
    if (fp == NULL) {
        ffc_unit_fail(FFC_IOSTAT_OPEN);
        return NULL;
    }
    ffc_units[unit].fp = fp;
    ffc_units[unit].connected = 1;
    ffc_unit_last_status = 0;
    return fp;
}

/* Repositions the unit to its first record. */
int _ffc_unit_rewind(int unit) {
    FILE *fp = _ffc_unit_file(unit);
    if (fp == NULL) {
        return ffc_unit_last_status;
    }
    rewind(fp);
    ffc_unit_last_status = 0;
    return 0;
}

/* Disconnects the unit. CLOSE on a unit that is not connected is
 * not an error in Fortran, so it reports success and leaves the
 * unit free; only a bad unit number fails. */
int _ffc_unit_close(int unit) {
    if (!ffc_unit_valid(unit)) {
        return ffc_unit_fail(FFC_IOSTAT_BADUNIT);
    }
    if (ffc_units[unit].connected) {
        fclose(ffc_units[unit].fp);
        ffc_units[unit].fp = NULL;
        ffc_units[unit].connected = 0;
    }
    ffc_unit_last_status = 0;
    return 0;
}
