/* The runtime states the language level and the feature set it
 * needs, before any include, instead of inheriting whatever the
 * invoking driver happens to default to.
 *
 * Three drivers compile this file: the system C compiler that
 * links every emitted executable, clang in
 * runtime/CMakeLists.txt, and any packager's. random() and
 * srandom() are POSIX, not ISO C, so without this a strict
 * driver reaches them only through an implicit declaration: a
 * warning on a lenient toolchain, a hard error on a strict one
 * or on a C23 default. Declaring the macro makes every driver
 * agree. */
#define _XOPEN_SOURCE 700

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
#include <stdlib.h>
#include <string.h>

#define FFC_PATH_MAX 4096
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

/* Units the process starts with: 5 is standard input, 6 standard
 * output, and 0 standard error. */
static int ffc_unit_standard(int unit) {
    return unit == 0 || unit == 5 || unit == 6;
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
                   int path_len, const char *status) {
    FILE *fp;
    char name[FFC_PATH_MAX];
    int n;
    if (!ffc_unit_valid(unit)) {
        return ffc_unit_fail(FFC_IOSTAT_BADUNIT);
    }
    if (ffc_units[unit].connected) {
        return ffc_unit_fail(FFC_IOSTAT_INUSE);
    }
    /* FILE= carries the whole declared width of a Fortran
     * character value, so trailing blanks are padding and never
     * part of the file name. */
    n = path == NULL ? 0 : path_len;
    if (n < 0) {
        n = 0;
    }
    if (n > (int)sizeof(name) - 1) {
        n = (int)sizeof(name) - 1;
    }
    while (n > 0 &&
           (path[n - 1] == ' ' || path[n - 1] == '\0')) {
        n--;
    }
    if (n > 0) {
        memcpy(name, path, (size_t)n);
    }
    name[n] = '\0';
    /* OPEN on a preconnected unit without FILE= reconfigures
     * that connection rather than replacing it, so unit 6 keeps
     * writing to standard output. */
    if (n == 0 && ffc_unit_standard(unit)) {
        ffc_unit_last_status = 0;
        return 0;
    }
    if (n == 0 || ffc_streq(status, "scratch")) {
        fp = tmpfile();
    } else {
        fp = ffc_unit_fopen(name, status);
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
    /* Preconnected units. Fortran connects 5 to standard input
     * and 6 to standard output; gfortran also connects 0 to
     * standard error. They are never opened as fort.<N>, and
     * never closed. */
    if (unit == 5) {
        ffc_unit_last_status = 0;
        return stdin;
    }
    if (unit == 6) {
        ffc_unit_last_status = 0;
        return stdout;
    }
    if (unit == 0) {
        ffc_unit_last_status = 0;
        return stderr;
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

/* ---- RANDOM_SEED (issue #588) ---------------------------- */

/* RANDOM_NUMBER draws from glibc's random(), whose state is
 * seeded by srandom(). That state is one integer, so the
 * seed array RANDOM_SEED works with has size 1 and only its
 * first element is read or written. The last seed put is
 * kept here because srandom() offers no way to read it back,
 * and RANDOM_SEED(GET=) must report it. */

static int ffc_random_seed_state = 1;

/* RANDOM_SEED(SIZE=n): the seed array size, always 1. */
int _ffc_random_seed_size(void) {
    return 1;
}

/* RANDOM_SEED(PUT=seed): restart the generator from seed[0],
 * so an identical PUT replays an identical sequence. */
void _ffc_random_seed_put(const int *seed) {
    if (seed == NULL) {
        return;
    }
    ffc_random_seed_state = seed[0];
    srandom((unsigned int) seed[0]);
}

/* RANDOM_SEED(GET=seed): report the current seed. */
void _ffc_random_seed_get(int *seed) {
    if (seed == NULL) {
        return;
    }
    seed[0] = ffc_random_seed_state;
}

/* RANDOM_SEED() with no arguments: reset to the processor's
 * default seed, which is glibc's own initial random() state
 * (srandom(1)). Repeatable across runs, as F2018 permits. */
void _ffc_random_seed_default(void) {
    ffc_random_seed_state = 1;
    srandom(1u);
}

/* ---- Scalar formatted output (issue #423) ----------------- */

/* One entry point per scalar type, so the type tag is resolved at
 * compile time and the call is not variadic: a non-variadic ABI
 * is the same on every target, while a variadic one is not.
 *
 * The compiler supplies the unit, the C conversion descriptor it
 * derived from the Fortran edit descriptor, and the value. The
 * runtime owns the stream lookup, the conversion, and the status.
 * Output bytes are unchanged from the printf calls these replace.
 *
 * Each returns 0 on success, or the unit status when the unit is
 * unusable, or FFC_IOSTAT_WRITE when the conversion fails. */

#define FFC_IOSTAT_WRITE 5006

static int ffc_write_failed(int written) {
    if (written < 0) {
        ffc_unit_last_status = FFC_IOSTAT_WRITE;
        return FFC_IOSTAT_WRITE;
    }
    ffc_unit_last_status = 0;
    return 0;
}

int _ffc_write_i32(int unit, const char *fmt, int value) {
    FILE *fp = _ffc_unit_file(unit);
    if (fp == NULL) {
        return ffc_unit_last_status;
    }
    return ffc_write_failed(fprintf(fp, fmt, value));
}

int _ffc_write_i64(int unit, const char *fmt, long long value) {
    FILE *fp = _ffc_unit_file(unit);
    if (fp == NULL) {
        return ffc_unit_last_status;
    }
    return ffc_write_failed(fprintf(fp, fmt, value));
}

int _ffc_write_f64(int unit, const char *fmt, double value) {
    FILE *fp = _ffc_unit_file(unit);
    if (fp == NULL) {
        return ffc_unit_last_status;
    }
    return ffc_write_failed(fprintf(fp, fmt, value));
}

int _ffc_write_str(int unit, const char *fmt, const char *value) {
    FILE *fp = _ffc_unit_file(unit);
    if (fp == NULL) {
        return ffc_unit_last_status;
    }
    return ffc_write_failed(fprintf(fp, fmt, value));
}

/* Literal record text: the separating blank and the record
 * terminator carry no value to convert. */
int _ffc_write_text(int unit, const char *text) {
    FILE *fp = _ffc_unit_file(unit);
    if (fp == NULL) {
        return ffc_unit_last_status;
    }
    return ffc_write_failed(fputs(text, fp));
}

/* ---- IOSTAT and IOMSG (issue #427) ------------------------ */

/* Fortran reports I/O status through IOSTAT= and IOMSG=. The
 * classes are fixed by the standard and by what programs test
 * for:
 *
 *   0    success
 *   -1   end of file      (gfortran's IOSTAT_END)
 *   -2   end of record    (gfortran's IOSTAT_EOR)
 *   > 0  an error
 *
 * The runtime already records an internal status per unit
 * operation. These map that to the Fortran class and to the
 * message text, in one place, so every statement reports the
 * same value for the same condition. */

#define FFC_IOSTAT_END (-1)
#define FFC_IOSTAT_EOR (-2)

/* Fortran status of the most recent I/O operation. Internal
 * codes are already positive error numbers, and the end-of-file
 * and end-of-record classes are stored as themselves, so this
 * is the recorded status unchanged. It exists so lowering has
 * one name to call rather than knowing the mapping. */
int _ffc_iostat(void) {
    return ffc_unit_last_status;
}

/* Records an end-of-file condition, so a READ that hits it
 * reports the same -1 every other statement reports. */
void _ffc_iostat_set_end(void) {
    ffc_unit_last_status = FFC_IOSTAT_END;
}

void _ffc_iostat_clear(void) {
    ffc_unit_last_status = 0;
}

static const char *ffc_iostat_text(int status) {
    switch (status) {
    case 0:
        return "";
    case FFC_IOSTAT_END:
        return "End of file";
    case FFC_IOSTAT_EOR:
        return "End of record";
    case FFC_IOSTAT_BADUNIT:
        return "Unit number is out of range";
    case FFC_IOSTAT_NOUNIT:
        return "Unit is not connected";
    case FFC_IOSTAT_INUSE:
        return "Unit is already connected";
    case FFC_IOSTAT_OPEN:
        return "Cannot open file";
    case FFC_IOSTAT_NOSPACE:
        return "No free unit for NEWUNIT";
    case FFC_IOSTAT_WRITE:
        return "Write failed";
    default:
        return "I/O error";
    }
}

/* IOMSG= for the most recent operation, written with Fortran
 * character assignment semantics: the text is truncated to len
 * and the remainder is blank filled, never NUL terminated.
 *
 * The standard defines IOMSG only when an error or end-of-file
 * condition occurs. After a successful operation this leaves the
 * variable all blanks rather than untouched, so the destination
 * is always defined and a program never reads whatever the
 * buffer happened to hold.
 *
 * Writes exactly len characters and a terminating NUL, so dest
 * must have room for len + 1: the compiler's character values
 * are NUL-terminated buffers of the declared length. */
void _ffc_iomsg(char *dest, int len) {
    const char *text;
    int i;
    if (dest == NULL || len <= 0) {
        return;
    }
    text = ffc_iostat_text(ffc_unit_last_status);
    for (i = 0; i < len && text[i] != '\0'; i++) {
        dest[i] = text[i];
    }
    for (; i < len; i++) {
        dest[i] = ' ';
    }
    dest[len] = '\0';
}

/* ---- Descriptor storage allocation (issue #428) ----------- */

/* Allocatable arrays and deferred-length characters used to
 * reach malloc() and free() directly from emitted code, which
 * meant every size computation, every overflow check, and every
 * ownership decision was open-coded at each site. These helpers
 * own that instead: the compiler still decides shape and type,
 * the runtime decides whether a size is representable, whether
 * a pointer may be released, and what the status is.
 *
 * Sizes arrive as a separate element count and element size, not
 * as a product, so the multiplication that can overflow happens
 * here, once, where it is checked. A count of zero is a valid
 * request: Fortran allows a zero-sized array, and the result is
 * a non-null pointer that can be released exactly like any
 * other.
 *
 * Status codes are stable, and follow the IOSTAT ranges:
 *
 *   0                        success
 *   FFC_ALLOC_NEGATIVE       negative count or element size
 *   FFC_ALLOC_OVERFLOW       count * element size is not
 *                            representable
 *   FFC_ALLOC_NOMEM          the allocator refused
 *   FFC_ALLOC_DOUBLE_FREE    release of a pointer that is not
 *                            live
 *   FFC_ALLOC_BORROWED       release of storage the descriptor
 *                            does not own
 */

#include <stdint.h>
#include <string.h>

#define FFC_ALLOC_NEGATIVE 6001
#define FFC_ALLOC_OVERFLOW 6002
#define FFC_ALLOC_NOMEM 6003
#define FFC_ALLOC_DOUBLE_FREE 6004
#define FFC_ALLOC_BORROWED 6005

static int ffc_alloc_last_status = 0;

/* Live allocations handed out by _ffc_alloc, so releasing a
 * pointer twice is reported instead of corrupting the heap.
 * Open addressing, power-of-two capacity, grown before it is
 * half full. Tombstones are not needed: a removed entry is
 * refilled by rehashing its cluster. */
static void **ffc_live;
static size_t ffc_live_cap;
static size_t ffc_live_count;

static size_t ffc_live_slot(void **table, size_t cap, void *p) {
    size_t mask = cap - 1;
    size_t i = (size_t)((uintptr_t)p >> 4) & mask;
    while (table[i] != NULL && table[i] != p) {
        i = (i + 1) & mask;
    }
    return i;
}

static int ffc_live_grow(void) {
    size_t new_cap = ffc_live_cap ? ffc_live_cap * 2 : 64;
    void **fresh = calloc(new_cap, sizeof(*fresh));
    size_t i;
    if (fresh == NULL) {
        return -1;
    }
    for (i = 0; i < ffc_live_cap; i++) {
        if (ffc_live[i] != NULL) {
            fresh[ffc_live_slot(fresh, new_cap, ffc_live[i])]
                = ffc_live[i];
        }
    }
    free(ffc_live);
    ffc_live = fresh;
    ffc_live_cap = new_cap;
    return 0;
}

static int ffc_live_add(void *p) {
    if (ffc_live_count * 2 + 1 >= ffc_live_cap) {
        if (ffc_live_grow() != 0) {
            return -1;
        }
    }
    ffc_live[ffc_live_slot(ffc_live, ffc_live_cap, p)] = p;
    ffc_live_count++;
    return 0;
}

/* Removes p and rehashes the rest of its cluster, so lookups
 * that probed past p still find their entries. */
static int ffc_live_remove(void *p) {
    size_t i, j, mask;
    if (ffc_live_cap == 0) {
        return 0;
    }
    mask = ffc_live_cap - 1;
    i = ffc_live_slot(ffc_live, ffc_live_cap, p);
    if (ffc_live[i] != p) {
        return 0;
    }
    ffc_live[i] = NULL;
    ffc_live_count--;
    j = (i + 1) & mask;
    while (ffc_live[j] != NULL) {
        void *moved = ffc_live[j];
        ffc_live[j] = NULL;
        ffc_live_count--;
        ffc_live_add(moved);
        j = (j + 1) & mask;
    }
    return 1;
}

/* Status of the most recent allocation operation. */
int _ffc_alloc_status(void) {
    return ffc_alloc_last_status;
}

/* count elements of elem_size bytes each. Returns NULL and sets
 * the status on any rejected request. */
void *_ffc_alloc(long long count, long long elem_size) {
    size_t bytes;
    void *p;
    if (count < 0 || elem_size < 0) {
        ffc_alloc_last_status = FFC_ALLOC_NEGATIVE;
        return NULL;
    }
    if (elem_size != 0 &&
        count > (long long)(SIZE_MAX / 2)
                    / elem_size) {
        ffc_alloc_last_status = FFC_ALLOC_OVERFLOW;
        return NULL;
    }
    bytes = (size_t)(count * elem_size);
    /* A zero-sized array still needs a releasable pointer. */
    p = malloc(bytes != 0 ? bytes : 1);
    if (p == NULL) {
        ffc_alloc_last_status = FFC_ALLOC_NOMEM;
        return NULL;
    }
    if (ffc_live_add(p) != 0) {
        free(p);
        ffc_alloc_last_status = FFC_ALLOC_NOMEM;
        return NULL;
    }
    ffc_alloc_last_status = 0;
    return p;
}

/* Like _ffc_alloc, with the storage zeroed. An allocatable
 * array of a derived element type needs this: every element's
 * inline component descriptors must start null. */
void *_ffc_calloc(long long count, long long elem_size) {
    void *p = _ffc_alloc(count, elem_size);
    if (p == NULL) {
        return NULL;
    }
    if (count > 0 && elem_size > 0) {
        memset(p, 0, (size_t)(count * elem_size));
    }
    return p;
}

/* Resizes an allocation, keeping min(old, new) bytes. old may
 * be NULL, which makes this a plain allocation. On failure the
 * old pointer is still live and unchanged. */
void *_ffc_realloc(void *old, long long count,
                   long long elem_size) {
    size_t bytes;
    void *p;
    if (old == NULL) {
        return _ffc_alloc(count, elem_size);
    }
    if (count < 0 || elem_size < 0) {
        ffc_alloc_last_status = FFC_ALLOC_NEGATIVE;
        return NULL;
    }
    if (elem_size != 0 &&
        count > (long long)(SIZE_MAX / 2)
                    / elem_size) {
        ffc_alloc_last_status = FFC_ALLOC_OVERFLOW;
        return NULL;
    }
    /* Drop the old key before realloc, which may free it: a
     * freed pointer must not be read again, even as a hash
     * key. A failed realloc leaves it valid, so it goes back. */
    if (!ffc_live_remove(old)) {
        ffc_alloc_last_status = FFC_ALLOC_DOUBLE_FREE;
        return NULL;
    }
    bytes = (size_t)(count * elem_size);
    p = realloc(old, bytes != 0 ? bytes : 1);
    if (p == NULL) {
        ffc_live_add(old);
        ffc_alloc_last_status = FFC_ALLOC_NOMEM;
        return NULL;
    }
    if (ffc_live_add(p) != 0) {
        ffc_alloc_last_status = FFC_ALLOC_NOMEM;
        return NULL;
    }
    ffc_alloc_last_status = 0;
    return p;
}

/* Releases storage. owns is the descriptor's ownership flag: a
 * borrowed descriptor, such as a section view or a dummy
 * argument, never frees, and says so rather than doing nothing
 * silently.
 *
 * Releasing a null pointer succeeds, matching Fortran's
 * deallocate of an unallocated variable and free(NULL). */
int _ffc_dealloc(void *p, int owns) {
    if (p == NULL) {
        ffc_alloc_last_status = 0;
        return 0;
    }
    if (!owns) {
        ffc_alloc_last_status = FFC_ALLOC_BORROWED;
        return FFC_ALLOC_BORROWED;
    }
    if (!ffc_live_remove(p)) {
        ffc_alloc_last_status = FFC_ALLOC_DOUBLE_FREE;
        return FFC_ALLOC_DOUBLE_FREE;
    }
    free(p);
    ffc_alloc_last_status = 0;
    return 0;
}
