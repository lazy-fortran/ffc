! GENERATED FILE - DO NOT EDIT.
!
! Regenerate with scripts/generate_runtime_source.sh after editing
! runtime/ffc_runtime.c, which is the single source of truth for the ffc
! runtime. test_runtime_link_compiler fails when this copy drifts.
!
! ffc links its runtime into every executable it emits (issue #565), so the
! runtime source ships inside the compiler binary. That makes a
! compiler/runtime version mismatch impossible by construction: there is no
! separately installed artifact to go missing or go stale.
module ffc_runtime_source
    implicit none
    private

    public :: ffc_runtime_source_text

contains

    ! The verbatim contents of runtime/ffc_runtime.c, newline-terminated.
    subroutine ffc_runtime_source_text(text)
        character(len=:), allocatable, intent(out) :: text
        character(len=1), parameter :: NL = new_line('a')

        text = ''
        text = text// &
            '/* The runtime states the language level and the feature set it'//NL
        text = text// &
            ' * needs, before any include, instead of inheriting whatever the'//NL
        text = text// &
            ' * invoking driver happens to default to.'//NL
        text = text// &
            ' *'//NL
        text = text// &
            ' * Three drivers compile this file: the system C compiler that'//NL
        text = text// &
            ' * links every emitted executable, clang in'//NL
        text = text// &
            ' * runtime/CMakeLists.txt, and any packager''s. random() and'//NL
        text = text// &
            ' * srandom() are POSIX, not ISO C, so without this a strict'//NL
        text = text// &
            ' * driver reaches them only through an implicit declaration: a'//NL
        text = text// &
            ' * warning on a lenient toolchain, a hard error on a strict one'//NL
        text = text// &
            ' * or on a C23 default. Declaring the macro makes every driver'//NL
        text = text// &
            ' * agree. */'//NL
        text = text// &
            '#define _XOPEN_SOURCE 700'//NL
        text = text//NL
        text = text// &
            '/* ffc runtime support library.'//NL
        text = text// &
            ' *'//NL
        text = text// &
            ' * Single source of truth for the ffc runtime. Two consumers'//NL
        text = text// &
            ' * read this file:'//NL
        text = text// &
            ' *'//NL
        text = text// &
            ' *   - src/ffc_runtime_source.f90 embeds it verbatim in the'//NL
        text = text// &
            ' *     compiler, and ffc links it into every executable it'//NL
        text = text// &
            ' *     emits (issue #565). Regenerate that module with'//NL
        text = text// &
            ' *     scripts/generate_runtime_source.sh after every edit;'//NL
        text = text// &
            ' *     test_runtime_link_compiler checks the two agree.'//NL
        text = text// &
            ' *   - runtime/CMakeLists.txt packages it into the'//NL
        text = text// &
            ' *     backend-qualified LIRIC runtime archives (#374), used'//NL
        text = text// &
            ' *     by sessions that resolve runtime calls without a'//NL
        text = text// &
            ' *     system linker.'//NL
        text = text// &
            ' *'//NL
        text = text// &
            ' * Every entry point defined here must also be listed in'//NL
        text = text// &
            ' * ffc_runtime_link''s FFC_RUNTIME_SYMBOLS, and documented in'//NL
        text = text// &
            ' * docs/RUNTIME_ABI.md. Lines stay at or below 66 columns so'//NL
        text = text// &
            ' * the generated Fortran embedding fits in 88.'//NL
        text = text// &
            ' */'//NL
        text = text//NL
        text = text// &
            '/* Returns 42. The sole purpose is to give a consumer a'//NL
        text = text// &
            ' * cheap, unambiguous end-to-end check that the runtime it'//NL
        text = text// &
            ' * linked is really present and callable. */'//NL
        text = text// &
            'int _ffc_runtime_probe(void) {'//NL
        text = text// &
            '    return 42;'//NL
        text = text// &
            '}'//NL
        text = text//NL
        text = text// &
            '/* ---- File units (issue #396) ----------------------------- */'//NL
        text = text//NL
        text = text// &
            '/* The runtime owns file-unit state: which units are connected,'//NL
        text = text// &
            ' * the FILE* behind each, and the status of the last operation.'//NL
        text = text// &
            ' * Before #396 the compiler emitted one stack slot per unit inside'//NL
        text = text// &
            ' * the function that opened it, so unit state was scoped to a'//NL
        text = text// &
            ' * lowered function and keyed by a compile-time constant. Here it'//NL
        text = text// &
            ' * is per process and keyed by the unit number the program'//NL
        text = text// &
            ' * computes at run time, which is what Fortran describes.'//NL
        text = text// &
            ' *'//NL
        text = text// &
            ' * Status codes are stable and are the values IOSTAT= reports:'//NL
        text = text// &
            ' *'//NL
        text = text// &
            ' *   0                  success'//NL
        text = text// &
            ' *   FFC_IOSTAT_BADUNIT unit number outside the supported range'//NL
        text = text// &
            ' *   FFC_IOSTAT_NOUNIT  operation on an unconnected unit'//NL
        text = text// &
            ' *   FFC_IOSTAT_INUSE   OPEN on an already connected unit'//NL
        text = text// &
            ' *   FFC_IOSTAT_OPEN    the file could not be opened'//NL
        text = text// &
            ' *   FFC_IOSTAT_NOSPACE no free unit left for NEWUNIT'//NL
        text = text// &
            ' */'//NL
        text = text//NL
        text = text// &
            '#include <stdio.h>'//NL
        text = text// &
            '#include <stdlib.h>'//NL
        text = text// &
            '#include <string.h>'//NL
        text = text// &
            '#include <ctype.h>'//NL
        text = text// &
            '#include <sys/stat.h>'//NL
        text = text//NL
        text = text// &
            '#define FFC_PATH_MAX 4096'//NL
        text = text// &
            '#define FFC_UNIT_MIN 0'//NL
        text = text// &
            '#define FFC_UNIT_MAX 2048'//NL
        text = text// &
            '#define FFC_NEWUNIT_FIRST 1000'//NL
        text = text//NL
        text = text// &
            '#define FFC_IOSTAT_BADUNIT 5001'//NL
        text = text// &
            '#define FFC_IOSTAT_NOUNIT 5002'//NL
        text = text// &
            '#define FFC_IOSTAT_INUSE 5003'//NL
        text = text// &
            '#define FFC_IOSTAT_OPEN 5004'//NL
        text = text// &
            '#define FFC_IOSTAT_NOSPACE 5005'//NL
        text = text//NL
        text = text// &
            'struct ffc_unit {'//NL
        text = text// &
            '    FILE *fp;'//NL
        text = text// &
            '    int connected;'//NL
        text = text// &
            '};'//NL
        text = text//NL
        text = text// &
            'static struct ffc_unit ffc_units[FFC_UNIT_MAX + 1];'//NL
        text = text// &
            'static int ffc_unit_last_status = 0;'//NL
        text = text//NL
        text = text// &
            'static int ffc_unit_valid(int unit) {'//NL
        text = text// &
            '    return unit >= FFC_UNIT_MIN && unit <= FFC_UNIT_MAX;'//NL
        text = text// &
            '}'//NL
        text = text//NL
        text = text// &
            '/* Compare a Fortran STATUS= value case-insensitively and ignore'//NL
        text = text// &
            ' * fixed-length trailing blanks. */'//NL
        text = text// &
            'static int ffc_streq(const char *a, const char *b) {'//NL
        text = text// &
            '    int i = 0;'//NL
        text = text// &
            '    if (a == NULL || b == NULL) {'//NL
        text = text// &
            '        return 0;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    while (a[i] != ''\0'' && b[i] != ''\0'' &&'//NL
        text = text// &
            '           tolower((unsigned char)a[i]) =='//NL
        text = text// &
            '           tolower((unsigned char)b[i])) {'//NL
        text = text// &
            '        i++;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    while (a[i] == '' '') {'//NL
        text = text// &
            '        i++;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    return a[i] == ''\0'' && b[i] == ''\0'';'//NL
        text = text// &
            '}'//NL
        text = text//NL
        text = text// &
            '/* Units the process starts with: 5 is standard input, 6 standard'//NL
        text = text// &
            ' * output, and 0 standard error. */'//NL
        text = text// &
            'static int ffc_unit_standard(int unit) {'//NL
        text = text// &
            '    return unit == 0 || unit == 5 || unit == 6;'//NL
        text = text// &
            '}'//NL
        text = text//NL
        text = text// &
            'static int ffc_unit_fail(int status) {'//NL
        text = text// &
            '    ffc_unit_last_status = status;'//NL
        text = text// &
            '    return status;'//NL
        text = text// &
            '}'//NL
        text = text//NL
        text = text// &
            '/* Status of the most recent unit operation. */'//NL
        text = text// &
            'int _ffc_unit_status(void) {'//NL
        text = text// &
            '    return ffc_unit_last_status;'//NL
        text = text// &
            '}'//NL
        text = text//NL
        text = text// &
            '/* Lowest free unit at or above FFC_NEWUNIT_FIRST, which is above'//NL
        text = text// &
            ' * anything a program is expected to name explicitly. Returns -1'//NL
        text = text// &
            ' * on exhaustion, so a caller can store it and let the operation'//NL
        text = text// &
            ' * that follows fail on a bad unit number. */'//NL
        text = text// &
            'int _ffc_unit_newunit(void) {'//NL
        text = text// &
            '    int unit;'//NL
        text = text// &
            '    for (unit = FFC_NEWUNIT_FIRST; unit <= FFC_UNIT_MAX; unit++) {'//NL
        text = text// &
            '        if (!ffc_units[unit].connected) {'//NL
        text = text// &
            '            ffc_unit_last_status = 0;'//NL
        text = text// &
            '            return unit;'//NL
        text = text// &
            '        }'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    ffc_unit_fail(FFC_IOSTAT_NOSPACE);'//NL
        text = text// &
            '    return -1;'//NL
        text = text// &
            '}'//NL
        text = text//NL
        text = text// &
            '/* Opens the file for a Fortran STATUS= value. Units are readable'//NL
        text = text// &
            ' * and writable, so every mode is an update mode: a unit can be'//NL
        text = text// &
            ' * written, rewound, and read back. STATUS=''UNKNOWN'' and an absent'//NL
        text = text// &
            ' * STATUS keep an existing file''s contents and create the file'//NL
        text = text// &
            ' * otherwise, which is what gfortran does; that is why it probes'//NL
        text = text// &
            ' * "r+" before falling back to "w+" rather than truncating. */'//NL
        text = text// &
            'static FILE *ffc_unit_fopen(const char *path,'//NL
        text = text// &
            '                            const char *status) {'//NL
        text = text// &
            '    FILE *fp;'//NL
        text = text// &
            '    if (ffc_streq(status, "old")) {'//NL
        text = text// &
            '        return fopen(path, "r+");'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    if (ffc_streq(status, "new") ||'//NL
        text = text// &
            '        ffc_streq(status, "replace")) {'//NL
        text = text// &
            '        return fopen(path, "w+");'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    fp = fopen(path, "r+");'//NL
        text = text// &
            '    if (fp != NULL) {'//NL
        text = text// &
            '        return fp;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    return fopen(path, "w+");'//NL
        text = text// &
            '}'//NL
        text = text//NL
        text = text// &
            '/* Connects unit to path with the given Fortran STATUS= value. A'//NL
        text = text// &
            ' * null or empty path, or STATUS=''SCRATCH'', connects a temporary'//NL
        text = text// &
            ' * file that disappears when the unit is closed.'//NL
        text = text// &
            ' *'//NL
        text = text// &
            ' * Connecting a unit that is already connected is an error rather'//NL
        text = text// &
            ' * than a silent reconnection, so a leaked unit surfaces where it'//NL
        text = text// &
            ' * happens. */'//NL
        text = text// &
            'int _ffc_unit_open(int unit, const char *path,'//NL
        text = text// &
            '                   int path_len, const char *status) {'//NL
        text = text// &
            '    FILE *fp;'//NL
        text = text// &
            '    char name[FFC_PATH_MAX];'//NL
        text = text// &
            '    int n;'//NL
        text = text// &
            '    if (!ffc_unit_valid(unit)) {'//NL
        text = text// &
            '        return ffc_unit_fail(FFC_IOSTAT_BADUNIT);'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    if (ffc_units[unit].connected) {'//NL
        text = text// &
            '        return ffc_unit_fail(FFC_IOSTAT_INUSE);'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    /* FILE= carries the whole declared width of a Fortran'//NL
        text = text// &
            '     * character value, so trailing blanks are padding and never'//NL
        text = text// &
            '     * part of the file name. */'//NL
        text = text// &
            '    n = path == NULL ? 0 : path_len;'//NL
        text = text// &
            '    if (n < 0) {'//NL
        text = text// &
            '        n = 0;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    if (n > (int)sizeof(name) - 1) {'//NL
        text = text// &
            '        n = (int)sizeof(name) - 1;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    while (n > 0 &&'//NL
        text = text// &
            '           (path[n - 1] == '' '' || path[n - 1] == ''\0'')) {'//NL
        text = text// &
            '        n--;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    if (n > 0) {'//NL
        text = text// &
            '        memcpy(name, path, (size_t)n);'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    name[n] = ''\0'';'//NL
        text = text// &
            '    /* OPEN on a preconnected unit without FILE= reconfigures'//NL
        text = text// &
            '     * that connection rather than replacing it, so unit 6 keeps'//NL
        text = text// &
            '     * writing to standard output. */'//NL
        text = text// &
            '    if (n == 0 && ffc_unit_standard(unit)) {'//NL
        text = text// &
            '        ffc_unit_last_status = 0;'//NL
        text = text// &
            '        return 0;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    if (n == 0 || ffc_streq(status, "scratch")) {'//NL
        text = text// &
            '        fp = tmpfile();'//NL
        text = text// &
            '    } else {'//NL
        text = text// &
            '        fp = ffc_unit_fopen(name, status);'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    if (fp == NULL) {'//NL
        text = text// &
            '        return ffc_unit_fail(FFC_IOSTAT_OPEN);'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    ffc_units[unit].fp = fp;'//NL
        text = text// &
            '    ffc_units[unit].connected = 1;'//NL
        text = text// &
            '    ffc_unit_last_status = 0;'//NL
        text = text// &
            '    return 0;'//NL
        text = text// &
            '}'//NL
        text = text//NL
        text = text// &
            '/* Whether the unit is currently connected. */'//NL
        text = text// &
            'int _ffc_unit_is_open(int unit) {'//NL
        text = text// &
            '    if (!ffc_unit_valid(unit)) {'//NL
        text = text// &
            '        return 0;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    return ffc_units[unit].connected ? 1 : 0;'//NL
        text = text// &
            '}'//NL
        text = text//NL
        text = text// &
            '/* The FILE* behind a unit, connecting a numeric unit to fort.<N>'//NL
        text = text// &
            ' * on first use the way an unopened preconnected unit behaves.'//NL
        text = text// &
            ' * Returns NULL only when the unit is unusable, recording why. */'//NL
        text = text// &
            'FILE *_ffc_unit_file(int unit) {'//NL
        text = text// &
            '    char name[32];'//NL
        text = text// &
            '    FILE *fp;'//NL
        text = text// &
            '    if (!ffc_unit_valid(unit)) {'//NL
        text = text// &
            '        ffc_unit_fail(FFC_IOSTAT_BADUNIT);'//NL
        text = text// &
            '        return NULL;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    if (ffc_units[unit].connected) {'//NL
        text = text// &
            '        ffc_unit_last_status = 0;'//NL
        text = text// &
            '        return ffc_units[unit].fp;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    /* Preconnected units. Fortran connects 5 to standard input'//NL
        text = text// &
            '     * and 6 to standard output; gfortran also connects 0 to'//NL
        text = text// &
            '     * standard error. They are never opened as fort.<N>, and'//NL
        text = text// &
            '     * never closed. */'//NL
        text = text// &
            '    if (unit == 5) {'//NL
        text = text// &
            '        ffc_unit_last_status = 0;'//NL
        text = text// &
            '        return stdin;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    if (unit == 6) {'//NL
        text = text// &
            '        ffc_unit_last_status = 0;'//NL
        text = text// &
            '        return stdout;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    if (unit == 0) {'//NL
        text = text// &
            '        ffc_unit_last_status = 0;'//NL
        text = text// &
            '        return stderr;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    snprintf(name, sizeof name, "fort.%d", unit);'//NL
        text = text// &
            '    fp = fopen(name, "w+");'//NL
        text = text// &
            '    if (fp == NULL) {'//NL
        text = text// &
            '        ffc_unit_fail(FFC_IOSTAT_OPEN);'//NL
        text = text// &
            '        return NULL;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    ffc_units[unit].fp = fp;'//NL
        text = text// &
            '    ffc_units[unit].connected = 1;'//NL
        text = text// &
            '    ffc_unit_last_status = 0;'//NL
        text = text// &
            '    return fp;'//NL
        text = text// &
            '}'//NL
        text = text//NL
        text = text// &
            '/* Repositions the unit to its first record. */'//NL
        text = text// &
            'int _ffc_unit_rewind(int unit) {'//NL
        text = text// &
            '    FILE *fp = _ffc_unit_file(unit);'//NL
        text = text// &
            '    if (fp == NULL) {'//NL
        text = text// &
            '        return ffc_unit_last_status;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    rewind(fp);'//NL
        text = text// &
            '    ffc_unit_last_status = 0;'//NL
        text = text// &
            '    return 0;'//NL
        text = text// &
            '}'//NL
        text = text//NL
        text = text// &
            '/* Disconnects the unit. CLOSE on a unit that is not connected is'//NL
        text = text// &
            ' * not an error in Fortran, so it reports success and leaves the'//NL
        text = text// &
            ' * unit free; only a bad unit number fails. */'//NL
        text = text// &
            'int _ffc_unit_close(int unit) {'//NL
        text = text// &
            '    if (!ffc_unit_valid(unit)) {'//NL
        text = text// &
            '        return ffc_unit_fail(FFC_IOSTAT_BADUNIT);'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    if (ffc_units[unit].connected) {'//NL
        text = text// &
            '        fclose(ffc_units[unit].fp);'//NL
        text = text// &
            '        ffc_units[unit].fp = NULL;'//NL
        text = text// &
            '        ffc_units[unit].connected = 0;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    ffc_unit_last_status = 0;'//NL
        text = text// &
            '    return 0;'//NL
        text = text// &
            '}'//NL
        text = text//NL
        text = text// &
            '/* Returns the byte size of a named file, or -1 when the file'//NL
        text = text// &
            ' * cannot be inspected. The compiler maps this to'//NL
        text = text// &
            ' * INQUIRE(SIZE=). */'//NL
        text = text// &
            'long long _ffc_inquire_file_size(const char *path) {'//NL
        text = text// &
            '    struct stat info;'//NL
        text = text// &
            '    if (path == NULL || stat(path, &info) != 0 ||'//NL
        text = text// &
            '        info.st_size < 0) {'//NL
        text = text// &
            '        return -1;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    return (long long) info.st_size;'//NL
        text = text// &
            '}'//NL
        text = text//NL
        text = text// &
            '/* Returns the current size of a connected unit without changing'//NL
        text = text// &
            ' * its position, or -1 when the unit has no seekable stream. */'//NL
        text = text// &
            'long long _ffc_inquire_unit_size(int unit) {'//NL
        text = text// &
            '    FILE *fp;'//NL
        text = text// &
            '    long current;'//NL
        text = text// &
            '    long end;'//NL
        text = text//NL
        text = text// &
            '    if (!ffc_unit_valid(unit) || !ffc_units[unit].connected) {'//NL
        text = text// &
            '        return -1;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    fp = ffc_units[unit].fp;'//NL
        text = text// &
            '    if (fp == NULL || fflush(fp) != 0) {'//NL
        text = text// &
            '        return -1;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    current = ftell(fp);'//NL
        text = text// &
            '    if (current < 0 || fseek(fp, 0L, SEEK_END) != 0) {'//NL
        text = text// &
            '        return -1;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    end = ftell(fp);'//NL
        text = text// &
            '    if (fseek(fp, current, SEEK_SET) != 0 || end < 0) {'//NL
        text = text// &
            '        return -1;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    ffc_unit_last_status = 0;'//NL
        text = text// &
            '    return (long long) end;'//NL
        text = text// &
            '}'//NL
        text = text//NL
        text = text// &
            '/* ---- RANDOM_SEED (issue #588) ---------------------------- */'//NL
        text = text//NL
        text = text// &
            '/* RANDOM_NUMBER draws from glibc''s random(), whose state is'//NL
        text = text// &
            ' * seeded by srandom(). That state is one integer, so the'//NL
        text = text// &
            ' * seed array RANDOM_SEED works with has size 1 and only its'//NL
        text = text// &
            ' * first element is read or written. The last seed put is'//NL
        text = text// &
            ' * kept here because srandom() offers no way to read it back,'//NL
        text = text// &
            ' * and RANDOM_SEED(GET=) must report it. */'//NL
        text = text//NL
        text = text// &
            'static int ffc_random_seed_state = 1;'//NL
        text = text//NL
        text = text// &
            '/* RANDOM_SEED(SIZE=n): the seed array size, always 1. */'//NL
        text = text// &
            'int _ffc_random_seed_size(void) {'//NL
        text = text// &
            '    return 1;'//NL
        text = text// &
            '}'//NL
        text = text//NL
        text = text// &
            '/* RANDOM_SEED(PUT=seed): restart the generator from seed[0],'//NL
        text = text// &
            ' * so an identical PUT replays an identical sequence. */'//NL
        text = text// &
            'void _ffc_random_seed_put(const int *seed) {'//NL
        text = text// &
            '    if (seed == NULL) {'//NL
        text = text// &
            '        return;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    ffc_random_seed_state = seed[0];'//NL
        text = text// &
            '    srandom((unsigned int) seed[0]);'//NL
        text = text// &
            '}'//NL
        text = text//NL
        text = text// &
            '/* RANDOM_SEED(GET=seed): report the current seed. */'//NL
        text = text// &
            'void _ffc_random_seed_get(int *seed) {'//NL
        text = text// &
            '    if (seed == NULL) {'//NL
        text = text// &
            '        return;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    seed[0] = ffc_random_seed_state;'//NL
        text = text// &
            '}'//NL
        text = text//NL
        text = text// &
            '/* RANDOM_SEED() with no arguments: reset to the processor''s'//NL
        text = text// &
            ' * default seed, which is glibc''s own initial random() state'//NL
        text = text// &
            ' * (srandom(1)). Repeatable across runs, as F2018 permits. */'//NL
        text = text// &
            'void _ffc_random_seed_default(void) {'//NL
        text = text// &
            '    ffc_random_seed_state = 1;'//NL
        text = text// &
            '    srandom(1u);'//NL
        text = text// &
            '}'//NL
        text = text//NL
        text = text// &
            '/* ---- Scalar formatted output (issue #423) ----------------- */'//NL
        text = text//NL
        text = text// &
            '/* One entry point per scalar type, so the type tag is resolved at'//NL
        text = text// &
            ' * compile time and the call is not variadic: a non-variadic ABI'//NL
        text = text// &
            ' * is the same on every target, while a variadic one is not.'//NL
        text = text// &
            ' *'//NL
        text = text// &
            ' * The compiler supplies the unit, the C conversion descriptor it'//NL
        text = text// &
            ' * derived from the Fortran edit descriptor, and the value. The'//NL
        text = text// &
            ' * runtime owns the stream lookup, the conversion, and the status.'//NL
        text = text// &
            ' * Output bytes are unchanged from the printf calls these replace.'//NL
        text = text// &
            ' *'//NL
        text = text// &
            ' * Each returns 0 on success, or the unit status when the unit is'//NL
        text = text// &
            ' * unusable, or FFC_IOSTAT_WRITE when the conversion fails. */'//NL
        text = text//NL
        text = text// &
            '#define FFC_IOSTAT_WRITE 5006'//NL
        text = text//NL
        text = text// &
            'static int ffc_write_failed(int written) {'//NL
        text = text// &
            '    if (written < 0) {'//NL
        text = text// &
            '        ffc_unit_last_status = FFC_IOSTAT_WRITE;'//NL
        text = text// &
            '        return FFC_IOSTAT_WRITE;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    ffc_unit_last_status = 0;'//NL
        text = text// &
            '    return 0;'//NL
        text = text// &
            '}'//NL
        text = text//NL
        text = text// &
            'int _ffc_write_i32(int unit, const char *fmt, int value) {'//NL
        text = text// &
            '    FILE *fp = _ffc_unit_file(unit);'//NL
        text = text// &
            '    if (fp == NULL) {'//NL
        text = text// &
            '        return ffc_unit_last_status;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    return ffc_write_failed(fprintf(fp, fmt, value));'//NL
        text = text// &
            '}'//NL
        text = text//NL
        text = text// &
            'int _ffc_write_i64(int unit, const char *fmt, long long value) {'//NL
        text = text// &
            '    FILE *fp = _ffc_unit_file(unit);'//NL
        text = text// &
            '    if (fp == NULL) {'//NL
        text = text// &
            '        return ffc_unit_last_status;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    return ffc_write_failed(fprintf(fp, fmt, value));'//NL
        text = text// &
            '}'//NL
        text = text//NL
        text = text// &
            'int _ffc_write_f64(int unit, const char *fmt, double value) {'//NL
        text = text// &
            '    FILE *fp = _ffc_unit_file(unit);'//NL
        text = text// &
            '    if (fp == NULL) {'//NL
        text = text// &
            '        return ffc_unit_last_status;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    return ffc_write_failed(fprintf(fp, fmt, value));'//NL
        text = text// &
            '}'//NL
        text = text//NL
        text = text// &
            'int _ffc_write_str(int unit, const char *fmt, const char *value) {'//NL
        text = text// &
            '    FILE *fp = _ffc_unit_file(unit);'//NL
        text = text// &
            '    if (fp == NULL) {'//NL
        text = text// &
            '        return ffc_unit_last_status;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    return ffc_write_failed(fprintf(fp, fmt, value));'//NL
        text = text// &
            '}'//NL
        text = text//NL
        text = text// &
            '/* Literal record text: the separating blank and the record'//NL
        text = text// &
            ' * terminator carry no value to convert. */'//NL
        text = text// &
            'int _ffc_write_text(int unit, const char *text) {'//NL
        text = text// &
            '    FILE *fp = _ffc_unit_file(unit);'//NL
        text = text// &
            '    if (fp == NULL) {'//NL
        text = text// &
            '        return ffc_unit_last_status;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    return ffc_write_failed(fputs(text, fp));'//NL
        text = text// &
            '}'//NL
        text = text//NL
        text = text// &
            '/* Unformatted scalar transfer writes memory without list-directed'//NL
        text = text// &
            ' * separators or a record terminator. */'//NL
        text = text// &
            'static int ffc_write_unformatted_value('//NL
        text = text// &
            '    FILE *fp, const void *value, size_t size) {'//NL
        text = text// &
            '    if (fp == NULL || value == NULL ||'//NL
        text = text// &
            '        fwrite(value, size, 1, fp) != 1) {'//NL
        text = text// &
            '        ffc_unit_last_status = FFC_IOSTAT_WRITE;'//NL
        text = text// &
            '        return FFC_IOSTAT_WRITE;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    ffc_unit_last_status = 0;'//NL
        text = text// &
            '    return 0;'//NL
        text = text// &
            '}'//NL
        text = text//NL
        text = text// &
            'int _ffc_write_unformatted_i8(FILE *fp, signed char value) {'//NL
        text = text// &
            '    return ffc_write_unformatted_value(fp, &value, sizeof value);'//NL
        text = text// &
            '}'//NL
        text = text//NL
        text = text// &
            'int _ffc_write_unformatted_i16(FILE *fp, short value) {'//NL
        text = text// &
            '    return ffc_write_unformatted_value(fp, &value, sizeof value);'//NL
        text = text// &
            '}'//NL
        text = text//NL
        text = text// &
            'int _ffc_write_unformatted_i32(FILE *fp, int value) {'//NL
        text = text// &
            '    return ffc_write_unformatted_value(fp, &value, sizeof value);'//NL
        text = text// &
            '}'//NL
        text = text//NL
        text = text// &
            'int _ffc_write_unformatted_i64(FILE *fp, long long value) {'//NL
        text = text// &
            '    return ffc_write_unformatted_value(fp, &value, sizeof value);'//NL
        text = text// &
            '}'//NL
        text = text//NL
        text = text// &
            '/* ---- IOSTAT and IOMSG (issue #427) ------------------------ */'//NL
        text = text//NL
        text = text// &
            '/* Fortran reports I/O status through IOSTAT= and IOMSG=. The'//NL
        text = text// &
            ' * classes are fixed by the standard and by what programs test'//NL
        text = text// &
            ' * for:'//NL
        text = text// &
            ' *'//NL
        text = text// &
            ' *   0    success'//NL
        text = text// &
            ' *   -1   end of file      (gfortran''s IOSTAT_END)'//NL
        text = text// &
            ' *   -2   end of record    (gfortran''s IOSTAT_EOR)'//NL
        text = text// &
            ' *   > 0  an error'//NL
        text = text// &
            ' *'//NL
        text = text// &
            ' * The runtime already records an internal status per unit'//NL
        text = text// &
            ' * operation. These map that to the Fortran class and to the'//NL
        text = text// &
            ' * message text, in one place, so every statement reports the'//NL
        text = text// &
            ' * same value for the same condition. */'//NL
        text = text//NL
        text = text// &
            '#define FFC_IOSTAT_END (-1)'//NL
        text = text// &
            '#define FFC_IOSTAT_EOR (-2)'//NL
        text = text//NL
        text = text// &
            '/* Fortran status of the most recent I/O operation. Internal'//NL
        text = text// &
            ' * codes are already positive error numbers, and the end-of-file'//NL
        text = text// &
            ' * and end-of-record classes are stored as themselves, so this'//NL
        text = text// &
            ' * is the recorded status unchanged. It exists so lowering has'//NL
        text = text// &
            ' * one name to call rather than knowing the mapping. */'//NL
        text = text// &
            'int _ffc_iostat(void) {'//NL
        text = text// &
            '    return ffc_unit_last_status;'//NL
        text = text// &
            '}'//NL
        text = text//NL
        text = text// &
            '/* Records an end-of-file condition, so a READ that hits it'//NL
        text = text// &
            ' * reports the same -1 every other statement reports. */'//NL
        text = text// &
            'void _ffc_iostat_set_end(void) {'//NL
        text = text// &
            '    ffc_unit_last_status = FFC_IOSTAT_END;'//NL
        text = text// &
            '}'//NL
        text = text//NL
        text = text// &
            'void _ffc_iostat_clear(void) {'//NL
        text = text// &
            '    ffc_unit_last_status = 0;'//NL
        text = text// &
            '}'//NL
        text = text//NL
        text = text// &
            'static const char *ffc_iostat_text(int status) {'//NL
        text = text// &
            '    switch (status) {'//NL
        text = text// &
            '    case 0:'//NL
        text = text// &
            '        return "";'//NL
        text = text// &
            '    case FFC_IOSTAT_END:'//NL
        text = text// &
            '        return "End of file";'//NL
        text = text// &
            '    case FFC_IOSTAT_EOR:'//NL
        text = text// &
            '        return "End of record";'//NL
        text = text// &
            '    case FFC_IOSTAT_BADUNIT:'//NL
        text = text// &
            '        return "Unit number is out of range";'//NL
        text = text// &
            '    case FFC_IOSTAT_NOUNIT:'//NL
        text = text// &
            '        return "Unit is not connected";'//NL
        text = text// &
            '    case FFC_IOSTAT_INUSE:'//NL
        text = text// &
            '        return "Unit is already connected";'//NL
        text = text// &
            '    case FFC_IOSTAT_OPEN:'//NL
        text = text// &
            '        return "Cannot open file";'//NL
        text = text// &
            '    case FFC_IOSTAT_NOSPACE:'//NL
        text = text// &
            '        return "No free unit for NEWUNIT";'//NL
        text = text// &
            '    case FFC_IOSTAT_WRITE:'//NL
        text = text// &
            '        return "Write failed";'//NL
        text = text// &
            '    default:'//NL
        text = text// &
            '        return "I/O error";'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '}'//NL
        text = text//NL
        text = text// &
            '/* IOMSG= for the most recent operation, written with Fortran'//NL
        text = text// &
            ' * character assignment semantics: the text is truncated to len'//NL
        text = text// &
            ' * and the remainder is blank filled, never NUL terminated.'//NL
        text = text// &
            ' *'//NL
        text = text// &
            ' * The standard defines IOMSG only when an error or end-of-file'//NL
        text = text// &
            ' * condition occurs. After a successful operation this leaves the'//NL
        text = text// &
            ' * variable all blanks rather than untouched, so the destination'//NL
        text = text// &
            ' * is always defined and a program never reads whatever the'//NL
        text = text// &
            ' * buffer happened to hold.'//NL
        text = text// &
            ' *'//NL
        text = text// &
            ' * Writes exactly len characters and a terminating NUL, so dest'//NL
        text = text// &
            ' * must have room for len + 1: the compiler''s character values'//NL
        text = text// &
            ' * are NUL-terminated buffers of the declared length. */'//NL
        text = text// &
            'void _ffc_iomsg(char *dest, int len) {'//NL
        text = text// &
            '    const char *text;'//NL
        text = text// &
            '    int i;'//NL
        text = text// &
            '    if (dest == NULL || len <= 0) {'//NL
        text = text// &
            '        return;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    text = ffc_iostat_text(ffc_unit_last_status);'//NL
        text = text// &
            '    for (i = 0; i < len && text[i] != ''\0''; i++) {'//NL
        text = text// &
            '        dest[i] = text[i];'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    for (; i < len; i++) {'//NL
        text = text// &
            '        dest[i] = '' '';'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    dest[len] = ''\0'';'//NL
        text = text// &
            '}'//NL
        text = text//NL
        text = text// &
            '/* ---- Descriptor storage allocation (issue #428) ----------- */'//NL
        text = text//NL
        text = text// &
            '/* Allocatable arrays and deferred-length characters used to'//NL
        text = text// &
            ' * reach malloc() and free() directly from emitted code, which'//NL
        text = text// &
            ' * meant every size computation, every overflow check, and every'//NL
        text = text// &
            ' * ownership decision was open-coded at each site. These helpers'//NL
        text = text// &
            ' * own that instead: the compiler still decides shape and type,'//NL
        text = text// &
            ' * the runtime decides whether a size is representable, whether'//NL
        text = text// &
            ' * a pointer may be released, and what the status is.'//NL
        text = text// &
            ' *'//NL
        text = text// &
            ' * Sizes arrive as a separate element count and element size, not'//NL
        text = text// &
            ' * as a product, so the multiplication that can overflow happens'//NL
        text = text// &
            ' * here, once, where it is checked. A count of zero is a valid'//NL
        text = text// &
            ' * request: Fortran allows a zero-sized array, and the result is'//NL
        text = text// &
            ' * a non-null pointer that can be released exactly like any'//NL
        text = text// &
            ' * other.'//NL
        text = text// &
            ' *'//NL
        text = text// &
            ' * Status codes are stable, and follow the IOSTAT ranges:'//NL
        text = text// &
            ' *'//NL
        text = text// &
            ' *   0                        success'//NL
        text = text// &
            ' *   FFC_ALLOC_NEGATIVE       negative count or element size'//NL
        text = text// &
            ' *   FFC_ALLOC_OVERFLOW       count * element size is not'//NL
        text = text// &
            ' *                            representable'//NL
        text = text// &
            ' *   FFC_ALLOC_NOMEM          the allocator refused'//NL
        text = text// &
            ' *   FFC_ALLOC_DOUBLE_FREE    release of a pointer that is not'//NL
        text = text// &
            ' *                            live'//NL
        text = text// &
            ' *   FFC_ALLOC_BORROWED       release of storage the descriptor'//NL
        text = text// &
            ' *                            does not own'//NL
        text = text// &
            ' */'//NL
        text = text//NL
        text = text// &
            '#include <stdint.h>'//NL
        text = text// &
            '#include <string.h>'//NL
        text = text//NL
        text = text// &
            '#define FFC_ALLOC_NEGATIVE 6001'//NL
        text = text// &
            '#define FFC_ALLOC_OVERFLOW 6002'//NL
        text = text// &
            '#define FFC_ALLOC_NOMEM 6003'//NL
        text = text// &
            '#define FFC_ALLOC_DOUBLE_FREE 6004'//NL
        text = text// &
            '#define FFC_ALLOC_BORROWED 6005'//NL
        text = text//NL
        text = text// &
            'static int ffc_alloc_last_status = 0;'//NL
        text = text//NL
        text = text// &
            '/* Live allocations handed out by _ffc_alloc, so releasing a'//NL
        text = text// &
            ' * pointer twice is reported instead of corrupting the heap.'//NL
        text = text// &
            ' * Open addressing, power-of-two capacity, grown before it is'//NL
        text = text// &
            ' * half full. Tombstones are not needed: a removed entry is'//NL
        text = text// &
            ' * refilled by rehashing its cluster. */'//NL
        text = text// &
            'static void **ffc_live;'//NL
        text = text// &
            'static size_t ffc_live_cap;'//NL
        text = text// &
            'static size_t ffc_live_count;'//NL
        text = text//NL
        text = text// &
            'static size_t ffc_live_slot(void **table, size_t cap, void *p) {'//NL
        text = text// &
            '    size_t mask = cap - 1;'//NL
        text = text// &
            '    size_t i = (size_t)((uintptr_t)p >> 4) & mask;'//NL
        text = text// &
            '    while (table[i] != NULL && table[i] != p) {'//NL
        text = text// &
            '        i = (i + 1) & mask;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    return i;'//NL
        text = text// &
            '}'//NL
        text = text//NL
        text = text// &
            'static int ffc_live_grow(void) {'//NL
        text = text// &
            '    size_t new_cap = ffc_live_cap ? ffc_live_cap * 2 : 64;'//NL
        text = text// &
            '    void **fresh = calloc(new_cap, sizeof(*fresh));'//NL
        text = text// &
            '    size_t i;'//NL
        text = text// &
            '    if (fresh == NULL) {'//NL
        text = text// &
            '        return -1;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    for (i = 0; i < ffc_live_cap; i++) {'//NL
        text = text// &
            '        if (ffc_live[i] != NULL) {'//NL
        text = text// &
            '            fresh[ffc_live_slot(fresh, new_cap, ffc_live[i])]'//NL
        text = text// &
            '                = ffc_live[i];'//NL
        text = text// &
            '        }'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    free(ffc_live);'//NL
        text = text// &
            '    ffc_live = fresh;'//NL
        text = text// &
            '    ffc_live_cap = new_cap;'//NL
        text = text// &
            '    return 0;'//NL
        text = text// &
            '}'//NL
        text = text//NL
        text = text// &
            'static int ffc_live_add(void *p) {'//NL
        text = text// &
            '    if (ffc_live_count * 2 + 1 >= ffc_live_cap) {'//NL
        text = text// &
            '        if (ffc_live_grow() != 0) {'//NL
        text = text// &
            '            return -1;'//NL
        text = text// &
            '        }'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    ffc_live[ffc_live_slot(ffc_live, ffc_live_cap, p)] = p;'//NL
        text = text// &
            '    ffc_live_count++;'//NL
        text = text// &
            '    return 0;'//NL
        text = text// &
            '}'//NL
        text = text//NL
        text = text// &
            '/* Removes p and rehashes the rest of its cluster, so lookups'//NL
        text = text// &
            ' * that probed past p still find their entries. */'//NL
        text = text// &
            'static int ffc_live_remove(void *p) {'//NL
        text = text// &
            '    size_t i, j, mask;'//NL
        text = text// &
            '    if (ffc_live_cap == 0) {'//NL
        text = text// &
            '        return 0;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    mask = ffc_live_cap - 1;'//NL
        text = text// &
            '    i = ffc_live_slot(ffc_live, ffc_live_cap, p);'//NL
        text = text// &
            '    if (ffc_live[i] != p) {'//NL
        text = text// &
            '        return 0;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    ffc_live[i] = NULL;'//NL
        text = text// &
            '    ffc_live_count--;'//NL
        text = text// &
            '    j = (i + 1) & mask;'//NL
        text = text// &
            '    while (ffc_live[j] != NULL) {'//NL
        text = text// &
            '        void *moved = ffc_live[j];'//NL
        text = text// &
            '        ffc_live[j] = NULL;'//NL
        text = text// &
            '        ffc_live_count--;'//NL
        text = text// &
            '        ffc_live_add(moved);'//NL
        text = text// &
            '        j = (j + 1) & mask;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    return 1;'//NL
        text = text// &
            '}'//NL
        text = text//NL
        text = text// &
            '/* Status of the most recent allocation operation. */'//NL
        text = text// &
            'int _ffc_alloc_status(void) {'//NL
        text = text// &
            '    return ffc_alloc_last_status;'//NL
        text = text// &
            '}'//NL
        text = text//NL
        text = text// &
            '/* count elements of elem_size bytes each. Returns NULL and sets'//NL
        text = text// &
            ' * the status on any rejected request. */'//NL
        text = text// &
            'void *_ffc_alloc(long long count, long long elem_size) {'//NL
        text = text// &
            '    size_t bytes;'//NL
        text = text// &
            '    void *p;'//NL
        text = text// &
            '    if (count < 0 || elem_size < 0) {'//NL
        text = text// &
            '        ffc_alloc_last_status = FFC_ALLOC_NEGATIVE;'//NL
        text = text// &
            '        return NULL;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    if (elem_size != 0 &&'//NL
        text = text// &
            '        count > (long long)(SIZE_MAX / 2)'//NL
        text = text// &
            '                    / elem_size) {'//NL
        text = text// &
            '        ffc_alloc_last_status = FFC_ALLOC_OVERFLOW;'//NL
        text = text// &
            '        return NULL;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    bytes = (size_t)(count * elem_size);'//NL
        text = text// &
            '    /* A zero-sized array still needs a releasable pointer. */'//NL
        text = text// &
            '    p = malloc(bytes != 0 ? bytes : 1);'//NL
        text = text// &
            '    if (p == NULL) {'//NL
        text = text// &
            '        ffc_alloc_last_status = FFC_ALLOC_NOMEM;'//NL
        text = text// &
            '        return NULL;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    if (ffc_live_add(p) != 0) {'//NL
        text = text// &
            '        free(p);'//NL
        text = text// &
            '        ffc_alloc_last_status = FFC_ALLOC_NOMEM;'//NL
        text = text// &
            '        return NULL;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    ffc_alloc_last_status = 0;'//NL
        text = text// &
            '    return p;'//NL
        text = text// &
            '}'//NL
        text = text//NL
        text = text// &
            '/* Like _ffc_alloc, with the storage zeroed. An allocatable'//NL
        text = text// &
            ' * array of a derived element type needs this: every element''s'//NL
        text = text// &
            ' * inline component descriptors must start null. */'//NL
        text = text// &
            'void *_ffc_calloc(long long count, long long elem_size) {'//NL
        text = text// &
            '    void *p = _ffc_alloc(count, elem_size);'//NL
        text = text// &
            '    if (p == NULL) {'//NL
        text = text// &
            '        return NULL;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    if (count > 0 && elem_size > 0) {'//NL
        text = text// &
            '        memset(p, 0, (size_t)(count * elem_size));'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    return p;'//NL
        text = text// &
            '}'//NL
        text = text//NL
        text = text// &
            '/* Resizes an allocation, keeping min(old, new) bytes. old may'//NL
        text = text// &
            ' * be NULL, which makes this a plain allocation. On failure the'//NL
        text = text// &
            ' * old pointer is still live and unchanged. */'//NL
        text = text// &
            'void *_ffc_realloc(void *old, long long count,'//NL
        text = text// &
            '                   long long elem_size) {'//NL
        text = text// &
            '    size_t bytes;'//NL
        text = text// &
            '    void *p;'//NL
        text = text// &
            '    if (old == NULL) {'//NL
        text = text// &
            '        return _ffc_alloc(count, elem_size);'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    if (count < 0 || elem_size < 0) {'//NL
        text = text// &
            '        ffc_alloc_last_status = FFC_ALLOC_NEGATIVE;'//NL
        text = text// &
            '        return NULL;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    if (elem_size != 0 &&'//NL
        text = text// &
            '        count > (long long)(SIZE_MAX / 2)'//NL
        text = text// &
            '                    / elem_size) {'//NL
        text = text// &
            '        ffc_alloc_last_status = FFC_ALLOC_OVERFLOW;'//NL
        text = text// &
            '        return NULL;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    /* Drop the old key before realloc, which may free it: a'//NL
        text = text// &
            '     * freed pointer must not be read again, even as a hash'//NL
        text = text// &
            '     * key. A failed realloc leaves it valid, so it goes back. */'//NL
        text = text// &
            '    if (!ffc_live_remove(old)) {'//NL
        text = text// &
            '        ffc_alloc_last_status = FFC_ALLOC_DOUBLE_FREE;'//NL
        text = text// &
            '        return NULL;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    bytes = (size_t)(count * elem_size);'//NL
        text = text// &
            '    p = realloc(old, bytes != 0 ? bytes : 1);'//NL
        text = text// &
            '    if (p == NULL) {'//NL
        text = text// &
            '        ffc_live_add(old);'//NL
        text = text// &
            '        ffc_alloc_last_status = FFC_ALLOC_NOMEM;'//NL
        text = text// &
            '        return NULL;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    if (ffc_live_add(p) != 0) {'//NL
        text = text// &
            '        ffc_alloc_last_status = FFC_ALLOC_NOMEM;'//NL
        text = text// &
            '        return NULL;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    ffc_alloc_last_status = 0;'//NL
        text = text// &
            '    return p;'//NL
        text = text// &
            '}'//NL
        text = text//NL
        text = text// &
            '/* Releases storage. owns is the descriptor''s ownership flag: a'//NL
        text = text// &
            ' * borrowed descriptor, such as a section view or a dummy'//NL
        text = text// &
            ' * argument, never frees, and says so rather than doing nothing'//NL
        text = text// &
            ' * silently.'//NL
        text = text// &
            ' *'//NL
        text = text// &
            ' * Releasing a null pointer succeeds, matching Fortran''s'//NL
        text = text// &
            ' * deallocate of an unallocated variable and free(NULL). */'//NL
        text = text// &
            'int _ffc_dealloc(void *p, int owns) {'//NL
        text = text// &
            '    if (p == NULL) {'//NL
        text = text// &
            '        ffc_alloc_last_status = 0;'//NL
        text = text// &
            '        return 0;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    if (!owns) {'//NL
        text = text// &
            '        ffc_alloc_last_status = FFC_ALLOC_BORROWED;'//NL
        text = text// &
            '        return FFC_ALLOC_BORROWED;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    if (!ffc_live_remove(p)) {'//NL
        text = text// &
            '        ffc_alloc_last_status = FFC_ALLOC_DOUBLE_FREE;'//NL
        text = text// &
            '        return FFC_ALLOC_DOUBLE_FREE;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    free(p);'//NL
        text = text// &
            '    ffc_alloc_last_status = 0;'//NL
        text = text// &
            '    return 0;'//NL
        text = text// &
            '}'//NL
    end subroutine ffc_runtime_source_text

end module ffc_runtime_source
