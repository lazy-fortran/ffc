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
        text = text//NL
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
            '/* Case-sensitive compare; the compiler lowers STATUS= values in'//NL
        text = text// &
            ' * lower case. */'//NL
        text = text// &
            'static int ffc_streq(const char *a, const char *b) {'//NL
        text = text// &
            '    if (a == NULL || b == NULL) {'//NL
        text = text// &
            '        return 0;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    while (*a != ''\0'' && *a == *b) {'//NL
        text = text// &
            '        a++;'//NL
        text = text// &
            '        b++;'//NL
        text = text// &
            '    }'//NL
        text = text// &
            '    return *a == *b;'//NL
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
            '                   const char *status) {'//NL
        text = text// &
            '    FILE *fp;'//NL
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
            '    if (path == NULL || path[0] == ''\0'' ||'//NL
        text = text// &
            '        ffc_streq(status, "scratch")) {'//NL
        text = text// &
            '        fp = tmpfile();'//NL
        text = text// &
            '    } else {'//NL
        text = text// &
            '        fp = ffc_unit_fopen(path, status);'//NL
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
    end subroutine ffc_runtime_source_text

end module ffc_runtime_source
