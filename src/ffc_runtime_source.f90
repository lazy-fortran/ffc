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
    end subroutine ffc_runtime_source_text

end module ffc_runtime_source
