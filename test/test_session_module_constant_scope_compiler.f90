program test_session_module_constant_scope_compiler
    use ffc_test_support, only: expect_output, expect_object_exists
    implicit none

    print *, '=== module named-constant host-association test ==='

    ! A module integer parameter is visible inside a module procedure (#1826),
    ! run end to end through a using program.
    if (.not. expect_output( &
         'module m'//new_line('a')// &
         '    integer, parameter :: LIMIT = 7'//new_line('a')// &
         'contains'//new_line('a')// &
         '    subroutine show()'//new_line('a')// &
         '        print *, LIMIT'//new_line('a')// &
         '    end subroutine show'//new_line('a')// &
         'end module m'//new_line('a')// &
         'program main'//new_line('a')// &
         '    use m'//new_line('a')// &
         '    call show()'//new_line('a')// &
         'end program main', &
         '           7'//new_line('a'), &
         '/tmp/ffc_modconst_int_test')) stop 1
    print *, 'PASS: module integer parameter visible in module procedure'

    ! A module enumerator is visible inside a module procedure (#1826).
    if (.not. expect_output( &
         'module colors'//new_line('a')// &
         '    enum, bind(c)'//new_line('a')// &
         '        enumerator :: RED = 1, GREEN = 2, BLUE = 3'//new_line('a')// &
         '    end enum'//new_line('a')// &
         'contains'//new_line('a')// &
         '    subroutine report()'//new_line('a')// &
         '        print *, RED, BLUE'//new_line('a')// &
         '    end subroutine report'//new_line('a')// &
         'end module colors'//new_line('a')// &
         'program main'//new_line('a')// &
         '    use colors'//new_line('a')// &
         '    call report()'//new_line('a')// &
         'end program main', &
         '           1           3'//new_line('a'), &
         '/tmp/ffc_modconst_enum_test')) stop 1
    print *, 'PASS: module enumerator visible in module procedure'

    ! Real, logical, and character module parameters bind inside a module
    ! procedure body. The using-program import path for non-integer constants
    ! is a separate boundary, so lower the module on its own to an object.
    if (.not. expect_object_exists( &
         'module m'//new_line('a')// &
         '    real, parameter :: SCALE = 2.5'//new_line('a')// &
         '    logical, parameter :: FLAG = .true.'//new_line('a')// &
         '    character(len=*), parameter :: TAG = "hi"'//new_line('a')// &
         'contains'//new_line('a')// &
         '    subroutine doit()'//new_line('a')// &
         '        if (FLAG) print *, SCALE, TAG'//new_line('a')// &
         '    end subroutine doit'//new_line('a')// &
         'end module m', &
         '/tmp/ffc_modconst_mixed_test.o')) stop 1
    print *, 'PASS: real/logical/character parameters bind in module procedure'

    print *, 'PASS: module named constants host-associate into procedures'
end program test_session_module_constant_scope_compiler
