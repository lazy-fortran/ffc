program test_session_unsupported_kind_rejection_compiler
    use ffc_test_support, only: expect_error_contains, expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== unsupported numeric kind rejection test ==='

    all_passed = .true.
    if (.not. test_unsupported_real_kind_rejected()) all_passed = .false.
    if (.not. test_unsupported_complex_kind_rejected()) all_passed = .false.
    if (.not. test_unsupported_integer_kind_rejected()) all_passed = .false.
    if (.not. test_supported_kinds_accepted()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: unsupported numeric kinds are diagnosed, not narrowed'

contains

    logical function test_unsupported_real_kind_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real(16) :: x'//new_line('a')// &
            '  x = 1.0'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_unsupported_real_kind_rejected = expect_error_contains( &
            source, 'unsupported real kind', &
            '/tmp/ffc_session_reject_kind_real16')
        if (.not. expect_error_contains(source, 'kind 16 is not supported', &
            '/tmp/ffc_session_reject_kind_real16b')) then
            test_unsupported_real_kind_rejected = .false.
        end if
    end function test_unsupported_real_kind_rejected

    logical function test_unsupported_complex_kind_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  complex(16) :: z'//new_line('a')// &
            '  z = (1.0, 2.0)'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_unsupported_complex_kind_rejected = expect_error_contains( &
            source, 'unsupported complex kind', &
            '/tmp/ffc_session_reject_kind_complex16')
        if (.not. expect_error_contains(source, 'kind 16 is not supported', &
            '/tmp/ffc_session_reject_kind_complex16b')) then
            test_unsupported_complex_kind_rejected = .false.
        end if
    end function test_unsupported_complex_kind_rejected

    logical function test_unsupported_integer_kind_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer(3) :: i'//new_line('a')// &
            '  i = 1'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_unsupported_integer_kind_rejected = expect_error_contains( &
            source, 'unsupported integer kind', &
            '/tmp/ffc_session_reject_kind_integer3')
        if (.not. expect_error_contains(source, 'kind 3 is not yet supported', &
            '/tmp/ffc_session_reject_kind_integer3b')) then
            test_unsupported_integer_kind_rejected = .false.
        end if
    end function test_unsupported_integer_kind_rejected

    logical function test_supported_kinds_accepted()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer(1) :: a'//new_line('a')// &
            '  integer(2) :: b'//new_line('a')// &
            '  integer(4) :: c'//new_line('a')// &
            '  integer(8) :: d'//new_line('a')// &
            '  real(4) :: r4'//new_line('a')// &
            '  real(8) :: r8'//new_line('a')// &
            '  complex(4) :: c4'//new_line('a')// &
            '  complex(8) :: c8'//new_line('a')// &
            '  a = 1'//new_line('a')// &
            '  b = 2'//new_line('a')// &
            '  c = 3'//new_line('a')// &
            '  d = 4'//new_line('a')// &
            '  r4 = 1.0'//new_line('a')// &
            '  r8 = 2.0d0'//new_line('a')// &
            '  c4 = (1.0, 0.0)'//new_line('a')// &
            '  c8 = (2.0d0, 0.0d0)'//new_line('a')// &
            '  stop 7'//new_line('a')// &
            'end program main'

        test_supported_kinds_accepted = expect_exit_status( &
            source, 7, '/tmp/ffc_session_supported_kinds_ok')
    end function test_supported_kinds_accepted

end program test_session_unsupported_kind_rejection_compiler
