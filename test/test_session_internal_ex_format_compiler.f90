program test_session_internal_ex_format_compiler
    ! Internal write with E editing followed by X positioning:
    ! write (buf, '(Ew.dEe,nX)') value emits an exact-width exponential field
    ! and then advances the cursor n blanks. Expected buffers match gfortran
    ! byte for byte.
    use ffc_test_support, only: expect_output, expect_error_contains
    implicit none

    logical :: all_passed

    print *, '=== direct session internal E/X format compiler test ==='

    all_passed = .true.
    if (.not. test_e_then_x_then_integer()) all_passed = .false.
    if (.not. test_e_with_exponent_width()) all_passed = .false.
    if (.not. test_trailing_x_blanks()) all_passed = .false.
    if (.not. test_missing_precision_is_diagnosed()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: internal E/X formatting lowers through direct LIRIC session'

contains

    logical function test_e_then_x_then_integer()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=32) :: res'//new_line('a')// &
            '  real(kind=8) :: x'//new_line('a')// &
            '  integer :: n'//new_line('a')// &
            '  x = 12345.678d0'//new_line('a')// &
            '  n = 7'//new_line('a')// &
            "  write(res, '(E12.4,2X,I3)') x, n"//new_line('a')// &
            "  print '(A)', '['//trim(res)//']'"//new_line('a')// &
            'end program main'

        test_e_then_x_then_integer = expect_output( &
            source, '[  0.1235E+05    7]'//new_line('a'), &
            '/tmp/ffc_internal_ex_basic_test')
    end function test_e_then_x_then_integer

    logical function test_e_with_exponent_width()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=32) :: res'//new_line('a')// &
            '  real(kind=8) :: x'//new_line('a')// &
            '  x = 12345.678d0'//new_line('a')// &
            "  write(res, '(E15.4E3)') x"//new_line('a')// &
            "  print '(A)', '['//trim(res)//']'"//new_line('a')// &
            'end program main'

        test_e_with_exponent_width = expect_output( &
            source, '[    0.1235E+005]'//new_line('a'), &
            '/tmp/ffc_internal_ex_expwidth_test')
    end function test_e_with_exponent_width

    logical function test_trailing_x_blanks()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=20) :: res'//new_line('a')// &
            '  real(kind=8) :: x'//new_line('a')// &
            '  x = 12345.678d0'//new_line('a')// &
            "  write(res, '(E12.4,4X)') x"//new_line('a')// &
            "  print '(A)', '['//res//']'"//new_line('a')// &
            'end program main'

        test_trailing_x_blanks = expect_output( &
            source, '[  0.1235E+05        ]'//new_line('a'), &
            '/tmp/ffc_internal_ex_trailing_test')
    end function test_trailing_x_blanks

    logical function test_missing_precision_is_diagnosed()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=32) :: res'//new_line('a')// &
            '  real(kind=8) :: x'//new_line('a')// &
            '  x = 1.0d0'//new_line('a')// &
            "  write(res, '(E12,2X)') x"//new_line('a')// &
            "  print '(A)', trim(res)"//new_line('a')// &
            'end program main'

        test_missing_precision_is_diagnosed = expect_error_contains( &
            source, 'E edit descriptor requires width and precision', &
            '/tmp/ffc_internal_ex_baddesc_test')
    end function test_missing_precision_is_diagnosed

end program test_session_internal_ex_format_compiler
