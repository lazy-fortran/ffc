program test_session_internal_read_list_directed_compiler
    ! List-directed internal read: read (buf, *) value parses an integer,
    ! real, or character scalar from a character variable.
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session list-directed internal read compiler test ==='

    all_passed = .true.
    if (.not. test_list_directed_integer()) all_passed = .false.
    if (.not. test_list_directed_real()) all_passed = .false.
    if (.not. test_list_directed_character()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: list-directed internal read lowers through direct LIRIC session'

contains

    logical function test_list_directed_integer()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=20) :: buf'//new_line('a')// &
            '  integer :: v'//new_line('a')// &
            "  buf = '  42'"//new_line('a')// &
            '  read (buf, *) v'//new_line('a')// &
            '  print *, v'//new_line('a')// &
            'end program main'

        test_list_directed_integer = expect_output( &
            source, '          42'//new_line('a'), '/tmp/ffc_ilread_i_test')
    end function test_list_directed_integer

    logical function test_list_directed_real()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=20) :: buf'//new_line('a')// &
            '  real :: x'//new_line('a')// &
            "  buf = '2.5'"//new_line('a')// &
            '  read (buf, *) x'//new_line('a')// &
            '  print *, x'//new_line('a')// &
            'end program main'

        test_list_directed_real = expect_output( &
            source, '   2.50000000    '//new_line('a'), '/tmp/ffc_ilread_r_test')
    end function test_list_directed_real

    logical function test_list_directed_character()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=10) :: buf'//new_line('a')// &
            '  character(len=4) :: v'//new_line('a')// &
            "  buf = 'ab'"//new_line('a')// &
            '  read (buf, *) v'//new_line('a')// &
            "  print '(A)', '['//v//']'"//new_line('a')// &
            'end program main'

        test_list_directed_character = expect_output( &
            source, '[ab  ]'//new_line('a'), '/tmp/ffc_ilread_c_test')
    end function test_list_directed_character

end program test_session_internal_read_list_directed_compiler
