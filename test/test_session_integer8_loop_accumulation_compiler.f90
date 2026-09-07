program test_session_integer8_loop_accumulation_compiler
    use ffc_test_support, only: expect_output_matches_gfortran
    implicit none

    logical :: all_passed

    all_passed = .true.
    if (.not. test_i64_counted()) all_passed = .false.
    if (.not. test_i64_cycle()) all_passed = .false.
    if (.not. test_i64_nested()) all_passed = .false.
    if (.not. test_i64_while()) all_passed = .false.
    if (.not. test_i64_infinite_reference()) all_passed = .false.
    if (.not. test_i64_branch()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: integer(8) loop state matches gfortran without truncation'

contains

    logical function test_i64_counted()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer :: i'//new_line('a')// &
            'integer(8) :: s'//new_line('a')// &
            's = 4294967296_8'//new_line('a')// &
            'do i = 1, 4'//new_line('a')// &
            ' s = s + 3_8'//new_line('a')// &
            'end do'//new_line('a')// &
            'print *, s'//new_line('a')// &
            'if (s /= 4294967308_8) error stop 1'//new_line('a')// &
            'end program'

        test_i64_counted = expect_output_matches_gfortran( &
            source, 'i64_counted')
    end function test_i64_counted

    logical function test_i64_cycle()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer :: i'//new_line('a')// &
            'integer(8) :: s'//new_line('a')// &
            's = 4294967296_8'//new_line('a')// &
            'do i = 1, 4'//new_line('a')// &
            ' s = s + i'//new_line('a')// &
            ' if (i == 2) then'//new_line('a')// &
            '  s = s + 100_8'//new_line('a')// &
            '  cycle'//new_line('a')// &
            ' end if'//new_line('a')// &
            ' s = s + 10_8'//new_line('a')// &
            'end do'//new_line('a')// &
            'print *, s'//new_line('a')// &
            'if (s /= 4294967436_8) error stop 1'//new_line('a')// &
            'end program'

        test_i64_cycle = expect_output_matches_gfortran( &
            source, 'i64_cycle')
    end function test_i64_cycle

    logical function test_i64_nested()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer :: i, j'//new_line('a')// &
            'integer(8) :: s'//new_line('a')// &
            's = 4294967296_8'//new_line('a')// &
            'do i = 1, 4'//new_line('a')// &
            ' if (i == 4) exit'//new_line('a')// &
            ' do j = 1, 3'//new_line('a')// &
            '  s = s + i + j'//new_line('a')// &
            ' end do'//new_line('a')// &
            ' if (i == 2) cycle'//new_line('a')// &
            ' s = s + 10_8'//new_line('a')// &
            'end do'//new_line('a')// &
            'print *, s'//new_line('a')// &
            'if (s /= 4294967352_8) error stop 1'//new_line('a')// &
            'end program'

        test_i64_nested = expect_output_matches_gfortran( &
            source, 'i64_nested')
    end function test_i64_nested

    logical function test_i64_while()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer :: i'//new_line('a')// &
            'integer(8) :: s'//new_line('a')// &
            'i = 0'//new_line('a')// &
            's = 4294967296_8'//new_line('a')// &
            'do while (i < 4)'//new_line('a')// &
            ' i = i + 1'//new_line('a')// &
            ' s = s + 4294967296_8'//new_line('a')// &
            ' if (i == 2) cycle'//new_line('a')// &
            ' s = s + 1_8'//new_line('a')// &
            'end do'//new_line('a')// &
            'print *, s'//new_line('a')// &
            'if (s /= 21474836483_8) error stop 1'//new_line('a')// &
            'end program'

        test_i64_while = expect_output_matches_gfortran( &
            source, 'i64_while')
    end function test_i64_while

    logical function test_i64_infinite_reference()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer(8) :: s'//new_line('a')// &
            's = -4294967296_8'//new_line('a')// &
            'call work(s)'//new_line('a')// &
            'print *, s'//new_line('a')// &
            'if (s /= -12884901888_8) error stop 1'//new_line('a')// &
            'contains'//new_line('a')// &
            'subroutine work(value)'//new_line('a')// &
            ' integer(8), intent(inout) :: value'//new_line('a')// &
            ' integer :: i'//new_line('a')// &
            ' i = 0'//new_line('a')// &
            ' do'//new_line('a')// &
            '  i = i + 1'//new_line('a')// &
            '  if (i > 4) exit'//new_line('a')// &
            '  value = value - 2147483648_8'//new_line('a')// &
            ' end do'//new_line('a')// &
            'end subroutine'//new_line('a')// &
            'end program'

        test_i64_infinite_reference = expect_output_matches_gfortran( &
            source, 'i64_infinite_reference')
    end function test_i64_infinite_reference

    logical function test_i64_branch()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer :: i'//new_line('a')// &
            'integer(8) :: s'//new_line('a')// &
            's = 4294967296_8'//new_line('a')// &
            'do i = 1, 4'//new_line('a')// &
            ' if (i == 2) then'//new_line('a')// &
            '  s = s + 100_8'//new_line('a')// &
            ' else'//new_line('a')// &
            '  s = s + i'//new_line('a')// &
            ' end if'//new_line('a')// &
            'end do'//new_line('a')// &
            'print *, s'//new_line('a')// &
            'if (s /= 4294967404_8) error stop 1'//new_line('a')// &
            'end program'

        test_i64_branch = expect_output_matches_gfortran( &
            source, 'i64_branch')
    end function test_i64_branch

end program test_session_integer8_loop_accumulation_compiler
