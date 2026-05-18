program test_counted_do_negative_step_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== counted DO negative literal step compiler test ==='

    all_passed = .true.
    if (.not. test_negative_literal_step_descends()) all_passed = .false.
    if (.not. test_negative_step_skip_two()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: counted DO with negative literal step lowers through direct LIRIC'

contains

    logical function test_negative_literal_step_descends()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  integer :: total'//new_line('a')// &
            '  total = 0'//new_line('a')// &
            '  do i = 10, 1, -1'//new_line('a')// &
            '    total = total + i'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  stop total'//new_line('a')// &
            'end program main'

        test_negative_literal_step_descends = expect_exit_status( &
            source, 55, '/tmp/ffc_do_neg_step_test')
    end function test_negative_literal_step_descends

    logical function test_negative_step_skip_two()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  integer :: total'//new_line('a')// &
            '  total = 0'//new_line('a')// &
            '  do i = 10, 0, -2'//new_line('a')// &
            '    total = total + i'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  stop total'//new_line('a')// &
            'end program main'

        test_negative_step_skip_two = expect_exit_status( &
            source, 30, '/tmp/ffc_do_neg_step_skip2_test')
    end function test_negative_step_skip_two

end program test_counted_do_negative_step_compiler
