program test_session_rank2_array_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none

    print *, '=== direct session rank-2 array compiler test ==='
    if (.not. test_rank2_array_with_explicit_bounds()) stop 1
    print *, 'PASS: rank-2 fixed-size integer arrays lower through direct LIRIC'

contains

    logical function test_rank2_array_with_explicit_bounds()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(0:1, 2:4)'//new_line('a')// &
            '  a(0, 2) = 4'//new_line('a')// &
            '  a(1, 4) = 5'//new_line('a')// &
            '  call bump(a(1, 4))'//new_line('a')// &
            '  stop a(0, 2) + a(1, 4)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine bump(x)'//new_line('a')// &
            '    integer :: x'//new_line('a')// &
            '    x = x + 1'//new_line('a')// &
            '  end subroutine bump'//new_line('a')// &
            'end program main'

        test_rank2_array_with_explicit_bounds = expect_exit_status( &
            source, 10, '/tmp/ffc_session_rank2_array_test')
    end function test_rank2_array_with_explicit_bounds

end program test_session_rank2_array_compiler
