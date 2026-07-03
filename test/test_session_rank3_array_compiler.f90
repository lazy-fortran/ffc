program test_session_rank3_array_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== direct session rank-3/4 array compiler test ==='

    all_passed = .true.
    if (.not. test_rank3_index()) all_passed = .false.
    if (.not. test_rank3_explicit_bounds()) all_passed = .false.
    if (.not. test_rank3_size_dim()) all_passed = .false.
    if (.not. test_rank4_index()) all_passed = .false.
    if (.not. test_rank3_real_roundtrip()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: rank-3 and rank-4 fixed arrays lower through direct LIRIC'

contains

    logical function test_rank3_index()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(2, 3, 4)'//new_line('a')// &
            '  integer :: i, j, k'//new_line('a')// &
            '  do k = 1, 4'//new_line('a')// &
            '    do j = 1, 3'//new_line('a')// &
            '      do i = 1, 2'//new_line('a')// &
            '        a(i, j, k) = i + 2*j + 4*k'//new_line('a')// &
            '      end do'//new_line('a')// &
            '    end do'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  stop a(2, 3, 4)'//new_line('a')// &
            'end program main'

        test_rank3_index = expect_exit_status( &
            source, 24, '/tmp/ffc_session_rank3_index_test')
    end function test_rank3_index

    logical function test_rank3_explicit_bounds()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(2:4, 0:1, 5)'//new_line('a')// &
            '  stop lbound(a,1) + ubound(a,1) + lbound(a,2)'// &
            ' + ubound(a,2) + ubound(a,3)'//new_line('a')// &
            'end program main'

        test_rank3_explicit_bounds = expect_exit_status( &
            source, 12, '/tmp/ffc_session_rank3_bounds_test')
    end function test_rank3_explicit_bounds

    logical function test_rank3_size_dim()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(2, 3, 4)'//new_line('a')// &
            '  stop size(a,1)*100 + size(a,2)*10 + size(a,3)'//new_line('a')// &
            'end program main'

        test_rank3_size_dim = expect_exit_status( &
            source, 234, '/tmp/ffc_session_rank3_size_test')
    end function test_rank3_size_dim

    logical function test_rank4_index()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(2, 2, 2, 2)'//new_line('a')// &
            '  a = 0'//new_line('a')// &
            '  a(2, 2, 2, 2) = 7'//new_line('a')// &
            '  a(1, 1, 1, 1) = 3'//new_line('a')// &
            '  stop a(2, 2, 2, 2) + a(1, 1, 1, 1)'//new_line('a')// &
            'end program main'

        test_rank4_index = expect_exit_status( &
            source, 10, '/tmp/ffc_session_rank4_index_test')
    end function test_rank4_index

    logical function test_rank3_real_roundtrip()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: r(3, 3, 3)'//new_line('a')// &
            '  r = 0.0'//new_line('a')// &
            '  r(2, 3, 1) = 5.0'//new_line('a')// &
            '  stop int(r(2, 3, 1)) + 1'//new_line('a')// &
            'end program main'

        test_rank3_real_roundtrip = expect_exit_status( &
            source, 6, '/tmp/ffc_session_rank3_real_test')
    end function test_rank3_real_roundtrip

end program test_session_rank3_array_compiler
