program test_session_reshape_compiler
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session reshape compiler test ==='

    all_passed = .true.
    if (.not. test_identifier_source_rank2()) all_passed = .false.
    if (.not. test_literal_source_rank2()) all_passed = .false.
    if (.not. test_real_literal_source_rank2()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: reshape lowers through direct LIRIC session'

contains

    ! reshape of a declared rank-1 array into a rank-2 target: elements fill
    ! column-major, so m(1,1),m(1,2),m(1,3) read 1,3,5 and m(2,*) read 2,4,6.
    logical function test_identifier_source_rank2()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: src(6)'//new_line('a')// &
            '  integer :: m(2, 3)'//new_line('a')// &
            '  integer :: i, j'//new_line('a')// &
            '  src = [1, 2, 3, 4, 5, 6]'//new_line('a')// &
            '  m = reshape(src, [2, 3])'//new_line('a')// &
            '  do i = 1, 2'//new_line('a')// &
            '     do j = 1, 3'//new_line('a')// &
            '        print *, m(i, j)'//new_line('a')// &
            '     end do'//new_line('a')// &
            '  end do'//new_line('a')// &
            'end program main'
        test_identifier_source_rank2 = expect_output( &
            source, '           1'//new_line('a')// &
            '           3'//new_line('a')// &
            '           5'//new_line('a')// &
            '           2'//new_line('a')// &
            '           4'//new_line('a')// &
            '           6'//new_line('a'), &
            '/tmp/ffc_session_reshape_ident_test')
    end function test_identifier_source_rank2

    ! reshape of an inline array literal into a rank-2 target.
    logical function test_literal_source_rank2()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: m(2, 2)'//new_line('a')// &
            '  m = reshape([10, 20, 30, 40], [2, 2])'//new_line('a')// &
            '  print *, m(1, 1)'//new_line('a')// &
            '  print *, m(2, 1)'//new_line('a')// &
            '  print *, m(1, 2)'//new_line('a')// &
            '  print *, m(2, 2)'//new_line('a')// &
            'end program main'
        test_literal_source_rank2 = expect_output( &
            source, '          10'//new_line('a')// &
            '          20'//new_line('a')// &
            '          30'//new_line('a')// &
            '          40'//new_line('a'), &
            '/tmp/ffc_session_reshape_literal_test')
    end function test_literal_source_rank2

    ! reshape of a real array literal: element kind follows the target.
    logical function test_real_literal_source_rank2()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: r(2, 2)'//new_line('a')// &
            '  r = reshape([1.5, 2.5, 3.5, 4.5], [2, 2])'//new_line('a')// &
            '  print *, r(1, 1)'//new_line('a')// &
            '  print *, r(2, 2)'//new_line('a')// &
            'end program main'
        test_real_literal_source_rank2 = expect_output( &
            source, '   1.50000000    '//new_line('a')// &
            '   4.50000000    '//new_line('a'), &
            '/tmp/ffc_session_reshape_real_test')
    end function test_real_literal_source_rank2

end program test_session_reshape_compiler
