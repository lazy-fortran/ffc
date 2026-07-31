program test_session_reshape_compiler
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session reshape compiler test ==='

    all_passed = .true.
    if (.not. test_identifier_source_rank2()) all_passed = .false.
    if (.not. test_literal_source_rank2()) all_passed = .false.
    if (.not. test_real_literal_source_rank2()) all_passed = .false.
    if (.not. test_zero_sized_expression_source()) all_passed = .false.
    if (.not. test_zero_sized_assignment()) all_passed = .false.
    if (.not. test_pad_from_zero_sized_source()) all_passed = .false.

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

    ! reshape of a zero-sized array expression (shape(1) has zero elements)
    ! into a zero-sized target: no source element is read and the result is
    ! the standard zero-sized array.
    logical function test_zero_sized_expression_source()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, parameter :: empty(0, 0) = reshape(shape(1), [0, 0])'// &
            new_line('a')// &
            '  print *, size(empty)'//new_line('a')// &
            'end program main'
        test_zero_sized_expression_source = expect_output( &
            source, '           0'//new_line('a'), &
            '/tmp/ffc_session_reshape_zero_expr_test')
    end function test_zero_sized_expression_source

    ! Whole-array assignment of a zero-sized reshape result stays a no-op.
    logical function test_zero_sized_assignment()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: src(0)'//new_line('a')// &
            '  integer :: m(0)'//new_line('a')// &
            '  m = reshape(src, [0])'//new_line('a')// &
            '  print *, size(m)'//new_line('a')// &
            'end program main'
        test_zero_sized_assignment = expect_output( &
            source, '           0'//new_line('a'), &
            '/tmp/ffc_session_reshape_zero_assign_test')
    end function test_zero_sized_assignment


    ! A zero-sized source with pad: every result element comes from pad,
    ! cycling in column-major order.
    logical function test_pad_from_zero_sized_source()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: m(2, 2)'//new_line('a')// &
            '  m = reshape(shape(1), [2, 2], pad=[7, 8])'//new_line('a')// &
            '  print *, m(1, 1)'//new_line('a')// &
            '  print *, m(2, 1)'//new_line('a')// &
            '  print *, m(1, 2)'//new_line('a')// &
            '  print *, m(2, 2)'//new_line('a')// &
            'end program main'
        test_pad_from_zero_sized_source = expect_output( &
            source, '           7'//new_line('a')// &
            '           8'//new_line('a')// &
            '           7'//new_line('a')// &
            '           8'//new_line('a'), &
            '/tmp/ffc_session_reshape_pad_test')
    end function test_pad_from_zero_sized_source

end program test_session_reshape_compiler
