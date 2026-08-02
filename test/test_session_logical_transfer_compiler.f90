program test_session_logical_transfer_compiler
    use ffc_test_support, only: expect_output
    implicit none

    character(len=*), parameter :: source = &
        'program main'//new_line('a')// &
        '  use iso_fortran_env, only: int8, int16, int32, int64'//new_line('a')// &
        '  logical(int8) :: v1'//new_line('a')// &
        '  logical(int16) :: v2'//new_line('a')// &
        '  logical(int32) :: v4'//new_line('a')// &
        '  logical(int64) :: v8'//new_line('a')// &
        '  integer(int8) :: b1(1), b2(2), b4(4), b8(8)'//new_line('a')// &
        '  integer :: i'//new_line('a')// &
        '  v1 = .true.; v2 = .true.; v4 = .true.; v8 = .true.'//new_line('a')// &
        '  b1 = transfer(v1, b1)'//new_line('a')// &
        '  b2 = transfer(v2, b2)'//new_line('a')// &
        '  b4 = transfer(v4, b4)'//new_line('a')// &
        '  b8 = transfer(v8, b8)'//new_line('a')// &
        '  if (b1(1) /= 1) stop 1'//new_line('a')// &
        '  if (b2(1) /= 1 .or. b2(2) /= 0) stop 2'//new_line('a')// &
        '  if (b4(1) /= 1) stop 3'//new_line('a')// &
        '  if (b8(1) /= 1) stop 4'//new_line('a')// &
        '  do i = 2, 8'//new_line('a')// &
        '    if (b8(i) /= 0) stop 5'//new_line('a')// &
        '  end do'//new_line('a')// &
        '  v1 = .false.; v2 = .false.; v4 = .false.; v8 = .false.'//new_line('a')// &
        '  b8 = transfer(v8, b8)'//new_line('a')// &
        '  do i = 1, 8'//new_line('a')// &
        '    if (b8(i) /= 0) stop 6'//new_line('a')// &
        '  end do'//new_line('a')// &
        "  print *, 'PASS: logical kind transfer'"//new_line('a')// &
        'end program main'

    if (.not. expect_output(source, &
            ' PASS: logical kind transfer'//new_line('a'), &
            '/tmp/ffc_session_logical_transfer')) stop 1
end program test_session_logical_transfer_compiler
