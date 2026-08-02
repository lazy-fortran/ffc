program test_session_character_module_intrinsics_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none

    character(len=*), parameter :: source = &
        'module m'//new_line('a')// &
        'contains'//new_line('a')// &
        '  integer elemental function len_trim(s) result(n)'//new_line('a')// &
        '    character(len=*), intent(in) :: s'//new_line('a')// &
        '    integer :: i'//new_line('a')// &
        '    n = 0'//new_line('a')// &
        '    do i = 1, len(s)'//new_line('a')// &
        '      if (s(i:i) /= " ") n = i'//new_line('a')// &
        '    end do'//new_line('a')// &
        '  end function len_trim'//new_line('a')// &
        '  function trim(s) result(t)'//new_line('a')// &
        '    character(len=*), intent(in) :: s'//new_line('a')// &
        '    character(len=len_trim(s)) :: t'//new_line('a')// &
        '    if (len(t) > 0) t = s(1:len(t))'//new_line('a')// &
        '  end function trim'//new_line('a')// &
        '  subroutine f_string0(value)'//new_line('a')// &
        '    character(len=*), intent(in) :: value'//new_line('a')// &
        '    if (len(value) < 1 .or. value(1:1) /= "x") error stop 4'// &
        new_line('a')// &
        '  end subroutine f_string0'//new_line('a')// &
        'end module m'//new_line('a')// &
        'program main'//new_line('a')// &
        '  use m'//new_line('a')// &
        '  use iso_c_binding, only: c_null_char'//new_line('a')// &
        '  character(len=*), parameter :: p = " A B "'//new_line('a')// &
        '  character(len=5) :: s'//new_line('a')// &
        '  s = " A B "'//new_line('a')// &
        '  if (len_trim(p) /= 4) error stop 10'//new_line('a')// &
        '  if (trim(p) /= " A B") error stop 11'//new_line('a')// &
        '  if (len_trim(s) /= 4) error stop 1'//new_line('a')// &
        '  if (trim(s) /= " A B") error stop 2'//new_line('a')// &
        '  if (len(trim(s)) /= 4) error stop 3'//new_line('a')// &
        '  call f_string0("x" // c_null_char)'//new_line('a')// &
        '  print *, trim(s)'//new_line('a')// &
        'end program main'

    print *, '=== module character intrinsic shadowing test ==='
    if (.not. expect_exit_status(source, 0, &
        '/tmp/ffc_session_character_module_intrinsics_test')) stop 1
    print *, 'PASS: module len_trim/trim calls use character procedure lowering'
end program test_session_character_module_intrinsics_compiler
