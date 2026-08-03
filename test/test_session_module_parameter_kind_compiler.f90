program test_session_module_parameter_kind_compiler
    use ffc_test_support, only: expect_output
    implicit none

    character(len=*), parameter :: source = &
        'module math_utils_kinds'//new_line('a')// &
        '  implicit none'//new_line('a')// &
        '  integer, parameter :: dp = selected_real_kind(15, 307)'// &
        new_line('a')// &
        'contains'//new_line('a')// &
        '  function square(x) result(res)'//new_line('a')// &
        '    implicit none'//new_line('a')// &
        '    real(dp), intent(in) :: x'//new_line('a')// &
        '    real(dp) :: res'//new_line('a')// &
        '    res = x * x'//new_line('a')// &
        '  end function square'//new_line('a')// &
        'end module math_utils_kinds'//new_line('a')// &
        'program test_module_only'//new_line('a')// &
        '  use math_utils_kinds, only: square'//new_line('a')// &
        '  implicit none'//new_line('a')// &
        '  real(selected_real_kind(15, 307)) :: x'//new_line('a')// &
        '  x = 2.5'//new_line('a')// &
        "  print *, 'Square:', square(x)"//new_line('a')// &
        'end program test_module_only'

    if (.not. expect_output(source, &
            ' Square:   6.2500000000000000     '//new_line('a'), &
            '/tmp/ffc_session_module_parameter_kind_test')) stop 1

    print *, 'PASS: selected_real_kind propagates through module procedure arguments'
end program test_session_module_parameter_kind_compiler
