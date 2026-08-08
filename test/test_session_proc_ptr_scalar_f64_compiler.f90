program test_session_proc_ptr_scalar_f64_compiler
    use ffc_test_support, only: expect_error_contains, expect_output
    implicit none
    character(len=1), parameter :: nl = new_line('a')
    logical :: all_passed

    all_passed = .true.

    ! A typed procedure pointer to a same-unit real(8) function must use the
    ! f64 indirect-call ABI and produce the function's actual value.
    if (.not. expect_output( &
        'program main'//nl// &
        '  implicit none'//nl// &
        '  procedure(add), pointer :: f'//nl// &
        '  real(8) :: result'//nl// &
        '  f => add'//nl// &
        '  result = f(1.0d0, 2.0d0)'//nl// &
        '  if (result /= 3.0d0) stop 1'//nl// &
        '  write(*,''(A)'') "PASS: scalar real(8) procedure pointer call"'//nl// &
        '  stop 0'//nl// &
        'contains'//nl// &
        '  real(8) function add(x, y)'//nl// &
        '    real(8), intent(in) :: x, y'//nl// &
        '    add = x + y'//nl// &
        '  end function add'//nl// &
        'end program main', &
        'PASS: scalar real(8) procedure pointer call'//nl, &
        '/tmp/ffc_session_proc_ptr_scalar_f64_test')) all_passed = .false.

    ! No target assignment means there is no statically valid result ABI.
    if (.not. expect_error_contains( &
        'program main'//nl// &
        '  implicit none'//nl// &
        '  procedure(add), pointer :: f'//nl// &
        '  real(8) :: result'//nl// &
        '  result = f(1.0d0, 2.0d0)'//nl// &
        '  stop 0'//nl// &
        'contains'//nl// &
        '  real(8) function add(x, y)'//nl// &
        '    real(8), intent(in) :: x, y'//nl// &
        '    add = x + y'//nl// &
        '  end function add'//nl// &
        'end program main', &
        'procedure pointer call result is unsupported or ambiguous', &
        '/tmp/ffc_session_proc_ptr_scalar_f64_unresolved_test')) all_passed = .false.

    ! Two possible targets are flow-sensitive even when their result kinds
    ! happen to agree; the direct LIRIC slice refuses to infer one target.
    if (.not. expect_error_contains( &
        'program main'//nl// &
        '  implicit none'//nl// &
        '  procedure(add), pointer :: f'//nl// &
        '  logical :: choose'//nl// &
        '  real(8) :: result'//nl// &
        '  choose = .true.'//nl// &
        '  if (choose) then'//nl// &
        '    f => add'//nl// &
        '  else'//nl// &
        '    f => other'//nl// &
        '  end if'//nl// &
        '  result = f(1.0d0, 2.0d0)'//nl// &
        '  stop 0'//nl// &
        'contains'//nl// &
        '  real(8) function add(x, y)'//nl// &
        '    real(8), intent(in) :: x, y'//nl// &
        '    add = x + y'//nl// &
        '  end function add'//nl// &
        '  real(8) function other(x, y)'//nl// &
        '    real(8), intent(in) :: x, y'//nl// &
        '    other = x - y'//nl// &
        '  end function other'//nl// &
        'end program main', &
        'procedure pointer call result is unsupported or ambiguous', &
        '/tmp/ffc_session_proc_ptr_scalar_f64_flow_test')) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: scalar real(8) procedure pointer boundary'
end program test_session_proc_ptr_scalar_f64_compiler
