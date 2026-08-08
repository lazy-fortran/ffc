program test_session_proc_ptr_scalar_f32_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none
    character(len=1), parameter :: nl = new_line('a')

    ! gfortran.dg/proc_ptr_25.f90 (#448): a same-unit scalar default-real
    ! function called through a typed procedure pointer.
    if (.not. expect_exit_status( &
        'program main'//nl// &
        '  procedure(add), pointer :: f'//nl// &
        '  logical :: g'//nl// &
        '  g = greater(4.0, add(1.0, 2.0))'//nl// &
        '  if (.not. g) stop 1'//nl// &
        '  f => add'//nl// &
        '  g = greater(4.0, f(1.0, 2.0))'//nl// &
        '  if (.not. g) stop 2'//nl// &
        '  stop 0'//nl// &
        'contains'//nl// &
        '  real function add(x, y)'//nl// &
        '    real, intent(in) :: x, y'//nl// &
        '    print *, "add:", x, y'//nl// &
        '    add = x + y'//nl// &
        '  end function add'//nl// &
        '  logical function greater(x, y)'//nl// &
        '    real, intent(in) :: x, y'//nl// &
        '    greater = x > y'//nl// &
        '  end function greater'//nl// &
        'end program main', 0, &
        '/tmp/ffc_session_proc_ptr_scalar_f32_test')) stop 1

    print *, 'PASS: scalar default-real procedure pointer call'
end program test_session_proc_ptr_scalar_f32_compiler
