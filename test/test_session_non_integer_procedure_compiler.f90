program test_session_non_integer_procedure_compiler
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session non-integer procedure compiler test ==='

    all_passed = .true.
    if (.not. test_real_subroutine()) all_passed = .false.
    if (.not. test_logical_subroutine()) all_passed = .false.
    if (.not. test_real_function()) all_passed = .false.
    if (.not. test_mixed_real_logical_function()) all_passed = .false.
    if (.not. test_logical_function()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: non-integer contained procedures lower through direct LIRIC'

contains

    logical function test_real_subroutine()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  real :: x'//new_line('a')// &
                                       '  x = 1.5'//new_line('a')// &
                                       '  call bump(x)'//new_line('a')// &
                                       '  print *, x'//new_line('a')// &
                                       'contains'//new_line('a')// &
                                       '  subroutine bump(value)'//new_line('a')// &
                                   '    real, intent(inout) :: value'//new_line('a')// &
                                       '    value = value + 1.0'//new_line('a')// &
                                       '  end subroutine bump'//new_line('a')// &
                                       'end program main'

        test_real_subroutine = expect_output(source, '2.500000'//new_line('a'), &
                                              '/tmp/ffc_session_real_sub_test')
    end function test_real_subroutine

    logical function test_logical_subroutine()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  logical :: flag'//new_line('a')// &
                                       '  flag = .false.'//new_line('a')// &
                                       '  call enable(flag)'//new_line('a')// &
                                       '  if (flag) then'//new_line('a')// &
                                       '    print *, 1'//new_line('a')// &
                                       '  else'//new_line('a')// &
                                       '    print *, 0'//new_line('a')// &
                                       '  end if'//new_line('a')// &
                                       'contains'//new_line('a')// &
                                       '  subroutine enable(value)'//new_line('a')// &
                                '    logical, intent(inout) :: value'//new_line('a')// &
                                       '    value = .true.'//new_line('a')// &
                                       '  end subroutine enable'//new_line('a')// &
                                       'end program main'

        test_logical_subroutine = expect_output(source, '           1'//new_line('a'), &
                                                 '/tmp/ffc_session_logical_sub_test')
    end function test_logical_subroutine

    logical function test_real_function()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  real :: x'//new_line('a')// &
                                       '  x = add(1.5, 3.0)'//new_line('a')// &
                                       '  print *, x'//new_line('a')// &
                                       'contains'//new_line('a')// &
                                       '  real function add(a, b)'//new_line('a')// &
                                       '    real, intent(in) :: a'//new_line('a')// &
                                       '    real, intent(in) :: b'//new_line('a')// &
                                       '    add = a + b'//new_line('a')// &
                                       '  end function add'//new_line('a')// &
                                       'end program main'

        test_real_function = expect_output(source, '4.500000'//new_line('a'), &
                                            '/tmp/ffc_session_real_fn_test')
    end function test_real_function

    logical function test_mixed_real_logical_function()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  real :: x'//new_line('a')// &
                                       '  logical :: enabled'//new_line('a')// &
                                       '  x = 1.25'//new_line('a')// &
                                       '  enabled = .true.'//new_line('a')// &
                                       '  print *, choose(x, enabled)'// &
                                       new_line('a')// &
                                       'contains'//new_line('a')// &
                                       '  real function choose(value, flag)'// &
                                       new_line('a')// &
                                       '    real, intent(in) :: value'// &
                                       new_line('a')// &
                                       '    logical, intent(in) :: flag'// &
                                       new_line('a')// &
                                       '    if (flag) then'//new_line('a')// &
                                       '      choose = value + 0.75'//new_line('a')// &
                                       '    else'//new_line('a')// &
                                       '      choose = value'//new_line('a')// &
                                       '    end if'//new_line('a')// &
                                       '  end function choose'//new_line('a')// &
                                       'end program main'

        test_mixed_real_logical_function = expect_output( &
                                            source, '2.000000'//new_line('a'), &
                                            '/tmp/ffc_session_mixed_fn_test')
    end function test_mixed_real_logical_function

    logical function test_logical_function()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  logical :: flag'//new_line('a')// &
                                       '  logical :: ok'//new_line('a')// &
                                       '  flag = .false.'//new_line('a')// &
                                       '  ok = enabled(flag)'//new_line('a')// &
                                       '  if (ok) then'//new_line('a')// &
                                       '    print *, 1'//new_line('a')// &
                                       '  else'//new_line('a')// &
                                       '    print *, 0'//new_line('a')// &
                                       '  end if'//new_line('a')// &
                                       'contains'//new_line('a')// &
                                   '  logical function enabled(flag)'//new_line('a')// &
                                    '    logical, intent(in) :: flag'//new_line('a')// &
                                       '    if (flag) then'//new_line('a')// &
                                       '      enabled = .false.'//new_line('a')// &
                                       '    else'//new_line('a')// &
                                       '      enabled = .true.'//new_line('a')// &
                                       '    end if'//new_line('a')// &
                                       '  end function enabled'//new_line('a')// &
                                       'end program main'

        test_logical_function = expect_output(source, '           1'//new_line('a'), &
                                               '/tmp/ffc_session_logical_fn_test')
    end function test_logical_function

end program test_session_non_integer_procedure_compiler
