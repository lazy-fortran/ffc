program test_session_associate
    use ffc_test_support, only: expect_exit_status, expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session associate compiler test ==='

    all_passed = .true.
    if (.not. test_associate_scalar_alias()) all_passed = .false.
    if (.not. test_associate_expression_selector()) all_passed = .false.
    if (.not. test_associate_two_names()) all_passed = .false.
    if (.not. test_associate_scope_drops_binding()) all_passed = .false.
    if (.not. test_associate_print_value()) all_passed = .false.
    if (.not. test_associate_real_expr_alias()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: associate constructs lower through direct LIRIC'

contains

    logical function test_associate_scalar_alias()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       'integer :: n'//new_line('a')// &
                                       'n = 7'//new_line('a')// &
                                       'associate (x => n)'//new_line('a')// &
                                       '    stop x'//new_line('a')// &
                                       'end associate'//new_line('a')// &
                                       'end program main'

        test_associate_scalar_alias = expect_exit_status( &
                                  source, 7, &
                                  '/tmp/ffc_session_associate_scalar')
    end function test_associate_scalar_alias

    logical function test_associate_expression_selector()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       'integer :: a'//new_line('a')// &
                                       'integer :: b'//new_line('a')// &
                                       'a = 3'//new_line('a')// &
                                       'b = 5'//new_line('a')// &
                                       'associate (s => a + b)'//new_line('a')// &
                                       '    stop s'//new_line('a')// &
                                       'end associate'//new_line('a')// &
                                       'end program main'

        test_associate_expression_selector = expect_exit_status( &
                                  source, 8, &
                                  '/tmp/ffc_session_associate_expr')
    end function test_associate_expression_selector

    logical function test_associate_two_names()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       'integer :: p'//new_line('a')// &
                                       'integer :: q'//new_line('a')// &
                                       'p = 4'//new_line('a')// &
                                       'q = 6'//new_line('a')// &
                                       'associate (u => p, v => q)'//new_line('a')// &
                                       '    stop u + v'//new_line('a')// &
                                       'end associate'//new_line('a')// &
                                       'end program main'

        test_associate_two_names = expect_exit_status( &
                                  source, 10, &
                                  '/tmp/ffc_session_associate_two')
    end function test_associate_two_names

    logical function test_associate_scope_drops_binding()
        ! The associate name is scoped to the construct; afterwards the original
        ! variable controls the result. Reuse the same identifier outside.
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       'integer :: n'//new_line('a')// &
                                       'n = 2'//new_line('a')// &
                                       'associate (x => n)'//new_line('a')// &
                                       '    stop x'//new_line('a')// &
                                       'end associate'//new_line('a')// &
                                       'stop 99'//new_line('a')// &
                                       'end program main'

        test_associate_scope_drops_binding = expect_exit_status( &
                                  source, 2, &
                                  '/tmp/ffc_session_associate_scope')
    end function test_associate_scope_drops_binding

    logical function test_associate_print_value()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       'integer :: n'//new_line('a')// &
                                       'n = 42'//new_line('a')// &
                                       'associate (x => n)'//new_line('a')// &
                                       '    print *, x'//new_line('a')// &
                                       'end associate'//new_line('a')// &
                                       'end program main'

        test_associate_print_value = expect_output( &
                                  source, '          42'//new_line('a'), &
                                  '/tmp/ffc_session_associate_print')
    end function test_associate_print_value

    logical function test_associate_real_expr_alias()
        ! A real-valued expression selector must lower at its own (real) kind,
        ! be usable in a further real expression, and print correctly. This
        ! guards the f32 selector path (val * 2.0).
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       'real :: val'//new_line('a')// &
                                       'real :: out'//new_line('a')// &
                                       'val = 3.0'//new_line('a')// &
                                       'associate (scaled => val * 2.0)'// &
                                       new_line('a')// &
                                       '    out = scaled + 1.0'//new_line('a')// &
                                       '    print *, out'//new_line('a')// &
                                       'end associate'//new_line('a')// &
                                       'end program main'

        test_associate_real_expr_alias = expect_output( &
                                  source, '   7.00000000    '//new_line('a'), &
                                  '/tmp/ffc_session_associate_real_expr')
    end function test_associate_real_expr_alias

end program test_session_associate
