program test_session_real8_function_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none

    ! A contained function whose result is real(8) declared via result(b) with a
    ! body declaration, called from main and from a sibling real(8) function.
    ! The body of scaled mixes an f32 dummy with an f64 literal, exercising the
    ! f32->f64 operand promotion in an f64 result body. nint() of the real(8)
    ! result drives a deterministic exit status.
    character(len=*), parameter :: source = &
        'program main'//new_line('a')// &
        '  integer, parameter :: dp = kind(0.d0)'//new_line('a')// &
        '  integer :: code'//new_line('a')// &
        '  code = nint(total())'//new_line('a')// &
        '  stop code'//new_line('a')// &
        'contains'//new_line('a')// &
        '  function scaled(x) result(b)'//new_line('a')// &
        '    real, intent(in) :: x'//new_line('a')// &
        '    real(8) :: b'//new_line('a')// &
        '    b = x + 1.0d0'//new_line('a')// &
        '  end function scaled'//new_line('a')// &
        '  function total() result(r)'//new_line('a')// &
        '    real(dp) :: r'//new_line('a')// &
        '    r = scaled(40.0) + scaled(1.0)'//new_line('a')// &
        '  end function total'//new_line('a')// &
        'end program main'

    print *, '=== real(8) contained function compiler test ==='

    ! scaled(40)=41.0, scaled(1)=2.0, total=43.0, nint=43.
    if (.not. expect_exit_status(source, 43, &
        '/tmp/ffc_session_real8_function_test')) stop 1

    print *, 'PASS: real(8) contained functions lower through direct LIRIC session'
end program test_session_real8_function_compiler
