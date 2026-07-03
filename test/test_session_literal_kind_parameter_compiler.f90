program test_session_literal_kind_parameter_compiler
    use ffc_test_support, only: expect_output
    implicit none

    print *, '=== direct session literal kind parameter compiler test ==='

    ! A real literal's kind suffix naming an arbitrary declared integer
    ! parameter (not just the conventional dp/wp names) resolves that
    ! parameter's folded value: rk = 8 makes 1.0_rk an f64 literal, printed
    ! with full double precision (17 significant digits).
    if (.not. expect_output( &
        'program main'//new_line('a')// &
        '  integer, parameter :: rk = 8'//new_line('a')// &
        '  print *, 1.0_rk'//new_line('a')// &
        'end program main', '   1.0000000000000000     '//new_line('a'), &
        '/tmp/ffc_session_literal_kind_param_f64_test')) stop 1

    ! A kind parameter folding to 4 keeps the literal f32 (9 significant
    ! digits), matching gfortran even though "sp" is not a whitelisted name.
    if (.not. expect_output( &
        'program main'//new_line('a')// &
        '  integer, parameter :: sp = 4'//new_line('a')// &
        '  print *, 1.5_sp'//new_line('a')// &
        'end program main', '   1.50000000    '//new_line('a'), &
        '/tmp/ffc_session_literal_kind_param_f32_test')) stop 1

    ! A kind parameter derived via kind()/selected_real_kind() rather than a
    ! bare literal folds the same way.
    if (.not. expect_output( &
        'program main'//new_line('a')// &
        '  integer, parameter :: dk = kind(0.d0)'//new_line('a')// &
        '  print *, 2.0_dk'//new_line('a')// &
        'end program main', '   2.0000000000000000     '//new_line('a'), &
        '/tmp/ffc_session_literal_kind_param_kind_fn_test')) stop 1

    print *, 'PASS: literal kind suffix naming a declared parameter '// &
        'resolves through direct LIRIC session'
end program test_session_literal_kind_parameter_compiler
