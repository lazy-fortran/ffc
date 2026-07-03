program test_session_declaration_kind_parameter_compiler
    use ffc_test_support, only: expect_output
    implicit none

    print *, '=== direct session declaration-side kind parameter compiler test ==='

    ! real(prec) :: v where prec is a declared "integer, parameter" (not a
    ! hardcoded dp/wp/sp name) folds to f64 storage when prec = 8.
    if (.not. expect_output( &
        'program main'//new_line('a')// &
        '  integer, parameter :: prec = 8'//new_line('a')// &
        '  real(prec) :: v'//new_line('a')// &
        '  v = 1.0_prec / 3.0_prec'//new_line('a')// &
        '  print *, v'//new_line('a')// &
        'end program main', &
        '  0.33333333333333331     '//new_line('a'), &
        '/tmp/ffc_decl_kind_param_f64')) stop 1

    ! Same declared-parameter kind name folds to f32 storage when it is 4,
    ! matching gfortran even though the name is not the conventional "sp".
    if (.not. expect_output( &
        'program main'//new_line('a')// &
        '  integer, parameter :: prec = 4'//new_line('a')// &
        '  real(prec) :: v'//new_line('a')// &
        '  v = 1.0_prec / 3.0_prec'//new_line('a')// &
        '  print *, v'//new_line('a')// &
        'end program main', &
        '  0.333333343    '//new_line('a'), &
        '/tmp/ffc_decl_kind_param_f32')) stop 1

    ! integer(prec) :: v with prec = 8 folds to i64 storage.
    if (.not. expect_output( &
        'program main'//new_line('a')// &
        '  integer, parameter :: prec = 8'//new_line('a')// &
        '  integer(prec) :: v'//new_line('a')// &
        '  v = 9223372036854775807_prec'//new_line('a')// &
        '  print *, v'//new_line('a')// &
        'end program main', &
        '  9223372036854775807'//new_line('a'), &
        '/tmp/ffc_decl_kind_param_i64')) stop 1

    ! A parameter declaration itself (real(prec), parameter :: r = ...) also
    ! resolves the declared kind name.
    if (.not. expect_output( &
        'program main'//new_line('a')// &
        '  integer, parameter :: prec = 8'//new_line('a')// &
        '  real(prec), parameter :: r = 1.0_prec / 3.0_prec'//new_line('a')// &
        '  print *, r'//new_line('a')// &
        'end program main', &
        '  0.33333333333333331     '//new_line('a'), &
        '/tmp/ffc_decl_kind_param_const')) stop 1

    print *, 'PASS: declaration-side real/integer kind specs naming a '// &
        'declared parameter resolve through direct LIRIC session'
end program test_session_declaration_kind_parameter_compiler
