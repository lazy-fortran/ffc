program test_session_use_module_derived_type_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none

    print *, '=== direct session use module derived type compiler test ==='

    if (.not. expect_exit_status( &
        'module point_mod'//new_line('a')// &
        '  type :: point_t'//new_line('a')// &
        '    integer :: x, y'//new_line('a')// &
        '  end type'//new_line('a')// &
        'end module point_mod'//new_line('a')// &
        'program main'//new_line('a')// &
        '  use point_mod'//new_line('a')// &
        '  type(point_t) :: p'//new_line('a')// &
        '  p%x = 7'//new_line('a')// &
        '  stop p%x'//new_line('a')// &
        'end program main', 7, &
        '/tmp/ffc_session_use_module_derived_type_test')) stop 1

    print *, 'PASS: USE module derived type lowers through direct LIRIC session'
end program test_session_use_module_derived_type_compiler
