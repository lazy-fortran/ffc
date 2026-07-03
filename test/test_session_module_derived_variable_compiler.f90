program test_session_module_derived_variable_compiler
    use ffc_test_support, only: expect_exit_status, expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session module derived variable compiler test ==='

    all_passed = .true.
    if (.not. test_module_derived_integer_defaults()) all_passed = .false.
    if (.not. test_module_derived_use_rename()) all_passed = .false.
    if (.not. test_module_derived_real_logical_defaults()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: module derived variables lower through direct LIRIC'

contains

    logical function test_module_derived_integer_defaults()
        ! A module scalar derived variable stores as a flat slot global; a using
        ! unit reads its compile-time component defaults.
        character(len=*), parameter :: source = &
            'module m_a'//new_line('a')// &
            '  type :: t_a'//new_line('a')// &
            '    integer :: x = 42'//new_line('a')// &
            '  end type t_a'//new_line('a')// &
            '  type(t_a) :: g_a'//new_line('a')// &
            'end module m_a'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m_a'//new_line('a')// &
            '  stop g_a%x'//new_line('a')// &
            'end program main'

        test_module_derived_integer_defaults = expect_exit_status( &
            source, 42, '/tmp/ffc_session_mod_derived_int_test')
    end function test_module_derived_integer_defaults

    logical function test_module_derived_use_rename()
        ! A USE rename aliases the module derived variable under a local name.
        character(len=*), parameter :: source = &
            'module m_b'//new_line('a')// &
            '  type :: t_b'//new_line('a')// &
            '    logical :: b = .true.'//new_line('a')// &
            '    integer :: n = 7'//new_line('a')// &
            '  end type t_b'//new_line('a')// &
            '  type(t_b) :: gv'//new_line('a')// &
            'end module m_b'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m_b, alias => gv'//new_line('a')// &
            '  if (.not. alias%b) stop 1'//new_line('a')// &
            '  stop alias%n'//new_line('a')// &
            'end program main'

        test_module_derived_use_rename = expect_exit_status( &
            source, 7, '/tmp/ffc_session_mod_derived_rename_test')
    end function test_module_derived_use_rename

    logical function test_module_derived_real_logical_defaults()
        ! Real (default-real f32) and logical component defaults fold into the
        ! module variable's static initial bytes.
        character(len=*), parameter :: source = &
            'module m_c'//new_line('a')// &
            '  type :: t_c'//new_line('a')// &
            '    real :: r = 2.5'//new_line('a')// &
            '    logical :: flag = .false.'//new_line('a')// &
            '  end type t_c'//new_line('a')// &
            '  type(t_c) :: gc'//new_line('a')// &
            'end module m_c'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m_c'//new_line('a')// &
            '  print *, gc%r, gc%flag'//new_line('a')// &
            'end program main'

        test_module_derived_real_logical_defaults = expect_output( &
            source, '   2.50000000     F'//new_line('a'), &
            '/tmp/ffc_session_mod_derived_real_test')
    end function test_module_derived_real_logical_defaults

end program test_session_module_derived_variable_compiler
