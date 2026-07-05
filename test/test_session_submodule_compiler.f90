program test_session_submodule_compiler
    use ffc_test_support, only: expect_exit_status, expect_error_contains
    implicit none

    logical :: all_passed

    print *, '=== submodule compiler test ==='

    all_passed = .true.
    if (.not. test_restated_module_function()) all_passed = .false.
    if (.not. test_separate_module_subroutine()) all_passed = .false.
    if (.not. test_separate_module_function()) all_passed = .false.
    if (.not. test_generic_interface_body_specific()) all_passed = .false.
    if (.not. test_parent_module_not_found()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: single-file submodules lower against the parent module'

contains

    logical function test_restated_module_function()
        ! #292: a submodule restates and implements a module procedure declared
        ! by an interface body in the parent module; the program calls through.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  interface'//new_line('a')// &
            '    module function f(x) result(y)'//new_line('a')// &
            '      integer, intent(in) :: x'//new_line('a')// &
            '      integer :: y'//new_line('a')// &
            '    end function'//new_line('a')// &
            '  end interface'//new_line('a')// &
            'end module'//new_line('a')// &
            'submodule (m) s'//new_line('a')// &
            'contains'//new_line('a')// &
            '  module function f(x) result(y)'//new_line('a')// &
            '    integer, intent(in) :: x'//new_line('a')// &
            '    integer :: y'//new_line('a')// &
            '    y = 2*x'//new_line('a')// &
            '  end function'//new_line('a')// &
            'end submodule'//new_line('a')// &
            'program p'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  stop f(21)'//new_line('a')// &
            'end program'

        test_restated_module_function = expect_exit_status( &
                source, 42, '/tmp/ffc_session_submod_restated_test')
        end function test_restated_module_function

        logical function test_separate_module_subroutine()
            ! #292: the separate `module procedure P` body inherits its dummy
            ! declarations from the parent interface, so an out argument resolves.
            character(len=*), parameter :: source = &
                'module m2'//new_line('a')// &
                '  interface'//new_line('a')// &
                '    module subroutine setval(x)'//new_line('a')// &
                '      integer, intent(out) :: x'//new_line('a')// &
                '    end subroutine'//new_line('a')// &
                '  end interface'//new_line('a')// &
                'end module'//new_line('a')// &
                'submodule (m2) s2'//new_line('a')// &
                'contains'//new_line('a')// &
                '  module procedure setval'//new_line('a')// &
                '    x = 9'//new_line('a')// &
                '  end procedure'//new_line('a')// &
                'end submodule'//new_line('a')// &
                'program p2'//new_line('a')// &
                '  use m2'//new_line('a')// &
                '  integer :: k'//new_line('a')// &
                '  call setval(k)'//new_line('a')// &
                '  stop k'//new_line('a')// &
                'end program'

            test_separate_module_subroutine = expect_exit_status( &
                    source, 9, '/tmp/ffc_session_submod_sep_sub_test')
            end function test_separate_module_subroutine

            logical function test_separate_module_function()
                ! #292: the separate form omits the function signature; the parent
                ! interface supplies both the result kind and the dummy declarations.
                character(len=*), parameter :: source = &
                    'module m3'//new_line('a')// &
                    '  interface'//new_line('a')// &
                    '    module function triple(n) result(r)'//new_line('a')// &
                    '      integer, intent(in) :: n'//new_line('a')// &
                    '      integer :: r'//new_line('a')// &
                    '    end function'//new_line('a')// &
                    '  end interface'//new_line('a')// &
                    'end module'//new_line('a')// &
                    'submodule (m3) s3'//new_line('a')// &
                    'contains'//new_line('a')// &
                    '  module procedure triple'//new_line('a')// &
                    '    r = 3*n'//new_line('a')// &
                    '  end procedure'//new_line('a')// &
                    'end submodule'//new_line('a')// &
                    'program p3'//new_line('a')// &
                    '  use m3, only: triple'//new_line('a')// &
                    '  stop triple(4)'//new_line('a')// &
                    'end program'

                test_separate_module_function = expect_exit_status( &
                        source, 12, '/tmp/ffc_session_submod_sep_fn_test')
                end function test_separate_module_function

                logical function test_generic_interface_body_specific()
                    ! #292: a named generic interface whose specific is a module-procedure
                    ! interface body (`module subroutine impl(...)`) implemented in the
                    ! submodule; a call through the generic name resolves to that specific.
                    character(len=*), parameter :: source = &
                        'module mg'//new_line('a')// &
                        '  interface gen'//new_line('a')// &
                        '    module subroutine impl_sub(n)'//new_line('a')// &
                        '      integer, intent(inout) :: n'//new_line('a')// &
                        '    end subroutine'//new_line('a')// &
                        '  end interface'//new_line('a')// &
                        'end module'//new_line('a')// &
                        'submodule (mg) mgs'//new_line('a')// &
                        'contains'//new_line('a')// &
                        '  module procedure impl_sub'//new_line('a')// &
                        '    n = 3'//new_line('a')// &
                        '  end procedure'//new_line('a')// &
                        'end submodule'//new_line('a')// &
                        'program pg'//new_line('a')// &
                        '  use mg'//new_line('a')// &
                        '  integer :: n'//new_line('a')// &
                        '  n = 2'//new_line('a')// &
                        '  call gen(n)'//new_line('a')// &
                        '  stop n'//new_line('a')// &
                        'end program'

                    test_generic_interface_body_specific = expect_exit_status( &
                        source, 3, '/tmp/ffc_session_submod_generic_body_test')
                end function test_generic_interface_body_specific

                logical function test_parent_module_not_found()
                    ! #292: a submodule whose parent module is absent from the compilation
                    ! set is rejected with a targeted diagnostic.
                    character(len=*), parameter :: source = &
                        'submodule (nonexistent_parent) orphan'//new_line('a')// &
                        'contains'//new_line('a')// &
                        '  module subroutine do_thing(x)'//new_line('a')// &
                        '    integer, intent(inout) :: x'//new_line('a')// &
                        '    x = 7'//new_line('a')// &
                        '  end subroutine'//new_line('a')// &
                        'end submodule'//new_line('a')// &
                        'program p'//new_line('a')// &
                        '  implicit none'//new_line('a')// &
                        '  integer :: n'//new_line('a')// &
                        '  n = 1'//new_line('a')// &
                        '  print *, n'//new_line('a')// &
                        'end program'

                    test_parent_module_not_found = expect_error_contains( &
                        source, 'submodule parent module not found', &
                        '/tmp/ffc_session_submod_orphan_test')
                end function test_parent_module_not_found

            end program test_session_submodule_compiler
