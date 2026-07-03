program test_session_target_attribute_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== direct session target attribute compiler test ==='

    all_passed = .true.
    if (.not. test_target_local()) all_passed = .false.
    if (.not. test_target_save()) all_passed = .false.
    if (.not. test_target_dummy()) all_passed = .false.
    if (.not. test_target_initializer()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: target attribute accepted and ignored'

contains

    logical function test_target_local()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, target :: x'//new_line('a')// &
            '  x = 5'//new_line('a')// &
            '  stop x'//new_line('a')// &
            'end program main'

        test_target_local = expect_exit_status( &
            source, 5, '/tmp/ffc_session_target_local_test')
    end function test_target_local

    logical function test_target_initializer()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, target :: a = 2, b = 1'//new_line('a')// &
            '  integer, pointer :: p'//new_line('a')// &
            '  p => a'//new_line('a')// &
            '  stop p + b'//new_line('a')// &
            'end program main'

        test_target_initializer = expect_exit_status( &
            source, 3, '/tmp/ffc_session_target_init_test')
    end function test_target_initializer

    logical function test_target_save()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, target, save :: x'//new_line('a')// &
            '  x = 6'//new_line('a')// &
            '  stop x'//new_line('a')// &
            'end program main'

        test_target_save = expect_exit_status( &
            source, 6, '/tmp/ffc_session_target_save_test')
    end function test_target_save

    logical function test_target_dummy()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  stop f(7)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  integer function f(x)'//new_line('a')// &
            '    integer, intent(in), target :: x'//new_line('a')// &
            '    f = x'//new_line('a')// &
            '  end function f'//new_line('a')// &
            'end program main'

        test_target_dummy = expect_exit_status( &
            source, 7, '/tmp/ffc_session_target_dummy_test')
    end function test_target_dummy

end program test_session_target_attribute_compiler
