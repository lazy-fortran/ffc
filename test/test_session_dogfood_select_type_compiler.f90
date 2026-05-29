program test_session_dogfood_select_type_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== dogfood: select type dispatch in ffc shape ==='

    all_passed = .true.
    if (.not. test_dogfood_dispatch_mirrors_ffc_shape()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: a select-type dispatch in ffc shape lowers and runs'

contains

    logical function test_dogfood_dispatch_mirrors_ffc_shape()
        ! #144: a hand-written program mimicking ffc's own arena-dispatch shape
        ! (a class(*) value examined with select type / type is, doing real work
        ! per concrete type) lowers cleanly through ffc and runs.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: code'//new_line('a')// &
            '  code = 0'//new_line('a')// &
            '  call dispatch(5, code)'//new_line('a')// &
            '  call dispatch(1.5d0, code)'//new_line('a')// &
            '  stop code'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine dispatch(node, code)'//new_line('a')// &
            '    class(*), intent(in) :: node'//new_line('a')// &
            '    integer, intent(inout) :: code'//new_line('a')// &
            '    select type (n => node)'//new_line('a')// &
            '    type is (integer)'//new_line('a')// &
            '      code = code + n'//new_line('a')// &
            '    type is (real)'//new_line('a')// &
            '      code = code + 10'//new_line('a')// &
            '    class default'//new_line('a')// &
            '      code = code + 100'//new_line('a')// &
            '    end select'//new_line('a')// &
            '  end subroutine dispatch'//new_line('a')// &
            'end program main'

        ! dispatch(5): code = 0 + 5 = 5; dispatch(1.5): code = 5 + 10 = 15.
        test_dogfood_dispatch_mirrors_ffc_shape = expect_exit_status( &
            source, 15, '/tmp/ffc_session_dogfood_st_test')
    end function test_dogfood_dispatch_mirrors_ffc_shape

end program test_session_dogfood_select_type_compiler
