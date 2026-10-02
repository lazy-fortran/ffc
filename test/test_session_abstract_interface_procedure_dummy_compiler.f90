! fo: dispatcher
module ffc_case_2b23d77d694bf09939b9d7c7
    implicit none
    private
    public :: case_test_session_abstract_interface_procedure_dummy_compiler
    interface
        subroutine case_test_session_abstract_interface_procedure_dummy_compiler()
        end subroutine case_test_session_abstract_interface_procedure_dummy_compiler
    end interface
end module ffc_case_2b23d77d694bf09939b9d7c7

subroutine case_test_session_abstract_interface_procedure_dummy_compiler()
    ! A PROCEDURE(interface) dummy names an ABSTRACT INTERFACE declared in the
    ! host scope. Its callable address must survive the dummy boundary.
    use ffc_test_support, only: expect_exit_status
    implicit none
    save

    character(len=*), parameter :: source = &
        'program main'//new_line('a')// &
        '  implicit none'//new_line('a')// &
        '  abstract interface'//new_line('a')// &
        '    integer function transform(x)'//new_line('a')// &
        '      integer, intent(in) :: x'//new_line('a')// &
        '    end function transform'//new_line('a')// &
        '  end interface'//new_line('a')// &
        '  if (apply(increment, 4) /= 9) stop 1'//new_line('a')// &
        '  stop 0'//new_line('a')// &
        'contains'//new_line('a')// &
        '  integer function apply(proc, x) result(y)'//new_line('a')// &
        '    procedure(transform) :: proc'//new_line('a')// &
        '    integer, intent(in) :: x'//new_line('a')// &
        '    y = proc(x)'//new_line('a')// &
        '  end function apply'//new_line('a')// &
        '  integer function increment(x)'//new_line('a')// &
        '    integer, intent(in) :: x'//new_line('a')// &
        '    increment = x + 5'//new_line('a')// &
        '  end function increment'//new_line('a')// &
        'end program main'

    if (.not. expect_exit_status(source, 0, &
            '/tmp/ffc_abstract_interface_procedure_dummy')) stop 1
    print *, 'PASS: abstract-interface procedure dummy calls run correctly'
end subroutine case_test_session_abstract_interface_procedure_dummy_compiler
