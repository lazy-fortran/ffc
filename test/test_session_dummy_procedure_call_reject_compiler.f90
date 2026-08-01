program test_session_dummy_procedure_call_reject
    ! Calling a dummy procedure declared by an interface block inside a
    ! contained scope has no body in the lowering unit (#576). Calls that pass
    ! actual arguments already diagnose this; an argument-less call skipped the
    ! check and emitted a reference to a nested-procedure symbol nothing ever
    ! defines, so the linked program died at load time with
    ! "undefined symbol: .ffc.nested.<n>.<m>.proc" (exit 127).
    ! The compiler must reject such a call instead of emitting a dangling
    ! reference.
    use ffc_test_support, only: expect_error_contains
    implicit none

    logical :: all_passed

    print *, '=== dummy procedure call rejection test ==='

    all_passed = .true.
    if (.not. test_argument_less_function_call_rejected()) all_passed = .false.
    if (.not. test_uncalled_host_still_rejected()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: argument-less dummy procedure calls are diagnosed'

contains

    logical function test_argument_less_function_call_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  print *, 1'//new_line('a')// &
            'contains'//new_line('a')// &
            '  function g(proc) result(res)'//new_line('a')// &
            '    integer :: res'//new_line('a')// &
            '    interface'//new_line('a')// &
            '      function proc()'//new_line('a')// &
            '        integer :: proc'//new_line('a')// &
            '      end function proc'//new_line('a')// &
            '    end interface'//new_line('a')// &
            '    res = proc()'//new_line('a')// &
            '  end function g'//new_line('a')// &
            'end program main'

        test_argument_less_function_call_rejected = expect_error_contains( &
            source, 'procedure body unavailable for call to proc', &
            '/tmp/ffc_dummy_proc_argless')
    end function test_argument_less_function_call_rejected

    logical function test_uncalled_host_still_rejected()
        ! The corpus shape (pure_formal_proc_3_valid.f90): the host functions
        ! are never called, yet the dangling symbol still killed the program.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            'contains'//new_line('a')// &
            '  pure function f(proc) result(res)'//new_line('a')// &
            '    integer :: res'//new_line('a')// &
            '    interface'//new_line('a')// &
            '      pure function proc()'//new_line('a')// &
            '        integer :: proc'//new_line('a')// &
            '      end function proc'//new_line('a')// &
            '    end interface'//new_line('a')// &
            '    res = proc()'//new_line('a')// &
            '  end function f'//new_line('a')// &
            'end program main'

        test_uncalled_host_still_rejected = expect_error_contains( &
            source, 'procedure body unavailable for call to proc', &
            '/tmp/ffc_dummy_proc_uncalled')
    end function test_uncalled_host_still_rejected

end program test_session_dummy_procedure_call_reject
