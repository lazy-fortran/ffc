program test_session_keyword_arguments_compiler
    ! #408: keyword actual arguments are mapped onto the callee's declared
    ! dummy names, so the caller may reorder or omit optional actuals. The
    ! oracle is the compiled program's exit status (and the compiler's
    ! diagnostic for invalid keyword usage).
    use ffc_test_support, only: expect_exit_status, expect_error_contains
    implicit none

    logical :: all_passed

    print *, '=== keyword argument compiler test ==='

    all_passed = .true.
    if (.not. test_reordered_keywords()) all_passed = .false.
    if (.not. test_positional_then_keyword()) all_passed = .false.
    if (.not. test_function_keywords()) all_passed = .false.
    if (.not. test_module_procedure_keywords()) all_passed = .false.
    if (.not. test_keyword_skips_optional()) all_passed = .false.
    if (.not. test_unknown_keyword_rejected()) all_passed = .false.
    if (.not. test_duplicate_keyword_rejected()) all_passed = .false.
    if (.not. test_positional_after_keyword_rejected()) all_passed = .false.
    if (.not. test_keyword_over_positional_rejected()) all_passed = .false.
    if (.not. test_separate_file_keywords()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: keyword actual arguments bind to declared dummy names'

contains

    logical function test_reordered_keywords()
        ! Fully reordered keyword actuals: diff(b=2, a=9) must compute 9 - 2.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  call diff(b=2, a=9)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine diff(a, b)'//new_line('a')// &
            '    integer, intent(in) :: a, b'//new_line('a')// &
            '    integer :: d'//new_line('a')// &
            '    d = a - b'//new_line('a')// &
            '    stop d'//new_line('a')// &
            '  end subroutine diff'//new_line('a')// &
            'end program main'

        test_reordered_keywords = expect_exit_status( &
            source, 7, '/tmp/ffc_keyword_reorder_test')
    end function test_reordered_keywords

    logical function test_positional_then_keyword()
        ! A positional prefix followed by keyword actuals in reverse order.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  call pick(1, c=4, b=2)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine pick(a, b, c)'//new_line('a')// &
            '    integer, intent(in) :: a, b, c'//new_line('a')// &
            '    integer :: d'//new_line('a')// &
            '    d = 100*a + 10*b + c'//new_line('a')// &
            '    stop d'//new_line('a')// &
            '  end subroutine pick'//new_line('a')// &
            'end program main'

        test_positional_then_keyword = expect_exit_status( &
            source, 124, '/tmp/ffc_keyword_mixed_test')
    end function test_positional_then_keyword

    logical function test_function_keywords()
        ! Keyword actuals in a function reference.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')//   &
            '  implicit none'//new_line('a')// &
            '  integer :: r'//new_line('a')// &
            '  r = sub2(b=3, a=11)'//new_line('a')// &
            '  stop r'//new_line('a')// &
            'contains'//new_line('a')// &
            '  integer function sub2(a, b)'//new_line('a')// &
            '    integer, intent(in) :: a, b'//new_line('a')// &
            '    sub2 = a - b'//new_line('a')// &
            '  end function sub2'//new_line('a')// &
            'end program main'

        test_function_keywords = expect_exit_status( &
            source, 8, '/tmp/ffc_keyword_function_test')
    end function test_function_keywords

    logical function test_module_procedure_keywords()
        ! Keyword actuals against a module procedure's public dummy names.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            'contains'//new_line('a')// &
            '  integer function gap(hi, lo)'//new_line('a')// &
            '    integer, intent(in) :: hi, lo'//new_line('a')// &
            '    gap = hi - lo'//new_line('a')// &
            '  end function gap'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: r'//new_line('a')// &
            '  r = gap(lo=4, hi=17)'//new_line('a')// &
            '  stop r'//new_line('a')// &
            'end program main'

        test_module_procedure_keywords = expect_exit_status( &
            source, 13, '/tmp/ffc_keyword_module_test')
    end function test_module_procedure_keywords

    logical function test_keyword_skips_optional()
        ! An interior optional dummy omitted by keyword association stays
        ! absent, exactly as if it had been dropped from a trailing slot.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  call sel(c=6, a=1)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine sel(a, b, c)'//new_line('a')// &
            '    integer, intent(in) :: a, c'//new_line('a')// &
            '    integer, optional, intent(in) :: b'//new_line('a')// &
            '    if (present(b)) stop 99'//new_line('a')// &
            '    stop a + c'//new_line('a')// &
            '  end subroutine sel'//new_line('a')// &
            'end program main'

        test_keyword_skips_optional = expect_exit_status( &
            source, 7, '/tmp/ffc_keyword_optional_test')
    end function test_keyword_skips_optional

    logical function test_unknown_keyword_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  call diff(a=1, z=2)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine diff(a, b)'//new_line('a')// &
            '    integer, intent(in) :: a, b'//new_line('a')// &
            '    stop a - b'//new_line('a')// &
            '  end subroutine diff'//new_line('a')// &
            'end program main'

        test_unknown_keyword_rejected = expect_error_contains( &
            source, 'no dummy argument', '/tmp/ffc_keyword_unknown_test')
    end function test_unknown_keyword_rejected

    logical function test_duplicate_keyword_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  call diff(a=1, a=2)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine diff(a, b)'//new_line('a')// &
            '    integer, intent(in) :: a, b'//new_line('a')// &
            '    stop a - b'//new_line('a')// &
            '  end subroutine diff'//new_line('a')// &
            'end program main'

        test_duplicate_keyword_rejected = expect_error_contains( &
            source, 'supplied more than once', '/tmp/ffc_keyword_dup_test')
    end function test_duplicate_keyword_rejected

    logical function test_positional_after_keyword_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  call diff(a=1, 2)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine diff(a, b)'//new_line('a')// &
            '    integer, intent(in) :: a, b'//new_line('a')// &
            '    stop a - b'//new_line('a')// &
            '  end subroutine diff'//new_line('a')// &
            'end program main'

        test_positional_after_keyword_rejected = expect_error_contains( &
            source, 'after a keyword argument', '/tmp/ffc_keyword_order_test')
    end function test_positional_after_keyword_rejected

    logical function test_keyword_over_positional_rejected()
        ! A keyword naming a dummy already bound positionally.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  call diff(1, a=2)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine diff(a, b)'//new_line('a')// &
            '    integer, intent(in) :: a, b'//new_line('a')// &
            '    stop a - b'//new_line('a')// &
            '  end subroutine diff'//new_line('a')// &
            'end program main'

        test_keyword_over_positional_rejected = expect_error_contains( &
            source, 'supplied more than once', '/tmp/ffc_keyword_clash_test')
    end function test_keyword_over_positional_rejected

    logical function test_separate_file_keywords() result(ok)
        ! A separately compiled module procedure carries its dummy names in the
        ! .fmod, so the using program may reorder its actuals by keyword.
        character(len=*), parameter :: m_src = '/tmp/ffc_kw_sep_m.f90'
        character(len=*), parameter :: main_src = '/tmp/ffc_kw_sep_main.f90'
        character(len=*), parameter :: m_obj = '/tmp/ffc_kw_sep_m.o'
        character(len=*), parameter :: main_exe = '/tmp/ffc_kw_sep_main'
        character(len=*), parameter :: fmod = '/tmp/ffc_kw_sep_state.fmod'
        integer :: exit_stat, cmd_stat

        ok = .false.
        if (.not. write_file(m_src, &
            'module ffc_kw_sep_state'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine report(hi, lo)'//new_line('a')// &
            '    integer, intent(in) :: hi, lo'//new_line('a')// &
            '    stop hi - lo'//new_line('a')// &
            '  end subroutine report'//new_line('a')// &
            'end module ffc_kw_sep_state')) return
        if (.not. write_file(main_src, &
            'program main'//new_line('a')// &
            '  use ffc_kw_sep_state'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  call report(lo=5, hi=28)'//new_line('a')// &
            'end program main')) return

        call execute_command_line('rm -f '//m_obj//' '//main_exe//' '//fmod)
        call execute_command_line( &
            "sh -c 'exe=$(ls -t build/*/app/ffc build/fo/bin/ffc 2>/dev/null | "// &
            'head -n 1); test -n "$exe" || exit 90; '// &
            '"$exe" -c '//m_src//' -o '//m_obj//' || exit 91; '// &
            '"$exe" '//main_src//' '//m_obj//' -o '//main_exe//" || exit 92'", &
            exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0 .or. exit_stat /= 0) then
            print *, 'FAIL: separate keyword compile pipeline failed, code ', &
                exit_stat
            return
        end if
        call execute_command_line(main_exe, exitstat=exit_stat, cmdstat=cmd_stat)
        call execute_command_line('rm -f '//m_src//' '//main_src//' '//m_obj// &
            ' '//main_exe//' '//fmod)
        if (cmd_stat /= 0) then
            print *, 'FAIL: could not run the linked keyword program'
            return
        end if
        if (exit_stat /= 23) then
            print *, 'FAIL: expected exit 23 from keyword call, got ', exit_stat
            return
        end if
        ok = .true.
    end function test_separate_file_keywords

    logical function write_file(path, contents) result(ok)
        character(len=*), intent(in) :: path
        character(len=*), intent(in) :: contents
        integer :: unit, io_stat

        ok = .false.
        open (newunit=unit, file=path, status='replace', action='write', &
            iostat=io_stat)
        if (io_stat /= 0) return
        write (unit, '(A)') contents
        close (unit)
        ok = .true.
    end function write_file

end program test_session_keyword_arguments_compiler
