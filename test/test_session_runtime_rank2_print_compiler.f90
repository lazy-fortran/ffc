program test_session_runtime_rank2_print_compiler
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, compile_frontend_from_string, &
        INPUT_MODE_STANDARD
    use ffc_test_support, only: expect_error_contains
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    logical :: all_passed

    print *, '=== runtime rank-2 automatic whole-array print test ==='
    all_passed = matches_gfortran(rank2_source(), 'runtime_rank2_print')
    if (.not. test_rank5_refusal()) all_passed = .false.
    if (.not. all_passed) stop 1
    print *, 'PASS: runtime rank-2 whole-array print matches gfortran and '// &
        'unsupported ranks retain a precise refusal'

contains

    function rank2_source() result(source)
        character(len=:), allocatable :: source

        source = 'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  call fill(3, 4)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine fill(n, m)'//new_line('a')// &
            '    integer, intent(in) :: n, m'//new_line('a')// &
            '    integer :: a(0:n, 2:m)'//new_line('a')// &
            '    integer :: i, j'//new_line('a')// &
            '    do j = 2, m'//new_line('a')// &
            '      do i = 0, n'//new_line('a')// &
            '        a(i, j) = 100*j + i'//new_line('a')// &
            '      end do'//new_line('a')// &
            '    end do'//new_line('a')// &
            '    print *, a'//new_line('a')// &
            '  end subroutine fill'//new_line('a')// &
            'end program main'
    end function rank2_source

    logical function test_rank5_refusal()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  call work(2, 2, 2, 2, 2)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine work(n1, n2, n3, n4, n5)'//new_line('a')// &
            '    integer, intent(in) :: n1, n2, n3, n4, n5'//new_line('a')// &
            '    integer :: values(n1, n2, n3, n4, n5)'//new_line('a')// &
            '    print *, values'//new_line('a')// &
            '  end subroutine work'//new_line('a')// &
            'end program main'

        test_rank5_refusal = expect_error_contains(source, &
            'runtime-sized array supports ranks 1 through 4 only', &
            '/tmp/ffc_session_runtime_rank5_print_refusal')
    end function test_rank5_refusal

    logical function matches_gfortran(program_source, stem)
        character(len=*), intent(in) :: program_source, stem
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result
        character(len=:), allocatable :: error_msg
        character(len=:), allocatable :: base, src, exe, ref, ffc_out, ref_out
        integer :: unit, exit_stat, status

        matches_gfortran = .false.
        base = '/tmp/ffc_runtime_rank2_print_'//stem
        src = base//'.f90'
        exe = base//'.ffc'
        ref = base//'.gf'
        ffc_out = base//'.ffc.out'
        ref_out = base//'.gf.out'

        options = compiler_frontend_options_t()
        options%run_semantics = .true.
        options%input_mode = INPUT_MODE_STANDARD
        call compile_frontend_from_string(program_source, frontend_result, options)
        if (.not. frontend_result%success()) then
            print *, 'FAIL[', stem, ']: FortFront rejected source: ', &
                trim(frontend_result%diagnostic_text)
            return
        end if

        call lower_program_to_liric_exe(frontend_result%arena, &
            frontend_result%root_index, exe, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL[', stem, ']: ffc lowering failed: ', trim(error_msg)
            return
        end if

        open (newunit=unit, file=src, status='replace', action='write')
        write (unit, '(A)') program_source
        close (unit)
        call execute_command_line('gfortran -w '//src//' -o '//ref, &
                                  exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL[', stem, ']: gfortran rejected source'
            return
        end if

        call execute_command_line(exe//' > '//ffc_out, exitstat=exit_stat)
        call execute_command_line(ref//' > '//ref_out, exitstat=exit_stat)
        call execute_command_line('diff '//ffc_out//' '//ref_out// &
                                  ' > /dev/null 2>&1', exitstat=status)
        if (status /= 0) then
            print *, 'FAIL[', stem, ']: ffc output differs from gfortran'
            call execute_command_line('diff '//ffc_out//' '//ref_out)
        else
            matches_gfortran = .true.
        end if
        call execute_command_line('rm -f '//src//' '//exe//' '//ref//' '// &
                                  ffc_out//' '//ref_out)
    end function matches_gfortran

end program test_session_runtime_rank2_print_compiler
