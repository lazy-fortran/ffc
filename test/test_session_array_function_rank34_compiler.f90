program test_session_array_function_rank34_compiler
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, compile_frontend_from_string, &
        INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    logical :: all_passed

    print *, '=== direct session rank-3/4 array function result test ==='

    all_passed = .true.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  integer :: a(2, 3, 2)'//new_line('a')// &
        '  a = make_rank3()'//new_line('a')// &
        '  if (a(1, 1, 1) /= 111 .or. a(2, 3, 2) /= 232 .or. '// &
        'sum(a) /= 2058) stop 31'//new_line('a')// &
        '  print *, a(1, 1, 1), a(2, 3, 2), sum(a)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  function make_rank3() result(out)'//new_line('a')// &
        '    integer :: out(2, 3, 2)'//new_line('a')// &
        '    integer :: i, j, k'//new_line('a')// &
        '    do k = 1, 2'//new_line('a')// &
        '      do j = 1, 3'//new_line('a')// &
        '        do i = 1, 2'//new_line('a')// &
        '          out(i, j, k) = 100*k + 10*j + i'//new_line('a')// &
        '        end do'//new_line('a')// &
        '      end do'//new_line('a')// &
        '    end do'//new_line('a')// &
        '  end function make_rank3'//new_line('a')// &
        'end program main', &
        'rank3')) all_passed = .false.

    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  real :: a(2, 2, 2, 2)'//new_line('a')// &
        '  a = make_rank4()'//new_line('a')// &
        '  if (a(1, 1, 1, 1) /= 1111.0 .or. '// &
        'a(2, 2, 2, 2) /= 2222.0 .or. sum(a) /= 26664.0) stop 41'// &
        new_line('a')// &
        '  print *, a(1, 1, 1, 1), a(2, 2, 2, 2), sum(a)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  function make_rank4() result(out)'//new_line('a')// &
        '    real :: out(2, 2, 2, 2)'//new_line('a')// &
        '    integer :: i, j, k, l'//new_line('a')// &
        '    do l = 1, 2'//new_line('a')// &
        '      do k = 1, 2'//new_line('a')// &
        '        do j = 1, 2'//new_line('a')// &
        '          do i = 1, 2'//new_line('a')// &
        '            out(i, j, k, l) = real(1000*l + 100*k + 10*j + i)'// &
        new_line('a')// &
        '          end do'//new_line('a')// &
        '        end do'//new_line('a')// &
        '      end do'//new_line('a')// &
        '    end do'//new_line('a')// &
        '  end function make_rank4'//new_line('a')// &
        'end program main', &
        'rank4')) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: rank-3 and rank-4 array function results match gfortran'

contains

    logical function matches_gfortran(source, stem)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: stem
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result
        character(len=:), allocatable :: error_msg
        character(len=:), allocatable :: base, src, exe, ref, ffc_out, ref_out
        integer :: unit, exit_stat, status

        matches_gfortran = .false.
        base = '/tmp/ffc_arrfn_rank34_'//trim(stem)
        src = base//'.f90'
        exe = base//'.ffc'
        ref = base//'.gf'
        ffc_out = base//'.ffc.out'
        ref_out = base//'.gf.out'

        options = compiler_frontend_options_t()
        options%run_semantics = .true.
        options%input_mode = INPUT_MODE_STANDARD
        call compile_frontend_from_string(source, frontend_result, options)
        if (.not. frontend_result%success()) then
            print *, 'FAIL[', trim(stem), ']: FortFront rejected source: ', &
                trim(frontend_result%diagnostic_text)
            return
        end if

        call lower_program_to_liric_exe(frontend_result%arena, &
            frontend_result%root_index, exe, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL[', trim(stem), ']: ffc lowering failed: ', trim(error_msg)
            return
        end if

        open (newunit=unit, file=src, status='replace', action='write')
        write (unit, '(A)') source
        close (unit)
        call execute_command_line('gfortran -w '//src//' -o '//ref, &
            exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL[', trim(stem), ']: gfortran rejected source'
            return
        end if

        call execute_command_line(exe//' > '//ffc_out//' 2>&1', exitstat=exit_stat)
        call execute_command_line(ref//' > '//ref_out//' 2>&1', exitstat=exit_stat)
        call execute_command_line('diff '//ffc_out//' '//ref_out// &
            ' > /dev/null 2>&1', exitstat=status)
        if (status /= 0) then
            print *, 'FAIL[', trim(stem), ']: ffc output differs from gfortran'
            call execute_command_line('diff '//ffc_out//' '//ref_out)
            return
        end if
        call execute_command_line('rm -f '//src//' '//exe//' '//ref// &
            ' '//ffc_out//' '//ref_out)
        matches_gfortran = .true.
    end function matches_gfortran

end program test_session_array_function_rank34_compiler
