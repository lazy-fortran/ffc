program test_session_allocatable_function_result_rank34_compiler
    ! Rank-3 and rank-4 allocatable function results through the descriptor-sret
    ! ABI. The source checks independent expected values, then the complete
    ! output is compared with gfortran.
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, compile_frontend_from_string, &
        INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    character(len=*), parameter :: source = &
        'program main'//new_line('a')// &
        '  implicit none'//new_line('a')// &
        '  integer, allocatable :: r3(:,:,:), r4(:,:,:,:)'//new_line('a')// &
        '  r3 = make3()'//new_line('a')// &
        '  r4 = make4()'//new_line('a')// &
        '  if (size(r3) /= 12 .or. size(r3,1) /= 2 .or. '// &
        'size(r3,2) /= 3 .or. size(r3,3) /= 2) error stop 1'//new_line('a')// &
        '  if (r3(2,1,2) /= 212 .or. r3(1,3,1) /= 131) error stop 2'// &
        new_line('a')// &
        '  if (size(r4) /= 24 .or. size(r4,1) /= 2 .or. '// &
        'size(r4,2) /= 2 .or. size(r4,3) /= 2 .or. '// &
        'size(r4,4) /= 3) error stop 3'//new_line('a')// &
        '  if (r4(2,2,1,3) /= 2213 .or. r4(1,2,2,1) /= 1221) '// &
        'error stop 4'//new_line('a')// &
        '  print *, size(r3), r3(2,1,2), r3(1,3,1)'//new_line('a')// &
        '  print *, size(r4), r4(2,2,1,3), r4(1,2,2,1)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  function make3() result(out)'//new_line('a')// &
        '    integer, allocatable :: out(:,:,:)'//new_line('a')// &
        '    integer :: i, j, k'//new_line('a')// &
        '    allocate(out(2,3,2))'//new_line('a')// &
        '    do k = 1, 2'//new_line('a')// &
        '      do j = 1, 3'//new_line('a')// &
        '        do i = 1, 2'//new_line('a')// &
        '          out(i,j,k) = 100*i + 10*j + k'//new_line('a')// &
        '        end do'//new_line('a')// &
        '      end do'//new_line('a')// &
        '    end do'//new_line('a')// &
        '  end function make3'//new_line('a')// &
        '  function make4() result(out)'//new_line('a')// &
        '    integer, allocatable :: out(:,:,:,:)'//new_line('a')// &
        '    integer :: i, j, k, l'//new_line('a')// &
        '    allocate(out(2,2,2,3))'//new_line('a')// &
        '    do l = 1, 3'//new_line('a')// &
        '      do k = 1, 2'//new_line('a')// &
        '        do j = 1, 2'//new_line('a')// &
        '          do i = 1, 2'//new_line('a')// &
        '            out(i,j,k,l) = 1000*i + 100*j + 10*k + l'// &
        new_line('a')// &
        '          end do'//new_line('a')// &
        '        end do'//new_line('a')// &
        '      end do'//new_line('a')// &
        '    end do'//new_line('a')// &
        '  end function make4'//new_line('a')// &
        'end program main'

    if (.not. matches_gfortran(source)) stop 1
    print *, 'PASS: rank-3 and rank-4 allocatable function results match gfortran'

contains

    logical function matches_gfortran(program_source)
        character(len=*), intent(in) :: program_source
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result
        character(len=:), allocatable :: error_msg
        character(len=:), allocatable :: base, src, exe, ref, ffc_out, ref_out
        integer :: unit, exit_stat, status

        matches_gfortran = .false.
        base = '/var/tmp/ert/ffc_alloc_function_result_rank34'
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
            print *, 'FAIL: FortFront rejected source: ', &
                trim(frontend_result%diagnostic_text)
            return
        end if
        call lower_program_to_liric_exe(frontend_result%arena, &
            frontend_result%root_index, exe, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: ffc lowering failed: ', trim(error_msg)
            return
        end if

        open (newunit=unit, file=src, status='replace', action='write')
        write (unit, '(A)') program_source
        close (unit)
        call execute_command_line('gfortran -w '//src//' -o '//ref, &
            exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: gfortran rejected source'
            return
        end if
        call execute_command_line(exe//' > '//ffc_out//' 2>&1', &
            exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: ffc executable failed'
            return
        end if
        call execute_command_line(ref//' > '//ref_out//' 2>&1', &
            exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: gfortran executable failed'
            return
        end if
        call execute_command_line('diff '//ffc_out//' '//ref_out// &
            ' > /dev/null 2>&1', exitstat=status)
        if (status /= 0) then
            print *, 'FAIL: ffc output differs from gfortran'
            call execute_command_line('diff '//ffc_out//' '//ref_out)
            return
        end if
        call execute_command_line('rm -f '//src//' '//exe//' '//ref//' '// &
            ffc_out//' '//ref_out)
        matches_gfortran = .true.
    end function matches_gfortran

end program test_session_allocatable_function_result_rank34_compiler
