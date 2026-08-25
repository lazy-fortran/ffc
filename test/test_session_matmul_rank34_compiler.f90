program test_session_matmul_rank34_compiler
    ! Fixed-shape MATMUL function results whose SIZE bounds refer to rank-3
    ! and rank-4 explicit-shape actual arguments.
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, compile_frontend_from_string, &
        INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    logical :: all_passed

    print *, '=== direct session MATMUL rank-3/rank-4 actual test ==='
    all_passed = .true.
    if (.not. test_rank3_actual()) all_passed = .false.
    if (.not. test_rank4_actual()) all_passed = .false.
    if (.not. all_passed) stop 1
    print *, 'PASS: fixed-shape MATMUL results use rank-3/rank-4 SIZE bounds'

contains

    logical function test_rank3_actual()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(2,3,2), b(3,2), c(2,2)'//new_line('a')// &
            '  integer :: i, j, k'//new_line('a')// &
            '  do k = 1, 2'//new_line('a')// &
            '    do j = 1, 3'//new_line('a')// &
            '      do i = 1, 2'//new_line('a')// &
            '        a(i,j,k) = 100*k + 10*j + i'//new_line('a')// &
            '      end do'//new_line('a')// &
            '    end do'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  b = reshape([1, 3, 5, 2, 4, 6], [3, 2])'//new_line('a')// &
            '  c = matmul(make_left(a), b)'//new_line('a')// &
            '  print *, c(1,1), c(2,1), c(1,2), c(2,2)'//new_line('a')// &
            '  if (c(1,1) /= 112 .or. c(2,1) /= 202) error stop 1'// &
            new_line('a')// &
            '  if (c(1,2) /= 148 .or. c(2,2) /= 268) error stop 2'// &
            new_line('a')// &
            '  print *, c(1,1), c(2,1), c(1,2), c(2,2)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  function make_left(x) result(r)'//new_line('a')// &
            '    integer :: x(2,3,2)'//new_line('a')// &
            '    integer :: r(size(x,1),size(x,2))'//new_line('a')// &
            '    integer :: i, j'//new_line('a')// &
            '    do j = 1, size(x,2)'//new_line('a')// &
            '      do i = 1, size(x,1)'//new_line('a')// &
            '        r(i,j) = 10*i + j'//new_line('a')// &
            '      end do'//new_line('a')// &
            '    end do'//new_line('a')// &
            '  end function make_left'//new_line('a')// &
            'end program main'

        test_rank3_actual = matches_gfortran(source, &
            '/var/tmp/ert/ffc_session_matmul_rank3_test')
    end function test_rank3_actual

    logical function test_rank4_actual()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(2,3,2,2), b(3,2), c(2,2)'//new_line('a')// &
            '  integer :: i, j, k, l'//new_line('a')// &
            '  do l = 1, 2'//new_line('a')// &
            '    do k = 1, 2'//new_line('a')// &
            '      do j = 1, 3'//new_line('a')// &
            '        do i = 1, 2'//new_line('a')// &
            '          a(i,j,k,l) = 1000*l + 100*k + 10*j + i'// &
            new_line('a')// &
            '        end do'//new_line('a')// &
            '      end do'//new_line('a')// &
            '    end do'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  b = reshape([1, 3, 5, 2, 4, 6], [3, 2])'//new_line('a')// &
            '  c = matmul(make_left(a), b)'//new_line('a')// &
            '  print *, c(1,1), c(2,1), c(1,2), c(2,2)'//new_line('a')// &
            '  if (c(1,1) /= 922 .or. c(2,1) /= 1822) error stop 1'// &
            new_line('a')// &
            '  if (c(1,2) /= 1228 .or. c(2,2) /= 2428) error stop 2'// &
            new_line('a')// &
            '  print *, c(1,1), c(2,1), c(1,2), c(2,2)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  function make_left(x) result(r)'//new_line('a')// &
            '    integer :: x(2,3,2,2)'//new_line('a')// &
            '    integer :: r(size(x,1),size(x,2))'//new_line('a')// &
            '    integer :: i, j'//new_line('a')// &
            '    do j = 1, size(x,2)'//new_line('a')// &
            '      do i = 1, size(x,1)'//new_line('a')// &
            '        r(i,j) = 100*i + j'//new_line('a')// &
            '      end do'//new_line('a')// &
            '    end do'//new_line('a')// &
            '  end function make_left'//new_line('a')// &
            'end program main'

        test_rank4_actual = matches_gfortran(source, &
            '/var/tmp/ert/ffc_session_matmul_rank4_test')
    end function test_rank4_actual

    logical function matches_gfortran(source, base)
        character(len=*), intent(in) :: source, base
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result
        character(len=:), allocatable :: error_msg
        character(len=:), allocatable :: src, exe, ref, ffc_out, ref_out
        integer :: unit, exit_stat, status

        matches_gfortran = .false.
        src = trim(base)//'.f90'
        exe = trim(base)//'.ffc'
        ref = trim(base)//'.gf'
        ffc_out = trim(base)//'.ffc.out'
        ref_out = trim(base)//'.gf.out'
        options = compiler_frontend_options_t()
        options%run_semantics = .true.
        options%input_mode = INPUT_MODE_STANDARD
        call compile_frontend_from_string(source, frontend_result, options)
        if (.not. frontend_result%success()) then
            print *, 'FAIL[', trim(base), ']: FortFront rejected source: ', &
                trim(frontend_result%diagnostic_text)
            return
        end if
        call lower_program_to_liric_exe(frontend_result%arena, &
            frontend_result%root_index, exe, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL[', trim(base), ']: ffc lowering failed: ', trim(error_msg)
            return
        end if
        open (newunit=unit, file=src, status='replace', action='write')
        write (unit, '(A)') source
        close (unit)
        call execute_command_line('gfortran -w '//src//' -o '//ref, &
            exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL[', trim(base), ']: gfortran rejected source'
            return
        end if
        call execute_command_line(exe//' > '//ffc_out//' 2>&1', &
            exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL[', trim(base), ']: ffc executable failed'
            return
        end if
        call execute_command_line(ref//' > '//ref_out//' 2>&1', &
            exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL[', trim(base), ']: gfortran executable failed'
            return
        end if
        call execute_command_line('diff '//ffc_out//' '//ref_out// &
            ' > /dev/null 2>&1', exitstat=status)
        if (status /= 0) then
            print *, 'FAIL[', trim(base), ']: ffc output differs from gfortran'
            call execute_command_line('diff '//ffc_out//' '//ref_out)
            return
        end if
        call execute_command_line('rm -f '//src//' '//exe//' '//ref//' '// &
            ffc_out//' '//ref_out)
        matches_gfortran = .true.
    end function matches_gfortran

end program test_session_matmul_rank34_compiler
