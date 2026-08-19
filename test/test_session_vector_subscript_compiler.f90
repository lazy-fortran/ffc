program test_session_vector_subscript_compiler
    ! A bounded read-gather oracle for fixed-size rank-1 intrinsic arrays.
    ! The positive behavior is compared byte-for-byte with the pinned
    ! /usr/bin/gfortran executable; the negative case checks the ffc runtime
    ! bounds diagnostic and that pinned gfortran also rejects the access with
    ! bounds checking enabled.
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, compile_frontend_from_string, &
        INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe
    use ffc_test_support, only: expect_stderr_and_exit
    implicit none

    logical :: all_passed

    print *, '=== fixed rank-1 vector-subscript gather compiler test ==='
    all_passed = .true.
    if (.not. matches_gfortran(valid_source(), 'vector_gather_reordered')) &
        all_passed = .false.
    if (.not. matches_gfortran(expression_source(), 'vector_gather_expression')) &
        all_passed = .false.
    if (.not. expect_stderr_and_exit(bounds_source(), &
        'Fortran runtime error: Vector subscript is out of bounds'//new_line('a'), &
        2, '/var/tmp/ert/ffc_vector_gather_bounds')) all_passed = .false.
    if (.not. gfortran_bounds_oracle()) all_passed = .false.
    if (.not. all_passed) stop 1
    print *, 'PASS: fixed rank-1 vector-subscript gather matches gfortran'

contains

    function valid_source() result(source)
        character(len=:), allocatable :: source

        source = 'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: src(-1:3), index_vector(4), out(4)'//new_line('a')// &
            '  src = [10, 20, 30, 40, 50]'//new_line('a')// &
            '  index_vector = [3, 0, 3, -1]'//new_line('a')// &
            '  out = src(index_vector)'//new_line('a')// &
            '  print *, out'//new_line('a')// &
            'end program main'//new_line('a')
    end function valid_source

    function expression_source() result(source)
        character(len=:), allocatable :: source

        source = 'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  real(8) :: src(5), out(3)'//new_line('a')// &
            '  integer :: index_vector(3)'//new_line('a')// &
            '  src = [1.0d0, 2.0d0, 3.0d0, 4.0d0, 5.0d0]'//new_line('a')// &
            '  index_vector = [4, 1, 4]'//new_line('a')// &
            '  out = src(index_vector) + 1.0d0'//new_line('a')// &
            '  print *, out'//new_line('a')// &
            'end program main'//new_line('a')
    end function expression_source

    function bounds_source() result(source)
        character(len=:), allocatable :: source

        source = 'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: src(3), index_vector(3), out(3)'//new_line('a')// &
            '  src = [10, 20, 30]'//new_line('a')// &
            '  index_vector = [1, 4, 2]'//new_line('a')// &
            '  out = src(index_vector)'//new_line('a')// &
            '  print *, out'//new_line('a')// &
            'end program main'//new_line('a')
    end function bounds_source

    logical function matches_gfortran(source, stem) result(ok)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: stem
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result
        character(len=:), allocatable :: error_msg
        character(len=:), allocatable :: base, src, ffc_exe, gfortran_exe
        character(len=:), allocatable :: ffc_out, gfortran_out
        integer :: unit, exit_stat, diff_status

        ok = .false.
        base = '/var/tmp/ert/ffc_vector_gather_'//trim(stem)
        src = base//'.f90'
        ffc_exe = base//'.ffc'
        gfortran_exe = base//'.gfortran'
        ffc_out = base//'.ffc.out'
        gfortran_out = base//'.gfortran.out'

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
            frontend_result%root_index, ffc_exe, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL[', trim(stem), ']: ffc lowering failed: ', &
                trim(error_msg)
            call cleanup(base)
            return
        end if

        open (newunit=unit, file=src, status='replace', action='write')
        write (unit, '(A)') source
        close (unit)
        call execute_command_line('/usr/bin/gfortran -std=f2018 -w '//src// &
            ' -o '//gfortran_exe, exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL[', trim(stem), ']: pinned gfortran rejected source'
            call cleanup(base)
            return
        end if

        call execute_command_line(ffc_exe//' > '//ffc_out//' 2>&1', &
            exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL[', trim(stem), ']: ffc executable failed'
            call cleanup(base)
            return
        end if
        call execute_command_line(gfortran_exe//' > '//gfortran_out//' 2>&1', &
            exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL[', trim(stem), ']: pinned gfortran executable failed'
            call cleanup(base)
            return
        end if
        call execute_command_line('diff -u '//ffc_out//' '//gfortran_out// &
            ' > /dev/null 2>&1', exitstat=diff_status)
        if (diff_status /= 0) then
            print *, 'FAIL[', trim(stem), ']: ffc output differs from gfortran'
            call execute_command_line('diff -u '//ffc_out//' '//gfortran_out)
            call cleanup(base)
            return
        end if
        call cleanup(base)
        ok = .true.
    end function matches_gfortran

    logical function gfortran_bounds_oracle() result(ok)
        character(len=*), parameter :: base = &
            '/var/tmp/ert/ffc_vector_gather_gfortran_bounds'
        character(len=*), parameter :: src = base//'.f90'
        character(len=*), parameter :: exe = base//'.exe'
        character(len=*), parameter :: out = base//'.out'
        character(len=:), allocatable :: source
        integer :: unit, exit_stat

        ok = .false.
        source = bounds_source()
        open (newunit=unit, file=src, status='replace', action='write')
        write (unit, '(A)') source
        close (unit)
        call execute_command_line('/usr/bin/gfortran -std=f2018 -w -fcheck=bounds '// &
            src//' -o '//exe, exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: pinned gfortran rejected bounds oracle source'
            call cleanup(base)
            return
        end if
        call execute_command_line(exe//' > '//out//' 2>&1', exitstat=exit_stat)
        if (exit_stat == 0) then
            print *, 'FAIL: pinned gfortran did not reject out-of-bounds gather'
            call cleanup(base)
            return
        end if
        call cleanup(base)
        ok = .true.
    end function gfortran_bounds_oracle

    subroutine cleanup(base)
        character(len=*), intent(in) :: base

        call execute_command_line('rm -f '//base//'*')
    end subroutine cleanup

end program test_session_vector_subscript_compiler
