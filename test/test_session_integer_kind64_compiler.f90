program test_session_integer_kind64_compiler
    ! Verify integer(8) declarations, assignment, arithmetic, and list-directed
    ! print match gfortran byte-for-byte (#246).
    use session_program_lowering, only: lower_program_to_liric_exe
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, &
        compile_frontend_from_string, &
        INPUT_MODE_STANDARD
    implicit none

    logical :: all_passed

    print *, '=== direct session integer(8) compiler test ==='

    all_passed = .true.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  integer(8) :: x'//new_line('a')// &
        '  x = 42_8'//new_line('a')// &
        '  print *, x'//new_line('a')// &
        'end program main', &
        'i64_basic')) all_passed = .false.

    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  integer(8) :: x, y'//new_line('a')// &
        '  x = 100_8'//new_line('a')// &
        '  y = 200_8'//new_line('a')// &
        '  print *, x + y'//new_line('a')// &
        'end program main', &
        'i64_add')) all_passed = .false.

    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  integer(8) :: x'//new_line('a')// &
        '  x = 9999999999_8'//new_line('a')// &
        '  print *, x'//new_line('a')// &
        'end program main', &
        'i64_large')) all_passed = .false.

    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  integer(8) :: x, y'//new_line('a')// &
        '  x = 3_8'//new_line('a')// &
        '  y = 4_8'//new_line('a')// &
        '  print *, x'//new_line('a')// &
        '  print *, y'//new_line('a')// &
        '  print *, x * y'//new_line('a')// &
        'end program main', &
        'i64_mul')) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: integer(8) lowers through direct LIRIC session'

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
        base = '/tmp/ffc_i64_'//trim(stem)
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

end program test_session_integer_kind64_compiler
