program test_session_integer_kind_i8_i16_compiler
    ! Verify integer(1)/i8 and integer(2)/i16 declarations, assignment,
    ! arithmetic, and list-directed print match gfortran byte-for-byte (#246).
    use session_program_lowering, only: lower_program_to_liric_exe
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, &
        compile_frontend_from_string, &
        INPUT_MODE_STANDARD
    implicit none

    logical :: all_passed

    print *, '=== direct session integer(1)/integer(2) compiler test ==='

    all_passed = .true.

    ! integer(1) basic
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  integer(1) :: x'//new_line('a')// &
        '  x = 42_1'//new_line('a')// &
        '  print *, x'//new_line('a')// &
        'end program main', &
        'i8_basic')) all_passed = .false.

    ! integer(1) arithmetic
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  integer(1) :: x, y'//new_line('a')// &
        '  x = 10_1'//new_line('a')// &
        '  y = 20_1'//new_line('a')// &
        '  print *, x + y'//new_line('a')// &
        'end program main', &
        'i8_add')) all_passed = .false.

    ! integer(1) negative value
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  integer(1) :: x'//new_line('a')// &
        '  x = -5_1'//new_line('a')// &
        '  print *, x'//new_line('a')// &
        'end program main', &
        'i8_neg')) all_passed = .false.

    ! integer(2) basic
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  integer(2) :: x'//new_line('a')// &
        '  x = 1000_2'//new_line('a')// &
        '  print *, x'//new_line('a')// &
        'end program main', &
        'i16_basic')) all_passed = .false.

    ! integer(2) arithmetic
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  integer(2) :: x, y'//new_line('a')// &
        '  x = 100_2'//new_line('a')// &
        '  y = 200_2'//new_line('a')// &
        '  print *, x + y'//new_line('a')// &
        'end program main', &
        'i16_add')) all_passed = .false.

    ! integer(2) multiple prints
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  integer(2) :: a, b'//new_line('a')// &
        '  a = 3_2'//new_line('a')// &
        '  b = 4_2'//new_line('a')// &
        '  print *, a'//new_line('a')// &
        '  print *, b'//new_line('a')// &
        '  print *, a * b'//new_line('a')// &
        'end program main', &
        'i16_mul')) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: integer(1)/integer(2) lower through direct LIRIC session'

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
        base = '/tmp/ffc_i8i16_'//trim(stem)
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

end program test_session_integer_kind_i8_i16_compiler
