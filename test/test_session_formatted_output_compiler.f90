program test_session_formatted_output_compiler
    ! Byte-for-byte checks of explicit-format write/print to stdout against
    ! gfortran: the I/F/ES/A/L/X edit descriptors, repeat counts, embedded
    ! string literals, the '/' record terminator, and format reversion.
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, &
        compile_frontend_from_string, INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    logical :: all_passed

    print *, '=== direct session formatted output compiler test ==='

    all_passed = .true.
    if (.not. test_int_descriptor()) all_passed = .false.
    if (.not. test_real_descriptor()) all_passed = .false.
    if (.not. test_es_descriptor()) all_passed = .false.
    if (.not. test_char_descriptor()) all_passed = .false.
    if (.not. test_logical_descriptor()) all_passed = .false.
    if (.not. test_x_descriptor()) all_passed = .false.
    if (.not. test_repeat_count()) all_passed = .false.
    if (.not. test_nested_group()) all_passed = .false.
    if (.not. test_f64_literal_precision()) all_passed = .false.
    if (.not. test_string_literal_and_slash()) all_passed = .false.
    if (.not. test_format_reversion()) all_passed = .false.
    if (.not. test_write_unit_star()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: formatted output matches gfortran across edit descriptors'

contains

    logical function test_int_descriptor()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  i = 42'//new_line('a')// &
            "  print '(I5)', i"//new_line('a')// &
            'end program main'
        test_int_descriptor = matches_gfortran(source, 'out_i')
    end function test_int_descriptor

    logical function test_real_descriptor()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: x'//new_line('a')// &
            '  x = 3.25'//new_line('a')// &
            "  print '(F8.3)', x"//new_line('a')// &
            'end program main'
        test_real_descriptor = matches_gfortran(source, 'out_f')
    end function test_real_descriptor

    logical function test_es_descriptor()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: x'//new_line('a')// &
            '  x = 3.14159'//new_line('a')// &
            "  print '(ES12.4)', x"//new_line('a')// &
            'end program main'
        test_es_descriptor = matches_gfortran(source, 'out_es')
    end function test_es_descriptor

    logical function test_char_descriptor()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            "  print '(A)', 'hello'"//new_line('a')// &
            'end program main'
        test_char_descriptor = matches_gfortran(source, 'out_a')
    end function test_char_descriptor

    logical function test_logical_descriptor()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  logical :: t, f'//new_line('a')// &
            '  t = .true.'//new_line('a')// &
            '  f = .false.'//new_line('a')// &
            "  print '(L3,L3)', t, f"//new_line('a')// &
            'end program main'
        test_logical_descriptor = matches_gfortran(source, 'out_l')
    end function test_logical_descriptor

    logical function test_x_descriptor()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a, b'//new_line('a')// &
            '  a = 1'//new_line('a')// &
            '  b = 2'//new_line('a')// &
            "  print '(I3,3X,I3)', a, b"//new_line('a')// &
            'end program main'
        test_x_descriptor = matches_gfortran(source, 'out_x')
    end function test_x_descriptor

    logical function test_repeat_count()
        ! 2I4 applies the I4 descriptor to two consecutive items.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a, b'//new_line('a')// &
            '  a = 1'//new_line('a')// &
            '  b = 2'//new_line('a')// &
            "  print '(2I4)', a, b"//new_line('a')// &
            'end program main'
        test_repeat_count = matches_gfortran(source, 'out_repeat')
    end function test_repeat_count

    logical function test_nested_group()
        ! A parenthesized group 2(f5.2,',') repeats its inner descriptor list,
        ! including the embedded doubled-quote literal, once per repeat.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: x, y'//new_line('a')// &
            '  x = 1.0'//new_line('a')// &
            '  y = 2.0'//new_line('a')// &
            "  write(*, '(2(f5.2,'',''))') x, y"//new_line('a')// &
            'end program main'
        test_nested_group = matches_gfortran(source, 'out_group')
    end function test_nested_group

    logical function test_f64_literal_precision()
        ! A real(8) literal (d0 exponent) printed through F/ES descriptors must
        ! keep double precision; the f32 path would round it to ~3.14159274.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            "  print '(F20.14)', 3.14159265358979d0"//new_line('a')// &
            "  print '(ES22.14)', 3.14159265358979d0"//new_line('a')// &
            'end program main'
        test_f64_literal_precision = matches_gfortran(source, 'out_f64lit')
    end function test_f64_literal_precision

    logical function test_string_literal_and_slash()
        ! Embedded string literal (with a doubled quote) and a '/' record break.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  i = 7'//new_line('a')// &
            "  print '(""I''m"",I3/""next"")', i"//new_line('a')// &
            'end program main'
        test_string_literal_and_slash = matches_gfortran(source, 'out_slash')
    end function test_string_literal_and_slash

    logical function test_format_reversion()
        ! Six items through a two-descriptor format produce three records.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a, b, c'//new_line('a')// &
            '  a = 1'//new_line('a')// &
            '  b = 2'//new_line('a')// &
            '  c = 3'//new_line('a')// &
            "  print '(A,I3)', 'X', a, 'Y', b, 'Z', c"//new_line('a')// &
            'end program main'
        test_format_reversion = matches_gfortran(source, 'out_revert')
    end function test_format_reversion

    logical function test_write_unit_star()
        ! write(*, fmt) shares the print lowering path.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  i = 99'//new_line('a')// &
            "  write(*, '(I6)') i"//new_line('a')// &
            'end program main'
        test_write_unit_star = matches_gfortran(source, 'out_write')
    end function test_write_unit_star

    logical function matches_gfortran(source, stem)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: stem
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result
        character(len=:), allocatable :: error_msg
        character(len=:), allocatable :: base, src, exe, ref, ffc_out, ref_out
        integer :: unit, exit_stat, status

        matches_gfortran = .false.
        base = '/tmp/ffc_'//stem
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
            print *, 'FAIL[', stem, ']: FortFront rejected source: ', &
                trim(frontend_result%diagnostic_text)
            return
        end if

        call lower_program_to_liric_exe(frontend_result%arena, &
            frontend_result%root_index, exe, &
            error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL[', stem, ']: ffc lowering failed: ', trim(error_msg)
            return
        end if

        open (newunit=unit, file=src, status='replace', action='write')
        write (unit, '(A)') source
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

end program test_session_formatted_output_compiler
