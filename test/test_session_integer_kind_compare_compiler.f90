program test_session_integer_kind_compare_compiler
    ! Verify integer(1)/(2)/(8) comparisons in an `if` condition lower to a
    ! same-width icmp (rather than falling through to the default integer(4)
    ! path, which rejects a non-default-kind identifier), and that the
    ! iso_c_binding size/pointer-difference kind names resolve to integer(8).
    use session_program_lowering, only: lower_program_to_liric_exe
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, &
        compile_frontend_from_string, &
        INPUT_MODE_STANDARD
    implicit none

    logical :: all_passed

    print *, '=== direct session non-default integer kind compare compiler test ==='

    all_passed = .true.

    ! integer(8) identifier compared against a kind-suffixed literal.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  integer(8) :: c'//new_line('a')// &
        '  c = 9000000000_8'//new_line('a')// &
        '  c = c + 1'//new_line('a')// &
        '  if (c == 9000000001_8) print *, "c ok"'//new_line('a')// &
        '  if (c > 0_8) print *, "c positive"'//new_line('a')// &
        'end program main', &
        'i64_compare')) all_passed = .false.

    ! integer(2) identifier compared in an if condition.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  integer(2) :: b'//new_line('a')// &
        '  b = 30000_2'//new_line('a')// &
        '  if (b > 50_2) print *, "b>50"'//new_line('a')// &
        '  if (b == 30000_2) print *, "b eq"'//new_line('a')// &
        'end program main', &
        'i16_compare')) all_passed = .false.

    ! integer(1) identifier compared in an if condition.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  integer(1) :: a'//new_line('a')// &
        '  a = 100_1'//new_line('a')// &
        '  if (a > 50_1) print *, "a>50"'//new_line('a')// &
        'end program main', &
        'i8_compare')) all_passed = .false.

    ! iso_c_binding c_size_t resolves to integer(8).
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  use, intrinsic :: iso_c_binding, only: c_size_t'//new_line('a')// &
        '  integer(c_size_t) :: n'//new_line('a')// &
        '  n = 5000000000_c_size_t'//new_line('a')// &
        '  print *, n'//new_line('a')// &
        'end program main', &
        'c_size_t_kind')) all_passed = .false.

    ! real(8) via the conventional working-precision "wp" kind alias.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  use, intrinsic :: iso_fortran_env, only: wp => real64'//new_line('a')// &
        '  real(wp) :: x'//new_line('a')// &
        '  x = 1.0_wp / 3.0_wp'//new_line('a')// &
        '  print *, x'//new_line('a')// &
        'end program main', &
        'wp_kind_alias')) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: non-default integer kind comparisons and kind aliases '// &
        'match gfortran'

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
        base = '/tmp/ffc_ikc_'//trim(stem)
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

end program test_session_integer_kind_compare_compiler
