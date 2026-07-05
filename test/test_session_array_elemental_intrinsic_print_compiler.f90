program test_session_array_elemental_intrinsic_print_compiler
    ! Verify list-directed print of a whole-array elemental intrinsic expression
    ! (print *, sin(x) for an array x) materialises the elementwise result and
    ! prints byte-for-byte like gfortran, instead of leaking uninitialised
    ! memory. Covers single and multi-argument real intrinsics in both real
    ! kinds plus an intrinsic embedded in whole-array arithmetic.
    use session_program_lowering, only: lower_program_to_liric_exe
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, &
        compile_frontend_from_string, &
        INPUT_MODE_STANDARD
    implicit none

    logical :: all_passed

    print *, '=== direct session array elemental-intrinsic print test ==='

    all_passed = .true.

    ! double-precision sin over a whole array
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  implicit none'//new_line('a')// &
        '  real(8) :: x(3)'//new_line('a')// &
        '  x = [1.0d0, 2.0d0, 3.0d0]'//new_line('a')// &
        '  print *, sin(x)'//new_line('a')// &
        'end program main', &
        'f64_sin')) all_passed = .false.

    ! several double-precision hyperbolic/trig intrinsics
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  implicit none'//new_line('a')// &
        '  double precision :: x(4)'//new_line('a')// &
        '  x = [1.0d0, 2.0d0, 3.0d0, 4.0d0]'//new_line('a')// &
        '  print *, tan(x)'//new_line('a')// &
        '  print *, cosh(x)'//new_line('a')// &
        '  print *, tanh(x)'//new_line('a')// &
        'end program main', &
        'f64_hyperbolic')) all_passed = .false.

    ! single-precision intrinsics over a whole array
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  implicit none'//new_line('a')// &
        '  real :: x(3)'//new_line('a')// &
        '  x = [1.0, 4.0, 9.0]'//new_line('a')// &
        '  print *, sqrt(x)'//new_line('a')// &
        '  print *, exp(x)'//new_line('a')// &
        'end program main', &
        'f32_sqrt_exp')) all_passed = .false.

    ! intrinsic embedded in whole-array arithmetic
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  implicit none'//new_line('a')// &
        '  real(8) :: x(3)'//new_line('a')// &
        '  x = [0.5d0, 1.0d0, 1.5d0]'//new_line('a')// &
        '  print *, cos(x) + 1.0d0'//new_line('a')// &
        'end program main', &
        'f64_cos_plus')) all_passed = .false.

    ! array intrinsic among other list-directed items
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  implicit none'//new_line('a')// &
        '  real(8) :: x(3)'//new_line('a')// &
        '  x = [1.0d0, 2.0d0, 3.0d0]'//new_line('a')// &
        '  print *, "s:", sin(x)'//new_line('a')// &
        'end program main', &
        'f64_mixed_items')) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: whole-array elemental-intrinsic print lowers through '// &
        'direct LIRIC session'

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
        base = '/tmp/ffc_arrelem_'//trim(stem)
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

end program test_session_array_elemental_intrinsic_print_compiler
