program test_session_array_constructor_compiler
    use fortfront_compiler, only: compiler_frontend_options_t, &
                                  compiler_frontend_result_t, &
                                  compile_frontend_from_string, INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    logical :: all_passed

    print *, '=== direct session array constructor compiler test ==='

    all_passed = .true.

    ! Typed integer constructor with real literals: int() truncation per element.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  integer :: a(3)'//new_line('a')// &
        '  a = (/ integer :: 1.5, 2.7, 3.9 /)'//new_line('a')// &
        '  print *, a'//new_line('a')// &
        'end program main', 'ctor_typed_int')) all_passed = .false.

    ! Typed real constructor with integer literals: promote each to real.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  real :: a(5)'//new_line('a')// &
        '  a = (/ real :: 1, 2, 3, 4, 5 /)'//new_line('a')// &
        '  print *, a'//new_line('a')// &
        'end program main', 'ctor_typed_real')) all_passed = .false.

    ! Implied-do integer constructor folds the body for each loop index.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  integer :: i'//new_line('a')// &
        '  integer :: a(5)'//new_line('a')// &
        '  a = [(i*i, i=1, 5)]'//new_line('a')// &
        '  print *, a'//new_line('a')// &
        'end program main', 'ctor_implied_int')) all_passed = .false.

    ! Implied-do real constructor emits a real body expression per index.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  integer :: i'//new_line('a')// &
        '  real :: a(5)'//new_line('a')// &
        '  a = [(real(i) / 2.0, i=1, 5)]'//new_line('a')// &
        '  print *, a'//new_line('a')// &
        'end program main', 'ctor_implied_real')) all_passed = .false.

    ! Whole array among other print items prints each element inline.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  integer :: i'//new_line('a')// &
        '  integer :: a(4)'//new_line('a')// &
        '  a = [(i, i=1, 4)]'//new_line('a')// &
        '  print *, "vals:", a'//new_line('a')// &
        'end program main', 'ctor_print_tag')) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: array constructors lower through direct LIRIC session'

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
        base = '/tmp/ffc_ctor_'//stem
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
                                        frontend_result%root_index, exe, error_msg)
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

end program test_session_array_constructor_compiler
