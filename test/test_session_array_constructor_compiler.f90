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

    ! Named constant with dimension(*): extent comes from the array
    ! constructor initializer, not a caller's actual.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  integer, parameter :: offset(*) = [0, 1, 2, 3, 4]'//new_line('a')// &
        '  print *, offset'//new_line('a')// &
        '  print *, size(offset)'//new_line('a')// &
        'end program main', 'ctor_param_star_int')) all_passed = .false.

    ! Same feature with a real named constant.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  real, parameter :: a(*) = [1.1, 3.0, 10.0, 2.1, 5.5]'//new_line('a')// &
        '  print *, a'//new_line('a')// &
        'end program main', 'ctor_param_star_real')) all_passed = .false.

    ! Empty typed constructor: zero elements, zero extent.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  integer :: a(0)'//new_line('a')// &
        '  a = [ integer :: ]'//new_line('a')// &
        '  print *, size(a)'//new_line('a')// &
        'end program main', 'ctor_empty')) all_passed = .false.

    ! Nested constructors flatten in array element order.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  integer :: a(4)'//new_line('a')// &
        '  a = [ 1, [2, [3, 4]] ]'//new_line('a')// &
        '  print *, a'//new_line('a')// &
        'end program main', 'ctor_nested')) all_passed = .false.

    ! Complex constructor keeps each element's real and imaginary parts.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  complex :: a(3)'//new_line('a')// &
        '  a = (/ complex :: (1.0,2.0), (3.0,4.0), (5.0,6.0) /)'//new_line('a')// &
        '  print *, a(1)'//new_line('a')// &
        '  print *, a(3)'//new_line('a')// &
        'end program main', 'ctor_complex')) all_passed = .false.

    ! Nondefault integer kind keeps the declared element kind.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  integer(8) :: a(3)'//new_line('a')// &
        '  a = (/ integer(8) :: 1, 2, 3 /)'//new_line('a')// &
        '  print *, a'//new_line('a')// &
        'end program main', 'ctor_int8')) all_passed = .false.

    ! Nondefault real kind promotes integer elements to real(8).
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  real(8) :: a(3)'//new_line('a')// &
        '  a = [ real(8) :: 1, 2.5, 3 ]'//new_line('a')// &
        '  print *, a'//new_line('a')// &
        'end program main', 'ctor_real8')) all_passed = .false.

    ! Nested implied-DO evaluates the inner control fastest.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  integer :: i, j'//new_line('a')// &
        '  integer :: a(6)'//new_line('a')// &
        '  a = [ ((i*10+j, j=1,3), i=1,2) ]'//new_line('a')// &
        '  print *, a'//new_line('a')// &
        'end program main', 'ctor_nested_implied')) all_passed = .false.

    ! Negative control: an untyped constructor mixing integer and real
    ! elements has no single element type, so it must be diagnosed rather
    ! than silently truncated.
    if (.not. rejects( &
        'program main'//new_line('a')// &
        '  integer :: a(3)'//new_line('a')// &
        '  a = [1, 2.5, 3]'//new_line('a')// &
        '  print *, a'//new_line('a')// &
        'end program main', 'ctor_mixed_int_real')) all_passed = .false.

    ! Negative control: real constructor with an integer element.
    if (.not. rejects( &
        'program main'//new_line('a')// &
        '  real :: a(3)'//new_line('a')// &
        '  a = [1.5, 2, 3.5]'//new_line('a')// &
        '  print *, a'//new_line('a')// &
        'end program main', 'ctor_mixed_real_int')) all_passed = .false.

    ! Negative control: logical element among numeric elements.
    if (.not. rejects( &
        'program main'//new_line('a')// &
        '  integer :: a(3)'//new_line('a')// &
        '  a = [1, 2, .true.]'//new_line('a')// &
        '  print *, a'//new_line('a')// &
        'end program main', 'ctor_mixed_logical')) all_passed = .false.

    ! Negative control: element count differs from the declared extent.
    if (.not. rejects( &
        'program main'//new_line('a')// &
        '  integer :: a(3)'//new_line('a')// &
        '  a = [1, 2, 3, 4]'//new_line('a')// &
        '  print *, a'//new_line('a')// &
        'end program main', 'ctor_count_mismatch')) all_passed = .false.

    ! Negative control: an implied-DO with a zero step has no element count.
    if (.not. rejects( &
        'program main'//new_line('a')// &
        '  integer :: i, a(3)'//new_line('a')// &
        '  a = [(i, i=1,3,0)]'//new_line('a')// &
        '  print *, a'//new_line('a')// &
        'end program main', 'ctor_zero_step')) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: array constructors lower through direct LIRIC session'

contains

    ! An invalid array constructor must be reported by the frontend or by
    ! ffc lowering; producing an executable at all is the failure.
    logical function rejects(source, stem)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: stem
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result
        character(len=:), allocatable :: error_msg, exe

        rejects = .false.
        exe = '/tmp/ffc_ctor_'//stem//'.ffc'
        options = compiler_frontend_options_t()
        options%run_semantics = .true.
        options%input_mode = INPUT_MODE_STANDARD
        call compile_frontend_from_string(source, frontend_result, options)
        if (.not. frontend_result%success()) then
            rejects = .true.
            return
        end if
        call lower_program_to_liric_exe(frontend_result%arena, &
            frontend_result%root_index, exe, error_msg)
        if (len_trim(error_msg) > 0) then
            rejects = .true.
        else
            print *, 'FAIL[', stem, ']: invalid array constructor accepted'
        end if
        call execute_command_line('rm -f '//exe)
    end function rejects

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
