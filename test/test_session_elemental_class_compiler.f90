program test_session_elemental_class_compiler
    ! Verify that an elemental function with a scalar polymorphic dummy keeps
    ! its declared-type contract: a scalar call dispatches normally and an
    ! array call applies the scalar operation elementwise at the actual
    ! argument's own element stride, printing byte-for-byte like gfortran
    ! (#369). Also verify that an actual whose declared type is unrelated to
    ! the dummy's declared type is rejected with a source diagnostic.
    use session_program_lowering, only: lower_program_to_liric_exe
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, &
        compile_frontend_from_string, &
        INPUT_MODE_STANDARD
    implicit none

    logical :: all_passed

    print *, '=== direct session elemental class-dummy test ==='

    all_passed = .true.

    ! Scalar and whole-array calls of an elemental function whose dummy is
    ! class(base_t), with an extending type as the actual argument.
    if (.not. matches_gfortran( &
        polymorphic_module()// &
        'program main'//new_line('a')// &
        '  use m'//new_line('a')// &
        '  implicit none'//new_line('a')// &
        '  type(derived_t) :: d'//new_line('a')// &
        '  type(derived_t) :: arr(3)'//new_line('a')// &
        '  integer :: out(3)'//new_line('a')// &
        '  integer :: i'//new_line('a')// &
        '  d%v = 5'//new_line('a')// &
        '  print *, twice(d)'//new_line('a')// &
        '  do i = 1, 3'//new_line('a')// &
        '    arr(i)%v = i'//new_line('a')// &
        '    arr(i)%w = 10 * i'//new_line('a')// &
        '  end do'//new_line('a')// &
        '  out = twice(arr)'//new_line('a')// &
        '  print *, out'//new_line('a')// &
        'end program main', &
        'extension')) all_passed = .false.

    ! The same elemental function applied to an array of its own declared type.
    if (.not. matches_gfortran( &
        polymorphic_module()// &
        'program main'//new_line('a')// &
        '  use m'//new_line('a')// &
        '  implicit none'//new_line('a')// &
        '  type(base_t) :: arr(4)'//new_line('a')// &
        '  integer :: out(4)'//new_line('a')// &
        '  integer :: i'//new_line('a')// &
        '  do i = 1, 4'//new_line('a')// &
        '    arr(i)%v = 2 * i'//new_line('a')// &
        '  end do'//new_line('a')// &
        '  out = twice(arr)'//new_line('a')// &
        '  print *, out'//new_line('a')// &
        'end program main', &
        'declared')) all_passed = .false.

    ! A NOPASS elemental binding selected through a CLASS allocatable still
    ! dispatches through the concrete vtable when its array result is used as
    ! an ALLOCATE(SOURCE=...) expression.
    if (.not. matches_gfortran( &
        type_bound_elemental_source()// &
        'program main'//new_line('a')// &
        '  use a'//new_line('a')// &
        '  implicit none'//new_line('a')// &
        '  integer, allocatable :: vec(:)'//new_line('a')// &
        '  class(base), allocatable :: instance'//new_line('a')// &
        '  allocate(derived :: instance)'//new_line('a')// &
        '  allocate(vec, source=instance%add([1, 2], [1, 2]))'//new_line('a')// &
        '  print *, vec'//new_line('a')// &
        'end program main', &
        'type_bound_source')) all_passed = .false.

    ! Negative: an actual of an unrelated declared type must be diagnosed.
    if (.not. rejects( &
        polymorphic_module()// &
        'program main'//new_line('a')// &
        '  use m'//new_line('a')// &
        '  implicit none'//new_line('a')// &
        '  type(other_t) :: arr(3)'//new_line('a')// &
        '  integer :: out(3)'//new_line('a')// &
        '  arr(1)%v = 1'//new_line('a')// &
        '  out = twice(arr)'//new_line('a')// &
        '  print *, out'//new_line('a')// &
        'end program main', &
        'mismatch')) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: elemental calls with a scalar class dummy lower through '// &
        'direct LIRIC session'

contains

    function polymorphic_module() result(text)
        character(len=:), allocatable :: text

        text = 'module m'//new_line('a')// &
               '  implicit none'//new_line('a')// &
               '  type :: base_t'//new_line('a')// &
               '    integer :: v = 0'//new_line('a')// &
               '  end type base_t'//new_line('a')// &
               '  type, extends(base_t) :: derived_t'//new_line('a')// &
               '    integer :: w = 0'//new_line('a')// &
               '  end type derived_t'//new_line('a')// &
               '  type :: other_t'//new_line('a')// &
               '    integer :: v = 0'//new_line('a')// &
               '  end type other_t'//new_line('a')// &
               'contains'//new_line('a')// &
               '  elemental function twice(b) result(r)'//new_line('a')// &
               '    class(base_t), intent(in) :: b'//new_line('a')// &
               '    integer :: r'//new_line('a')// &
               '    r = 2 * b%v'//new_line('a')// &
               '  end function twice'//new_line('a')// &
               'end module m'//new_line('a')
    end function polymorphic_module

    function type_bound_elemental_source() result(text)
        character(len=:), allocatable :: text

        text = 'module a'//new_line('a')// &
               '  type, abstract :: base'//new_line('a')// &
               '  contains'//new_line('a')// &
               '    procedure(elem_func), deferred, nopass :: add'//new_line('a')// &
               '  end type base'//new_line('a')// &
               '  type, extends(base) :: derived'//new_line('a')// &
               '  contains'//new_line('a')// &
               '    procedure, nopass :: add => add_derived'//new_line('a')// &
               '  end type derived'//new_line('a')// &
               '  abstract interface'//new_line('a')// &
               '    elemental function elem_func(x, y) result(out)'//new_line('a')// &
               '      integer, intent(in) :: x, y'//new_line('a')// &
               '      integer :: out'//new_line('a')// &
               '    end function elem_func'//new_line('a')// &
               '  end interface'//new_line('a')// &
               'contains'//new_line('a')// &
               '  elemental function add_derived(x, y) result(out)'//new_line('a')// &
               '    integer, intent(in) :: x, y'//new_line('a')// &
               '    integer :: out'//new_line('a')// &
               '    out = x + y'//new_line('a')// &
               '  end function add_derived'//new_line('a')// &
               'end module a'//new_line('a')
    end function type_bound_elemental_source

    logical function rejects(source, stem)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: stem
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result
        character(len=:), allocatable :: error_msg

        rejects = .false.
        options = compiler_frontend_options_t()
        options%run_semantics = .true.
        options%input_mode = INPUT_MODE_STANDARD
        call compile_frontend_from_string(source, frontend_result, options)
        if (.not. frontend_result%success()) then
            rejects = .true.
            return
        end if
        call lower_program_to_liric_exe(frontend_result%arena, &
            frontend_result%root_index, '/tmp/ffc_elemclass_'//stem, error_msg)
        if (len_trim(error_msg) == 0) then
            print *, 'FAIL[', trim(stem), ']: incompatible declared type was '// &
                'accepted without a diagnostic'
            return
        end if
        rejects = .true.
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
        base = '/tmp/ffc_elemclass_'//trim(stem)
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
            ' '//ffc_out//' '//ref_out//' '//base//'.mod m.mod')
        matches_gfortran = .true.
    end function matches_gfortran

end program test_session_elemental_class_compiler
