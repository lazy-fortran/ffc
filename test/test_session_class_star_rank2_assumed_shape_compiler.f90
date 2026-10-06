! fo: dispatcher
module ffc_case_test_session_class_star_rank2_assumed_shape_compiler
    implicit none
    private
    public :: case_test_session_class_star_rank2_assumed_shape_compiler
    interface
        subroutine case_test_session_class_star_rank2_assumed_shape_compiler()
        end subroutine case_test_session_class_star_rank2_assumed_shape_compiler
    end interface
end module ffc_case_test_session_class_star_rank2_assumed_shape_compiler

subroutine case_test_session_class_star_rank2_assumed_shape_compiler()
    ! Differential oracle for the bounded rank-1 through rank-4 CLASS(*) array ABI.
    ! Positive cases compare ffc output with gfortran; refusal cases first pass
    ! gfortran syntax checking and then require a named ffc diagnostic.
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, compile_frontend_from_string, &
        INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none
    save

    print *, '=== CLASS(*) assumed-shape array test ==='
    if (.not. test_integer_rank2()) stop 1
    if (.not. test_real8_rank2()) stop 1
    if (.not. test_real4_constructor()) stop 1
    if (.not. test_real_kind_dispatch()) stop 1
    if (.not. test_real4_rank3_rank4()) stop 1
    if (.not. test_integer_rank3()) stop 1
    if (.not. test_integer_rank4()) stop 1
    if (.not. test_refusal_contract()) stop 1
    print *, 'PASS: CLASS(*) arrays preserve shape, real kind, and narrowing'

contains

    logical function test_integer_rank2()
        character(len=*), parameter :: source = &
            'module rank2_integer_m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine inspect(values)'//new_line('a')// &
            '    class(*), intent(in) :: values(:,:)'//new_line('a')// &
            '    select type (items => values)'//new_line('a')// &
            '    type is (integer)'//new_line('a')// &
            '      print *, size(items,1), size(items,2), size(items)'// &
            new_line('a')// &
            '      print *, items(1,1), items(2,1), items(1,3), items(2,3)'// &
            new_line('a')// &
            '    class default'//new_line('a')// &
            '      print *, -1'//new_line('a')// &
            '    end select'//new_line('a')// &
            '  end subroutine inspect'//new_line('a')// &
            'end module rank2_integer_m'//new_line('a')// &
            'program rank2_integer_main'//new_line('a')// &
            '  use rank2_integer_m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: values(2,3)'//new_line('a')// &
            '  values(1,1) = 11'//new_line('a')// &
            '  values(2,1) = 21'//new_line('a')// &
            '  values(1,2) = 12'//new_line('a')// &
            '  values(2,2) = 22'//new_line('a')// &
            '  values(1,3) = 13'//new_line('a')// &
            '  values(2,3) = 23'//new_line('a')// &
            '  call inspect(values)'//new_line('a')// &
            'end program rank2_integer_main'

        test_integer_rank2 = run_differential(source, &
            '/var/tmp/ert/ffc_class_star_rank2_integer')
    end function test_integer_rank2

    logical function test_real8_rank2()
        character(len=*), parameter :: source = &
            'module rank2_real8_m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine inspect(values)'//new_line('a')// &
            '    class(*), intent(in) :: values(:,:)'//new_line('a')// &
            '    select type (items => values)'//new_line('a')// &
            '    type is (real(8))'//new_line('a')// &
            '      print *, size(items,1), size(items,2)'//new_line('a')// &
            '      print *, items(1,1), items(2,2), items(1,3)'//new_line('a')// &
            '    class default'//new_line('a')// &
            '      print *, -1.0d0'//new_line('a')// &
            '    end select'//new_line('a')// &
            '  end subroutine inspect'//new_line('a')// &
            'end module rank2_real8_m'//new_line('a')// &
            'program rank2_real8_main'//new_line('a')// &
            '  use rank2_real8_m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  real(8) :: values(2,3)'//new_line('a')// &
            '  values(1,1) = 1.25d0'//new_line('a')// &
            '  values(2,1) = -2.5d0'//new_line('a')// &
            '  values(1,2) = 3.75d0'//new_line('a')// &
            '  values(2,2) = 4.5d0'//new_line('a')// &
            '  values(1,3) = 5.0d0'//new_line('a')// &
            '  values(2,3) = 6.25d0'//new_line('a')// &
            '  call inspect(values)'//new_line('a')// &
            'end program rank2_real8_main'

        test_real8_rank2 = run_differential(source, &
            '/var/tmp/ert/ffc_class_star_rank2_real8')
    end function test_real8_rank2

    logical function test_real4_constructor()
        character(len=*), parameter :: source = &
            'program real4_constructor_main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  call inspect([real :: 1.0, 2.0, 3.0])'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine inspect(values)'//new_line('a')// &
            '    class(*), intent(in) :: values(:)'//new_line('a')// &
            '    select type (items => values)'//new_line('a')// &
            '    type is (real)'//new_line('a')// &
            '      print *, size(items), items(1), items(2), items(3)'// &
            new_line('a')// &
            '    class default'//new_line('a')// &
            '      print *, -1'//new_line('a')// &
            '    end select'//new_line('a')// &
            '  end subroutine inspect'//new_line('a')// &
            'end program real4_constructor_main'

        test_real4_constructor = run_differential(source, &
            '/var/tmp/ert/ffc_class_star_real4_constructor')
    end function test_real4_constructor

    logical function test_real_kind_dispatch()
        character(len=*), parameter :: source = &
            'program real_kind_dispatch_main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  real :: single(2,2)'//new_line('a')// &
            '  real(8) :: double(2,2)'//new_line('a')// &
            '  single(1,1) = 1.25'//new_line('a')// &
            '  single(2,1) = -2.5'//new_line('a')// &
            '  single(1,2) = 7.5'//new_line('a')// &
            '  single(2,2) = 8.25'//new_line('a')// &
            '  double(1,1) = 3.5d0'//new_line('a')// &
            '  double(2,1) = -4.75d0'//new_line('a')// &
            '  double(1,2) = 6.125d0'//new_line('a')// &
            '  double(2,2) = 9.25d0'//new_line('a')// &
            '  call inspect(single)'//new_line('a')// &
            '  call inspect(double)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine inspect(values)'//new_line('a')// &
            '    class(*), intent(in) :: values(:,:)'//new_line('a')// &
            '    select type (items => values)'//new_line('a')// &
            '    type is (real)'//new_line('a')// &
            '      print *, 4, size(items,1), size(items,2), '// &
            'items(1,1), items(2,2)'//new_line('a')// &
            '    type is (real(8))'//new_line('a')// &
            '      print *, 8, size(items,1), size(items,2), '// &
            'items(1,1), items(2,2)'//new_line('a')// &
            '    class default'//new_line('a')// &
            '      print *, -1'//new_line('a')// &
            '    end select'//new_line('a')// &
            '  end subroutine inspect'//new_line('a')// &
            'end program real_kind_dispatch_main'

        test_real_kind_dispatch = run_differential(source, &
            '/var/tmp/ert/ffc_class_star_real_kind_dispatch')
    end function test_real_kind_dispatch

    logical function test_real4_rank3_rank4()
        character(len=*), parameter :: source = &
            'program real4_rank3_rank4_main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  real :: values3(2,3,2), values4(2,3,2,2)'//new_line('a')// &
            '  values3(1,1,1) = 1.25'//new_line('a')// &
            '  values3(2,2,1) = -2.5'//new_line('a')// &
            '  values3(1,3,2) = 3.75'//new_line('a')// &
            '  values4(1,1,1,1) = 4.25'//new_line('a')// &
            '  values4(2,2,1,2) = -5.5'//new_line('a')// &
            '  values4(1,3,2,2) = 6.75'//new_line('a')// &
            '  call inspect3(values3)'//new_line('a')// &
            '  call inspect4(values4)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine inspect3(values)'//new_line('a')// &
            '    class(*), intent(in) :: values(:,:,:)'//new_line('a')// &
            '    select type (items => values)'//new_line('a')// &
            '    type is (real)'//new_line('a')// &
            '      print *, 3, size(items,1), size(items,2), '// &
            'size(items,3), size(items)'//new_line('a')// &
            '      print *, items(1,1,1), items(2,2,1), items(1,3,2)'// &
            new_line('a')// &
            '    class default'//new_line('a')// &
            '      print *, -1'//new_line('a')// &
            '    end select'//new_line('a')// &
            '  end subroutine inspect3'//new_line('a')// &
            '  subroutine inspect4(values)'//new_line('a')// &
            '    class(*), intent(in) :: values(:,:,:,:)'//new_line('a')// &
            '    select type (items => values)'//new_line('a')// &
            '    type is (real)'//new_line('a')// &
            '      print *, 4, size(items,1), size(items,2), '// &
            'size(items,3), size(items,4), size(items)'//new_line('a')// &
            '      print *, items(1,1,1,1), items(2,2,1,2), '// &
            'items(1,3,2,2)'//new_line('a')// &
            '    class default'//new_line('a')// &
            '      print *, -1'//new_line('a')// &
            '    end select'//new_line('a')// &
            '  end subroutine inspect4'//new_line('a')// &
            'end program real4_rank3_rank4_main'

        test_real4_rank3_rank4 = run_differential(source, &
            '/var/tmp/ert/ffc_class_star_real4_rank3_rank4')
    end function test_real4_rank3_rank4

    logical function test_integer_rank3()
        character(len=*), parameter :: source = &
            'module rank3_integer_m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine inspect(values)'//new_line('a')// &
            '    class(*), intent(in) :: values(:,:,:)'//new_line('a')// &
            '    select type (items => values)'//new_line('a')// &
            '    type is (integer)'//new_line('a')// &
            '      print *, size(items,1), size(items,2), size(items,3), '// &
            'size(items)'//new_line('a')// &
            '      print *, items(1,1,1), items(2,1,2), items(1,2,2)'// &
            new_line('a')// &
            '    class default'//new_line('a')// &
            '      print *, -1'//new_line('a')// &
            '    end select'//new_line('a')// &
            '  end subroutine inspect'//new_line('a')// &
            'end module rank3_integer_m'//new_line('a')// &
            'program rank3_integer_main'//new_line('a')// &
            '  use rank3_integer_m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: values(2,2,2)'//new_line('a')// &
            '  values = reshape([11, 21, 12, 22, 13, 23, 14, 24], '// &
            '[2,2,2])'//new_line('a')// &
            '  call inspect(values)'//new_line('a')// &
            'end program rank3_integer_main'

        test_integer_rank3 = run_differential(source, &
            '/var/tmp/ert/ffc_class_star_rank3_integer')
    end function test_integer_rank3

    logical function test_integer_rank4()
        character(len=*), parameter :: source = &
            'module rank4_integer_m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine inspect(values)'//new_line('a')// &
            '    class(*), intent(in) :: values(:,:,:,:)'//new_line('a')// &
            '    select type (items => values)'//new_line('a')// &
            '    type is (integer)'//new_line('a')// &
            '      print *, size(items,1), size(items,2), size(items,3), '// &
            'size(items,4), size(items)'//new_line('a')// &
            '      print *, items(1,1,1,1), items(2,1,1,2), '// &
            'items(1,2,2,2)'//new_line('a')// &
            '    class default'//new_line('a')// &
            '      print *, -1'//new_line('a')// &
            '    end select'//new_line('a')// &
            '  end subroutine inspect'//new_line('a')// &
            'end module rank4_integer_m'//new_line('a')// &
            'program rank4_integer_main'//new_line('a')// &
            '  use rank4_integer_m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: values(2,2,2,2)'//new_line('a')// &
            '  values(1,1,1,1) = 11'//new_line('a')// &
            '  values(2,1,1,2) = 42'//new_line('a')// &
            '  values(1,2,2,2) = 33'//new_line('a')// &
            '  call inspect(values)'//new_line('a')// &
            'end program rank4_integer_main'

        test_integer_rank4 = run_differential(source, &
            '/var/tmp/ert/ffc_class_star_rank4_integer')
    end function test_integer_rank4

    logical function test_refusal_contract()
        character(len=*), parameter :: unsupported_kind = &
            'program p'//new_line('a')// &
            '  interface'//new_line('a')// &
            '    subroutine inspect(values)'//new_line('a')// &
            '      class(*), intent(in) :: values(:,:)'//new_line('a')// &
            '    end subroutine inspect'//new_line('a')// &
            '  end interface'//new_line('a')// &
            '  logical :: values(2,2)'//new_line('a')// &
            '  call inspect(values)'//new_line('a')// &
            'end program p'
        character(len=*), parameter :: section = &
            'program p'//new_line('a')// &
            '  interface'//new_line('a')// &
            '    subroutine inspect(values)'//new_line('a')// &
            '      class(*), intent(in) :: values(:,:)'//new_line('a')// &
            '    end subroutine inspect'//new_line('a')// &
            '  end interface'//new_line('a')// &
            '  integer :: values(2,3)'//new_line('a')// &
            '  call inspect(values(1:2,2:3))'//new_line('a')// &
            'end program p'
        character(len=*), parameter :: allocatable = &
            'program p'//new_line('a')// &
            '  interface'//new_line('a')// &
            '    subroutine inspect(values)'//new_line('a')// &
            '      class(*), intent(in) :: values(:,:)'//new_line('a')// &
            '    end subroutine inspect'//new_line('a')// &
            '  end interface'//new_line('a')// &
            '  integer, allocatable :: values(:,:)'//new_line('a')// &
            '  allocate(values(2,2))'//new_line('a')// &
            '  call inspect(values)'//new_line('a')// &
            'end program p'
        character(len=*), parameter :: pointer = &
            'program p'//new_line('a')// &
            '  interface'//new_line('a')// &
            '    subroutine inspect(values)'//new_line('a')// &
            '      class(*), intent(in) :: values(:,:)'//new_line('a')// &
            '    end subroutine inspect'//new_line('a')// &
            '  end interface'//new_line('a')// &
            '  integer, target :: storage(2,2)'//new_line('a')// &
            '  integer, pointer :: values(:,:)'//new_line('a')// &
            '  values => storage'//new_line('a')// &
            '  call inspect(values)'//new_line('a')// &
            'end program p'
        character(len=*), parameter :: target = &
            'program p'//new_line('a')// &
            '  interface'//new_line('a')// &
            '    subroutine inspect(values)'//new_line('a')// &
            '      class(*), intent(in) :: values(:,:)'//new_line('a')// &
            '    end subroutine inspect'//new_line('a')// &
            '  end interface'//new_line('a')// &
            '  integer, target :: values(2,2)'//new_line('a')// &
            '  call inspect(values)'//new_line('a')// &
            'end program p'
        test_refusal_contract = &
            expect_refusal(unsupported_kind, &
                'default integer, real, and real(8)') .and. &
            expect_refusal(section, 'sections and non-array actuals') .and. &
            expect_refusal(allocatable, 'allocatable ownership') .and. &
            expect_refusal(pointer, 'pointer/target ownership') .and. &
            expect_refusal(target, 'pointer/target ownership')
    end function test_refusal_contract

    logical function run_differential(source, base)
        character(len=*), intent(in) :: source, base
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result
        character(len=:), allocatable :: error_msg
        character(len=:), allocatable :: src, exe, ref, ffc_out, ref_out
        integer :: unit, exit_stat, status

        run_differential = .false.
        src = trim(base)//'.f90'
        exe = trim(base)//'.ffc'
        ref = trim(base)//'.gf'
        ffc_out = trim(base)//'.ffc.out'
        ref_out = trim(base)//'.gf.out'
        open (newunit=unit, file=src, status='replace', action='write')
        write (unit, '(A)') source
        close (unit)

        options = compiler_frontend_options_t()
        options%run_semantics = .true.
        options%input_mode = INPUT_MODE_STANDARD
        call compile_frontend_from_string(source, frontend_result, options)
        if (.not. frontend_result%success()) then
            print *, 'FAIL: FortFront rejected CLASS(*) source: ', &
                trim(frontend_result%diagnostic_text)
            return
        end if
        call lower_program_to_liric_exe(frontend_result%arena, &
            frontend_result%root_index, exe, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: ffc CLASS(*) lowering failed: ', trim(error_msg)
            return
        end if
        call execute_command_line('gfortran -w '//src//' -o '//ref, &
            exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: gfortran rejected CLASS(*) source'
            return
        end if
        call execute_command_line(exe//' > '//ffc_out//' 2>&1', &
            exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: ffc CLASS(*) executable failed'
            return
        end if
        call execute_command_line(ref//' > '//ref_out//' 2>&1', &
            exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: gfortran CLASS(*) executable failed'
            return
        end if
        call execute_command_line('diff '//ffc_out//' '//ref_out// &
            ' > /dev/null 2>&1', exitstat=status)
        if (status /= 0) then
            print *, 'FAIL: CLASS(*) output differs from gfortran'
            call execute_command_line('diff '//ffc_out//' '//ref_out)
            return
        end if
        call execute_command_line('rm -f '//src//' '//exe//' '//ref//' '// &
            ffc_out//' '//ref_out)
        run_differential = .true.
    end function run_differential

    logical function expect_refusal(source, expected)
        character(len=*), intent(in) :: source, expected
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result
        character(len=:), allocatable :: error_msg
        character(len=128) :: base, src, exe
        integer :: exit_stat, status, unit

        expect_refusal = .false.
        base = '/var/tmp/ert/ffc_class_star_rank2_refusal'
        src = trim(base)//'.f90'
        exe = trim(base)//'.ffc'
        open (newunit=unit, file=src, status='replace', action='write')
        write (unit, '(A)') source
        close (unit)
        call execute_command_line('gfortran -w -fsyntax-only '//trim(src), &
            exitstat=status)
        if (status /= 0) then
            print *, 'FAIL: gfortran rejected valid rank-2 refusal fixture'
            return
        end if
        options = compiler_frontend_options_t()
        options%run_semantics = .true.
        options%input_mode = INPUT_MODE_STANDARD
        call compile_frontend_from_string(source, frontend_result, options)
        if (.not. frontend_result%success()) then
            print *, 'FAIL: FortFront rejected valid refusal fixture: ', &
                trim(frontend_result%diagnostic_text)
            return
        end if
        call lower_program_to_liric_exe(frontend_result%arena, &
            frontend_result%root_index, exe, error_msg)
        if (len_trim(error_msg) == 0) then
            print *, 'FAIL: unsupported rank-2 CLASS(*) fixture compiled'
            return
        end if
        if (index(error_msg, expected) == 0) then
            print *, 'FAIL: refusal diagnostic lacks [', trim(expected), ']: ', &
                trim(error_msg)
            return
        end if
        call execute_command_line('rm -f '//trim(src)//' '//trim(exe), &
            exitstat=exit_stat)
        expect_refusal = .true.
    end function expect_refusal

end subroutine case_test_session_class_star_rank2_assumed_shape_compiler
