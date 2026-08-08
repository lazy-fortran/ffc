program test_session_class_pointer_dispatch_compiler
    ! Compare the supported scalar CLASS pointer dispatch slice with gfortran,
    ! while keeping arrays, reassociation, and unsupported ownership explicit.
    use ffc_test_support, only: expect_error_contains
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, compile_frontend_from_string, &
        INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    logical :: all_passed

    print *, '=== scalar class-pointer dispatch compiler test ==='

    all_passed = .true.
    if (.not. test_dynamic_dispatch_matches_gfortran()) all_passed = .false.
    if (.not. test_class_pointer_array_is_rejected()) all_passed = .false.
    if (.not. test_class_pointer_reassociation_is_rejected()) all_passed = .false.
    if (.not. test_class_pointer_deallocation_is_rejected()) all_passed = .false.
    if (.not. test_class_pointer_finalization_is_rejected()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: scalar class-pointer dispatch and refusals'

contains

    character(len=:) function dispatch_source() result(text)
        allocatable :: text
        text = 'module dispatch_m'//new_line('a')// &
            '  type :: base_t'//new_line('a')// &
            '  contains'//new_line('a')// &
            '    procedure :: value => base_value'//new_line('a')// &
            '  end type base_t'//new_line('a')// &
            '  type, extends(base_t) :: child_t'//new_line('a')// &
            '  contains'//new_line('a')// &
            '    procedure :: value => child_value'//new_line('a')// &
            '  end type child_t'//new_line('a')// &
            'contains'//new_line('a')// &
            '  integer function base_value(self)'//new_line('a')// &
            '    class(base_t), intent(in) :: self'//new_line('a')// &
            '    base_value = 1'//new_line('a')// &
            '  end function base_value'//new_line('a')// &
            '  integer function child_value(self)'//new_line('a')// &
            '    class(child_t), intent(in) :: self'//new_line('a')// &
            '    child_value = 42'//new_line('a')// &
            '  end function child_value'//new_line('a')// &
            'end module dispatch_m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use dispatch_m'//new_line('a')// &
            '  class(base_t), pointer :: p'//new_line('a')// &
            '  nullify(p)'//new_line('a')// &
            '  if (associated(p)) stop 1'//new_line('a')// &
            '  allocate(child_t :: p)'//new_line('a')// &
            '  if (.not. associated(p)) stop 2'//new_line('a')// &
            '  if (p%value() /= 42) stop 3'//new_line('a')// &
            '  print *, "class pointer dispatch ok"'//new_line('a')// &
            'end program main'
    end function dispatch_source

    logical function test_dynamic_dispatch_matches_gfortran()
        test_dynamic_dispatch_matches_gfortran = matches_gfortran( &
            dispatch_source(), 'dispatch')
    end function test_dynamic_dispatch_matches_gfortran

    logical function test_class_pointer_array_is_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: base_t'//new_line('a')// &
            '  end type base_t'//new_line('a')// &
            '  class(base_t), pointer :: p(:)'//new_line('a')// &
            'end program main'

        test_class_pointer_array_is_rejected = expect_error_contains(source, &
            'derived pointer/target array', '/tmp/ffc_class_ptr_array_reject')
    end function test_class_pointer_array_is_rejected

    logical function test_class_pointer_reassociation_is_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: base_t'//new_line('a')// &
            '  end type base_t'//new_line('a')// &
            '  type(base_t), target :: target_value'//new_line('a')// &
            '  class(base_t), pointer :: p'//new_line('a')// &
            '  p => target_value'//new_line('a')// &
            'end program main'

        test_class_pointer_reassociation_is_rejected = expect_error_contains( &
            source, 'class pointer reassociation', &
            '/tmp/ffc_class_ptr_reassociation_reject')
    end function test_class_pointer_reassociation_is_rejected

    logical function test_class_pointer_deallocation_is_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: base_t'//new_line('a')// &
            '  end type base_t'//new_line('a')// &
            '  class(base_t), pointer :: p'//new_line('a')// &
            '  allocate(base_t :: p)'//new_line('a')// &
            '  deallocate(p)'//new_line('a')// &
            'end program main'

        test_class_pointer_deallocation_is_rejected = expect_error_contains( &
            source, 'ownership and finalization', &
            '/tmp/ffc_class_ptr_deallocate_reject')
    end function test_class_pointer_deallocation_is_rejected

    logical function test_class_pointer_finalization_is_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: base_t'//new_line('a')// &
            '  contains'//new_line('a')// &
            '    final :: finalize_base'//new_line('a')// &
            '  end type base_t'//new_line('a')// &
            '  class(base_t), pointer :: p'//new_line('a')// &
            '  allocate(base_t :: p)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine finalize_base(self)'//new_line('a')// &
            '    type(base_t), intent(inout) :: self'//new_line('a')// &
            '  end subroutine finalize_base'//new_line('a')// &
            'end program main'

        test_class_pointer_finalization_is_rejected = expect_error_contains( &
            source, 'finalizable', '/tmp/ffc_class_ptr_final_reject')
    end function test_class_pointer_finalization_is_rejected

    logical function matches_gfortran(source, stem)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: stem
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result
        character(len=:), allocatable :: error_msg
        character(len=:), allocatable :: base, src, exe, ref, ffc_out, ref_out
        integer :: unit, ffc_status, ref_status, status

        matches_gfortran = .false.
        base = '/tmp/ffc_class_ptr_'//trim(stem)
        src = base//'.f90'
        exe = base//'.ffc'
        ref = base//'.gfortran'
        ffc_out = base//'.ffc.out'
        ref_out = base//'.gfortran.out'

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
            exitstat=ref_status)
        if (ref_status /= 0) then
            print *, 'FAIL[', trim(stem), ']: gfortran rejected source'
            call execute_command_line('rm -f '//src//' '//exe)
            return
        end if

        call execute_command_line(exe//' > '//ffc_out//' 2>&1', &
            exitstat=ffc_status)
        call execute_command_line(ref//' > '//ref_out//' 2>&1', &
            exitstat=ref_status)
        if (ffc_status /= ref_status) then
            print *, 'FAIL[', trim(stem), ']: exit status differs from gfortran'
            call execute_command_line('rm -f '//src//' '//exe//' '//ref//' '// &
                ffc_out//' '//ref_out)
            return
        end if
        call execute_command_line('diff '//ffc_out//' '//ref_out// &
            ' > /dev/null 2>&1', exitstat=status)
        if (status /= 0) then
            print *, 'FAIL[', trim(stem), ']: output differs from gfortran'
            call execute_command_line('diff '//ffc_out//' '//ref_out)
            call execute_command_line('rm -f '//src//' '//exe//' '//ref//' '// &
                ffc_out//' '//ref_out)
            return
        end if
        call execute_command_line('rm -f '//src//' '//exe//' '//ref//' '// &
            ffc_out//' '//ref_out)
        matches_gfortran = .true.
    end function matches_gfortran

end program test_session_class_pointer_dispatch_compiler
