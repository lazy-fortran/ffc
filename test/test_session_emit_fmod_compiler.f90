program test_session_emit_fmod_compiler
    use fortfront, only: compiler_frontend_options_t, &
                         compiler_frontend_result_t, &
                         compile_frontend_from_string, INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_object
    implicit none

    logical :: all_passed

    print *, '=== module .fmod artefact emission tests ==='

    all_passed = .true.
    if (.not. test_compile_module_emits_fmod_with_constant()) all_passed = .false.
    if (.not. test_compile_module_with_derived_type_emits_fmod_with_type()) &
        all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: module .fmod artefact emission'

contains

    logical function test_compile_module_emits_fmod_with_constant()
        character(len=*), parameter :: source = &
            'module consts'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer, parameter :: width = 80'//new_line('a')// &
            'end module consts'
        character(len=*), parameter :: fmod = '/tmp/ffc_fmod_test_consts/consts.fmod'

        test_compile_module_emits_fmod_with_constant = .false.
        call execute_command_line('mkdir -p /tmp/ffc_fmod_test_consts')
        call execute_command_line('rm -f '//fmod)
        if (.not. compile_module(source, &
                '/tmp/ffc_fmod_test_consts/consts.o')) return
        if (.not. file_contains(fmod, 'name = "width"')) then
            print *, 'FAIL: .fmod missing parameter name'
            return
        end if
        if (.not. file_contains(fmod, 'value = 80')) then
            print *, 'FAIL: .fmod missing parameter value'
            return
        end if
        test_compile_module_emits_fmod_with_constant = .true.
    end function test_compile_module_emits_fmod_with_constant

    logical function test_compile_module_with_derived_type_emits_fmod_with_type()
        character(len=*), parameter :: source = &
            'module shapes'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type :: point_t'//new_line('a')// &
            '    integer :: x'//new_line('a')// &
            '    integer :: y'//new_line('a')// &
            '  end type point_t'//new_line('a')// &
            'end module shapes'
        character(len=*), parameter :: fmod = '/tmp/ffc_fmod_test_shapes/shapes.fmod'

        test_compile_module_with_derived_type_emits_fmod_with_type = .false.
        call execute_command_line('mkdir -p /tmp/ffc_fmod_test_shapes')
        call execute_command_line('rm -f '//fmod)
        if (.not. compile_module(source, &
                '/tmp/ffc_fmod_test_shapes/shapes.o')) return
        if (.not. file_contains(fmod, 'name = "point_t"')) then
            print *, 'FAIL: .fmod missing derived type name'
            return
        end if
        if (.not. file_contains(fmod, 'name = "x", kind = "integer"')) then
            print *, 'FAIL: .fmod missing derived type component'
            return
        end if
        test_compile_module_with_derived_type_emits_fmod_with_type = .true.
    end function test_compile_module_with_derived_type_emits_fmod_with_type

    logical function compile_module(source, object_path) result(ok)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: object_path
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result
        character(len=:), allocatable :: error_msg

        ok = .false.
        options = compiler_frontend_options_t()
        options%run_semantics = .true.
        options%input_mode = INPUT_MODE_STANDARD
        call compile_frontend_from_string(source, frontend_result, options)
        if (.not. frontend_result%success()) then
            print *, 'FAIL: FortFront rejected module: ', &
                trim(frontend_result%diagnostic_text)
            return
        end if
        call lower_program_to_liric_object(frontend_result%arena, &
                                           frontend_result%root_index, &
                                           object_path, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: module object lowering failed: ', trim(error_msg)
            return
        end if
        ok = .true.
    end function compile_module

    logical function file_contains(path, fragment) result(found)
        character(len=*), intent(in) :: path
        character(len=*), intent(in) :: fragment
        integer :: unit, io_stat
        character(len=512) :: line

        found = .false.
        open (newunit=unit, file=path, status='old', action='read', &
              iostat=io_stat)
        if (io_stat /= 0) then
            print *, 'FAIL: could not open ', path
            return
        end if
        do
            read (unit, '(A)', iostat=io_stat) line
            if (io_stat /= 0) exit
            if (index(line, fragment) > 0) then
                found = .true.
                exit
            end if
        end do
        close (unit)
    end function file_contains

end program test_session_emit_fmod_compiler
