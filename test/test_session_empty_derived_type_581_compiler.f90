program test_session_empty_derived_type_581_compiler
    !! The gfortran-dg witness is a valid empty BIND(C) module in the
    !! default language mode. Keep the neighbouring non-interoperable case
    !! rejected so this leaf does not relax BIND(C) component constraints.
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, compile_frontend_from_string, &
        INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    logical :: all_passed

    print *, '=== issue #581 empty BIND(C) type compiler test ==='

    all_passed = .true.
    if (.not. test_empty_bind_c_witness()) all_passed = .false.
    if (.not. test_non_interoperable_bind_c_control()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: empty BIND(C) witness matches gfortran'

contains

    logical function test_empty_bind_c_witness()
        character(len=*), parameter :: source = &
            'module stuff'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type, bind(C) :: junk'//new_line('a')// &
            '    ! Empty!'//new_line('a')// &
            '  end type junk'//new_line('a')// &
            'end module stuff'

        test_empty_bind_c_witness = compare_source_acceptance( &
            source, 'empty_valid', .true.)
    end function test_empty_bind_c_witness

    logical function test_non_interoperable_bind_c_control()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  use iso_c_binding'//new_line('a')// &
            '  type, bind(C) :: junk'//new_line('a')// &
            '    character(len=:), allocatable :: bad'//new_line('a')// &
            '  end type junk'//new_line('a')// &
            '  type(junk) :: x'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_non_interoperable_bind_c_control = compare_source_acceptance( &
            source, 'non_interoperable', .false.)
    end function test_non_interoperable_bind_c_control

    logical function compare_source_acceptance(source, stem, expected)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: stem
        logical, intent(in) :: expected
        character(len=:), allocatable :: base, source_path, gfortran_object
        character(len=:), allocatable :: ffc_exe, error_msg
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result
        integer :: unit, command_status, exit_status
        logical :: gfortran_accepts, ffc_accepts

        compare_source_acceptance = .false.
        base = '/tmp/ffc_581_'//trim(stem)
        source_path = base//'.f90'
        gfortran_object = base//'.gfortran.o'
        ffc_exe = base//'.ffc.exe'

        open (newunit=unit, file=source_path, status='replace', &
            action='write', iostat=command_status)
        if (command_status /= 0) return
        write (unit, '(A)') source
        close (unit)

        command_status = 1
        exit_status = 1
        call execute_command_line( &
            'gfortran -w -c '//source_path//' -o '//gfortran_object// &
            ' >/dev/null 2>&1', exitstat=exit_status, cmdstat=command_status)
        gfortran_accepts = .false.
        if (command_status == 0) gfortran_accepts = exit_status == 0

        options = compiler_frontend_options_t()
        options%run_semantics = .true.
        options%input_mode = INPUT_MODE_STANDARD
        call compile_frontend_from_string(source, frontend_result, options)
        ffc_accepts = frontend_result%success()
        if (ffc_accepts) then
            call lower_program_to_liric_exe(frontend_result%arena, &
                frontend_result%root_index, ffc_exe, error_msg)
            if (allocated(error_msg)) then
                if (len_trim(error_msg) > 0) ffc_accepts = .false.
            end if
        end if

        call execute_command_line('rm -f '//source_path//' '// &
            gfortran_object//' '//ffc_exe//' /tmp/stuff.mod '// &
            '/tmp/stuff.fmod')
        if (gfortran_accepts .neqv. ffc_accepts) then
            print *, 'FAIL[', trim(stem), ']: ffc/gfortran acceptance differs'
            return
        end if
        if (gfortran_accepts .neqv. expected) then
            print *, 'FAIL[', trim(stem), ']: unexpected gfortran result'
            return
        end if
        compare_source_acceptance = .true.
    end function compare_source_acceptance

end program test_session_empty_derived_type_581_compiler
