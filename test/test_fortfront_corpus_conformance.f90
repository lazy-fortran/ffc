program test_fortfront_corpus_conformance
    use fortfront_compiler, only: compiler_frontend_options_t, &
                                  compiler_frontend_result_t, &
                                  compile_frontend_from_file, INPUT_MODE_STANDARD, &
                                  INPUT_MODE_LAZY
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    character(len=*), parameter :: EXAMPLES_DIR = &
        '../fortfront/examples/f90/'
    character(len=*), parameter :: LF_EXAMPLES_DIR = &
        '../fortfront/examples/lf/'
    character(len=*), parameter :: SUPPORTED_MANIFEST = &
        'test/conformance/supported_f90.txt'
    character(len=*), parameter :: SUPPORTED_OUTPUT_MANIFEST = &
        'test/conformance/supported_with_output_f90.txt'
    character(len=*), parameter :: UNSUPPORTED_MANIFEST = &
        'test/conformance/unsupported_f90.txt'
    character(len=*), parameter :: LF_SUPPORTED_MANIFEST = &
        'test/conformance/supported_lf.txt'
    character(len=*), parameter :: LF_UNSUPPORTED_MANIFEST = &
        'test/conformance/unsupported_lf.txt'
    integer, parameter :: RUN_TIMEOUT_SECONDS = 5

    integer :: supported_failed
    integer :: supported_total
    integer :: output_failed
    integer :: output_total
    integer :: unsupported_failed
    integer :: unsupported_total
    integer :: lf_supported_failed
    integer :: lf_supported_total
    integer :: lf_unsupported_failed
    integer :: lf_unsupported_total

    print *, '=== fortfront corpus conformance test ==='

    call run_supported_manifest(SUPPORTED_MANIFEST, supported_total, &
                                supported_failed)
    call run_output_manifest(SUPPORTED_OUTPUT_MANIFEST, output_total, &
                             output_failed)
    call run_unsupported_manifest(UNSUPPORTED_MANIFEST, unsupported_total, &
                                  unsupported_failed)
    call run_lf_supported_manifest(LF_SUPPORTED_MANIFEST, &
                                   lf_supported_total, &
                                   lf_supported_failed)
    call run_lf_unsupported_manifest(LF_UNSUPPORTED_MANIFEST, &
                                     lf_unsupported_total, &
                                     lf_unsupported_failed)

    print '(A,I0,A,I0)', ' supported (exit 0): ', &
        supported_total - supported_failed, '/', supported_total
    print '(A,I0,A,I0)', ' supported with output (matches gfortran): ', &
        output_total - output_failed, '/', output_total
    print '(A,I0,A,I0)', ' unsupported (must reject): ', &
        unsupported_total - unsupported_failed, '/', unsupported_total
    print '(A,I0,A,I0)', ' lf supported (exit 0): ', &
        lf_supported_total - lf_supported_failed, '/', lf_supported_total
    print '(A,I0,A,I0)', ' lf unsupported (must reject): ', &
        lf_unsupported_total - lf_unsupported_failed, '/', &
        lf_unsupported_total

    if (supported_failed > 0 .or. output_failed > 0 .or. &
        unsupported_failed > 0 .or. lf_supported_failed > 0 .or. &
        lf_unsupported_failed > 0) stop 1
    print *, 'PASS: fortfront example corpus conforms to ffc support contract'

contains

    subroutine run_supported_manifest(manifest_path, total, failed)
        character(len=*), intent(in) :: manifest_path
        integer, intent(out) :: total
        integer, intent(out) :: failed
        character(len=256) :: example_name
        character(len=:), allocatable :: source_path
        character(len=:), allocatable :: exe_path
        character(len=:), allocatable :: error_msg
        integer :: exit_stat
        integer :: unit
        integer :: io_stat

        total = 0
        failed = 0

        open (newunit=unit, file=manifest_path, status='old', action='read', &
              iostat=io_stat)
        if (io_stat /= 0) then
            print *, 'FAIL: cannot open supported manifest ', manifest_path
            failed = 1
            return
        end if

        do
            read (unit, '(A)', iostat=io_stat) example_name
            if (io_stat /= 0) exit
            if (is_blank_or_comment(example_name)) cycle
            total = total + 1

            source_path = EXAMPLES_DIR//trim(adjustl(example_name))
            exe_path = '/tmp/ffc_conformance_supported_exe'
            call execute_command_line('rm -f '//exe_path)
            call compile_example(source_path, exe_path, error_msg)
            if (len_trim(error_msg) > 0) then
                failed = failed + 1
                print *, 'FAIL[supported] ', trim(adjustl(example_name)), &
                    ': ', trim(error_msg)
                cycle
            end if

            call run_executable(exe_path, exit_stat)
            call execute_command_line('rm -f '//exe_path)
            if (exit_stat /= 0) then
                failed = failed + 1
                print *, 'FAIL[supported] ', trim(adjustl(example_name)), &
                    ': non-zero exit ', exit_stat
            end if
        end do
        close (unit)
    end subroutine run_supported_manifest

    subroutine run_output_manifest(manifest_path, total, failed)
        character(len=*), intent(in) :: manifest_path
        integer, intent(out) :: total
        integer, intent(out) :: failed
        character(len=256) :: example_name
        character(len=:), allocatable :: source_path
        character(len=:), allocatable :: ffc_exe
        character(len=:), allocatable :: ref_exe
        character(len=:), allocatable :: ffc_out
        character(len=:), allocatable :: ref_out
        character(len=:), allocatable :: error_msg
        integer :: ffc_status
        integer :: ref_status
        integer :: unit
        integer :: io_stat

        total = 0
        failed = 0

        open (newunit=unit, file=manifest_path, status='old', action='read', &
              iostat=io_stat)
        if (io_stat /= 0) then
            print *, 'FAIL: cannot open output manifest ', manifest_path
            failed = 1
            return
        end if

        do
            read (unit, '(A)', iostat=io_stat) example_name
            if (io_stat /= 0) exit
            if (is_blank_or_comment(example_name)) cycle
            total = total + 1

            source_path = EXAMPLES_DIR//trim(adjustl(example_name))
            ffc_exe = '/tmp/ffc_conformance_output_ffc'
            ref_exe = '/tmp/ffc_conformance_output_ref'
            ffc_out = '/tmp/ffc_conformance_output_ffc.out'
            ref_out = '/tmp/ffc_conformance_output_ref.out'
            call execute_command_line('rm -f '//ffc_exe//' '//ref_exe//' '// &
                                      ffc_out//' '//ref_out)

            call compile_example(source_path, ffc_exe, error_msg)
            if (len_trim(error_msg) > 0) then
                failed = failed + 1
                print *, 'FAIL[output] ', trim(adjustl(example_name)), &
                    ': ffc lowering failed: ', trim(error_msg)
                cycle
            end if

            call compile_with_gfortran(source_path, ref_exe, ref_status)
            if (ref_status /= 0) then
                failed = failed + 1
                print *, 'FAIL[output] ', trim(adjustl(example_name)), &
                    ': gfortran failed to build reference binary'
                call execute_command_line('rm -f '//ffc_exe)
                cycle
            end if

            call run_executable_capture(ffc_exe, ffc_out, ffc_status)
            call run_executable_capture(ref_exe, ref_out, ref_status)
            call execute_command_line('rm -f '//ffc_exe//' '//ref_exe)

            if (ffc_status /= ref_status) then
                failed = failed + 1
                print *, 'FAIL[output] ', trim(adjustl(example_name)), &
                    ': exit code mismatch ffc=', ffc_status, ' ref=', ref_status
            else if (.not. files_equal(ffc_out, ref_out)) then
                failed = failed + 1
                print *, 'FAIL[output] ', trim(adjustl(example_name)), &
                    ': stdout differs from gfortran'
            end if
            call execute_command_line('rm -f '//ffc_out//' '//ref_out)
        end do
        close (unit)
    end subroutine run_output_manifest

    subroutine compile_with_gfortran(source_path, exe_path, status)
        character(len=*), intent(in) :: source_path
        character(len=*), intent(in) :: exe_path
        integer, intent(out) :: status

        call execute_command_line('gfortran -w '//source_path//' -o '// &
                                  exe_path//' 2>/dev/null', exitstat=status)
    end subroutine compile_with_gfortran

    subroutine run_executable_capture(exe_path, out_path, exit_stat)
        character(len=*), intent(in) :: exe_path
        character(len=*), intent(in) :: out_path
        integer, intent(out) :: exit_stat
        character(len=:), allocatable :: command
        character(len=16) :: timeout_text

        write (timeout_text, '(I0)') RUN_TIMEOUT_SECONDS
        command = 'timeout '//trim(timeout_text)//' '//exe_path// &
                  ' > '//out_path//' 2>&1'
        call execute_command_line(command, exitstat=exit_stat)
    end subroutine run_executable_capture

    logical function files_equal(left, right)
        character(len=*), intent(in) :: left
        character(len=*), intent(in) :: right
        integer :: status

        call execute_command_line('diff -q '//left//' '//right// &
                                  ' > /dev/null 2>&1', exitstat=status)
        files_equal = status == 0
    end function files_equal

    subroutine run_lf_supported_manifest(manifest_path, total, failed)
        character(len=*), intent(in) :: manifest_path
        integer, intent(out) :: total
        integer, intent(out) :: failed
        character(len=256) :: example_name
        character(len=:), allocatable :: source_path
        character(len=:), allocatable :: exe_path
        character(len=:), allocatable :: error_msg
        integer :: exit_stat
        integer :: unit
        integer :: io_stat

        total = 0
        failed = 0

        open (newunit=unit, file=manifest_path, status='old', action='read', &
              iostat=io_stat)
        if (io_stat /= 0) then
            print *, 'FAIL: cannot open lf supported manifest ', manifest_path
            failed = 1
            return
        end if

        do
            read (unit, '(A)', iostat=io_stat) example_name
            if (io_stat /= 0) exit
            if (is_blank_or_comment(example_name)) cycle
            total = total + 1

            source_path = LF_EXAMPLES_DIR//trim(adjustl(example_name))
            exe_path = '/tmp/ffc_conformance_lf_supported_exe'
            call execute_command_line('rm -f '//exe_path)
            call compile_example_with_mode(source_path, exe_path, &
                                           INPUT_MODE_LAZY, error_msg)
            if (len_trim(error_msg) > 0) then
                failed = failed + 1
                print *, 'FAIL[lf-supported] ', trim(adjustl(example_name)), &
                    ': ', trim(error_msg)
                cycle
            end if

            call run_executable(exe_path, exit_stat)
            call execute_command_line('rm -f '//exe_path)
            if (exit_stat /= 0) then
                failed = failed + 1
                print *, 'FAIL[lf-supported] ', trim(adjustl(example_name)), &
                    ': non-zero exit ', exit_stat
            end if
        end do
        close (unit)
    end subroutine run_lf_supported_manifest

    subroutine run_lf_unsupported_manifest(manifest_path, total, failed)
        character(len=*), intent(in) :: manifest_path
        integer, intent(out) :: total
        integer, intent(out) :: failed
        character(len=256) :: example_name
        character(len=:), allocatable :: source_path
        character(len=:), allocatable :: exe_path
        character(len=:), allocatable :: error_msg
        integer :: unit
        integer :: io_stat

        total = 0
        failed = 0

        open (newunit=unit, file=manifest_path, status='old', action='read', &
              iostat=io_stat)
        if (io_stat /= 0) then
            print *, 'FAIL: cannot open lf unsupported manifest ', manifest_path
            failed = 1
            return
        end if

        do
            read (unit, '(A)', iostat=io_stat) example_name
            if (io_stat /= 0) exit
            if (is_blank_or_comment(example_name)) cycle
            total = total + 1

            source_path = LF_EXAMPLES_DIR//trim(adjustl(example_name))
            exe_path = '/tmp/ffc_conformance_lf_unsupported_exe'
            call execute_command_line('rm -f '//exe_path)
            call compile_example_with_mode(source_path, exe_path, &
                                           INPUT_MODE_LAZY, error_msg)
            call execute_command_line('rm -f '//exe_path)
            if (len_trim(error_msg) == 0) then
                failed = failed + 1
                print *, 'FAIL[lf-unsupported] ', trim(adjustl(example_name)), &
                    ': lowered without any diagnostic'
            end if
        end do
        close (unit)
    end subroutine run_lf_unsupported_manifest

    subroutine compile_example_with_mode(source_path, exe_path, input_mode, &
                                         error_msg)
        character(len=*), intent(in) :: source_path
        character(len=*), intent(in) :: exe_path
        integer, intent(in) :: input_mode
        character(len=:), allocatable, intent(out) :: error_msg
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result

        options = compiler_frontend_options_t()
        options%run_semantics = .true.
        options%input_mode = input_mode

        call compile_frontend_from_file(source_path, frontend_result, options)
        if (.not. frontend_result%success()) then
            error_msg = 'FortFront rejected source: '// &
                        trim(frontend_result%diagnostic_text)
            return
        end if

        call lower_program_to_liric_exe(frontend_result%arena, &
                                        frontend_result%root_index, exe_path, &
                                        error_msg)
    end subroutine compile_example_with_mode

    subroutine run_unsupported_manifest(manifest_path, total, failed)
        character(len=*), intent(in) :: manifest_path
        integer, intent(out) :: total
        integer, intent(out) :: failed
        character(len=256) :: example_name
        character(len=:), allocatable :: source_path
        character(len=:), allocatable :: exe_path
        character(len=:), allocatable :: error_msg
        integer :: unit
        integer :: io_stat

        total = 0
        failed = 0

        open (newunit=unit, file=manifest_path, status='old', action='read', &
              iostat=io_stat)
        if (io_stat /= 0) then
            print *, 'FAIL: cannot open unsupported manifest ', manifest_path
            failed = 1
            return
        end if

        do
            read (unit, '(A)', iostat=io_stat) example_name
            if (io_stat /= 0) exit
            if (is_blank_or_comment(example_name)) cycle
            total = total + 1

            source_path = EXAMPLES_DIR//trim(adjustl(example_name))
            exe_path = '/tmp/ffc_conformance_unsupported_exe'
            call execute_command_line('rm -f '//exe_path)
            call compile_example(source_path, exe_path, error_msg)
            call execute_command_line('rm -f '//exe_path)
            if (len_trim(error_msg) == 0) then
                failed = failed + 1
                print *, 'FAIL[unsupported] ', trim(adjustl(example_name)), &
                    ': lowered without any diagnostic'
            end if
        end do
        close (unit)
    end subroutine run_unsupported_manifest

    subroutine compile_example(source_path, exe_path, error_msg)
        character(len=*), intent(in) :: source_path
        character(len=*), intent(in) :: exe_path
        character(len=:), allocatable, intent(out) :: error_msg
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result

        options = compiler_frontend_options_t()
        options%run_semantics = .true.
        options%input_mode = INPUT_MODE_STANDARD

        call compile_frontend_from_file(source_path, frontend_result, options)
        if (.not. frontend_result%success()) then
            error_msg = 'FortFront rejected source: '// &
                        trim(frontend_result%diagnostic_text)
            return
        end if

        call lower_program_to_liric_exe(frontend_result%arena, &
                                        frontend_result%root_index, exe_path, &
                                        error_msg)
    end subroutine compile_example

    subroutine run_executable(exe_path, exit_stat)
        character(len=*), intent(in) :: exe_path
        integer, intent(out) :: exit_stat
        character(len=:), allocatable :: command
        character(len=16) :: timeout_text

        write (timeout_text, '(I0)') RUN_TIMEOUT_SECONDS
        command = 'timeout '//trim(timeout_text)//' '//exe_path// &
                  ' > /dev/null 2>&1'
        call execute_command_line(command, exitstat=exit_stat)
    end subroutine run_executable

    logical function is_blank_or_comment(line)
        character(len=*), intent(in) :: line
        character(len=:), allocatable :: stripped

        stripped = trim(adjustl(line))
        if (len(stripped) == 0) then
            is_blank_or_comment = .true.
        else if (stripped(1:1) == '#') then
            is_blank_or_comment = .true.
        else
            is_blank_or_comment = .false.
        end if
    end function is_blank_or_comment
end program test_fortfront_corpus_conformance
