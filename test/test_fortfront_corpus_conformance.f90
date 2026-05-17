program test_fortfront_corpus_conformance
    use fortfront, only: compiler_frontend_options_t, &
                         compiler_frontend_result_t, &
                         compile_frontend_from_file, INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    character(len=*), parameter :: EXAMPLES_DIR = &
        '../fortfront/examples/f90/'
    character(len=*), parameter :: SUPPORTED_MANIFEST = &
        'test/conformance/supported_f90.txt'
    character(len=*), parameter :: UNSUPPORTED_MANIFEST = &
        'test/conformance/unsupported_f90.txt'
    integer, parameter :: RUN_TIMEOUT_SECONDS = 5

    integer :: supported_failed
    integer :: supported_total
    integer :: unsupported_failed
    integer :: unsupported_total

    print *, '=== fortfront corpus conformance test ==='

    call run_supported_manifest(SUPPORTED_MANIFEST, supported_total, &
                                supported_failed)
    call run_unsupported_manifest(UNSUPPORTED_MANIFEST, unsupported_total, &
                                  unsupported_failed)

    print '(A,I0,A,I0)', ' supported: ', &
        supported_total - supported_failed, '/', supported_total
    print '(A,I0,A,I0)', ' unsupported (must reject): ', &
        unsupported_total - unsupported_failed, '/', unsupported_total

    if (supported_failed > 0 .or. unsupported_failed > 0) stop 1
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
