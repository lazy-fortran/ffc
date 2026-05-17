program test_batch
    use fortfront, only: compiler_frontend_options_t, &
                         compiler_frontend_result_t, &
                         compile_frontend_from_file, INPUT_MODE_LAZY
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    character(len=*), parameter :: LF_DIR = '/home/ert/code/lazy-fortran/fortfront/examples/lf/'
    character(len=512) :: line
    integer :: supported_count
    integer :: unsupported_count
    integer :: total_count
    integer :: i
    integer :: exit_stat

    supported_count = 0
    unsupported_count = 0
    total_count = 0

    call execute_command_line('find /home/ert/code/lazy-fortran/fortfront/examples/lf/ -name "*.lf" -type f | sort > /tmp/lf_paths.txt 2>/dev/null', &
                              exitstat=i)

    open(newunit=i, file='/tmp/lf_paths.txt', status='old', action='read')
    do
        read(i, '(A)', iostat=i) line
        if (i /= 0) exit
        if (len_trim(line) == 0) cycle
        total_count = total_count + 1
        call test_file(trim(line), supported_count, unsupported_count)
    end do
    close(i)
    call execute_command_line('rm -f /tmp/lf_paths.txt')

    print '(A,I0)', 'Total: ', total_count
    print '(A,I0)', 'Supported: ', supported_count
    print '(A,I0)', 'Unsupported: ', unsupported_count

contains

    subroutine test_file(full_path, supported, unsupported)
        character(len=*), intent(in) :: full_path
        integer, intent(inout) :: supported
        integer, intent(inout) :: unsupported
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: result
        character(len=:), allocatable :: error_msg
        character(len=256) :: name
        integer :: pos

        options = compiler_frontend_options_t()
        options%run_semantics = .true.
        options%input_mode = INPUT_MODE_LAZY

        ! Extract basename
        pos = 0
        do while (pos < len_trim(full_path))
            pos = pos + 1
            if (full_path(pos:pos) == '/') name = full_path(pos+1:)
        end do
        ! Trim trailing .lf for display
        if (len_trim(name) > 3 .and. name(len_trim(name)-2:len_trim(name)) == '.lf') then
            name = name(1:len_trim(name)-3)
        end if

        call compile_frontend_from_file(full_path, result, options)
        if (.not. result%success()) then
            unsupported = unsupported + 1
            print '(A,A)', '  UNSUPPORTED: ', trim(name)
            return
        end if

        call lower_program_to_liric_exe(result%arena, result%root_index, '/tmp/ffc_batch_exe', error_msg)
        if (len_trim(error_msg) > 0) then
            unsupported = unsupported + 1
            print '(A,A)', '  UNSUPPORTED: ', trim(name)
            return
        end if

        call execute_command_line('timeout 5 /tmp/ffc_batch_exe > /dev/null 2>&1', exitstat=exit_stat)
        if (exit_stat == 0) then
            supported = supported + 1
            print '(A,A)', '  SUPPORTED:   ', trim(name)
        else
            unsupported = unsupported + 1
            print '(A,A,I0)', '  UNSUPPORTED: ', trim(name), ' (exit ', exit_stat, ')'
        end if
    end subroutine test_file

end program test_batch
