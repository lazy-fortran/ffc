program scan_lf
    use fortfront, only: compiler_frontend_options_t, &
                         compiler_frontend_result_t, &
                         compile_frontend_from_file, INPUT_MODE_LAZY
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    character(len=*), parameter :: LF_DIR = '../fortfront/examples/lf/'
    character(len=256) :: line
    character(len=:), allocatable :: supported_list
    character(len=:), allocatable :: unsupported_list
    integer :: supported_count
    integer :: unsupported_count
    integer :: total_count

    supported_count = 0
    unsupported_count = 0
    total_count = 0
    supported_list = ''
    unsupported_list = ''

    call scan_directory(LF_DIR)

    print '(A,I0)', 'Total .lf files: ', total_count
    print '(A,I0)', 'Supported: ', supported_count
    print '(A,I0)', 'Unsupported: ', unsupported_count

    print *, ''
    print *, '=== supported_lf.txt ==='
    print '(A)', trim(supported_list)
    print *, ''
    print *, '=== unsupported_lf.txt ==='
    print '(A)', trim(unsupported_list)

contains

    subroutine scan_directory(dir)
        character(len=*), intent(in) :: dir
        character(len=256) :: entry
        character(len=:), allocatable :: full_path
        character(len=:), allocatable :: exe_path
        character(len=:), allocatable :: error_msg
        integer :: unit
        integer :: io_stat
        integer :: exit_stat
        integer :: dir_unit
        integer :: dir_io

        call execute_command_line('ls "'//trim(dir)//'"*.lf > /tmp/lf_list.txt 2>/dev/null', &
                                  exitstat=io_stat)

        open(newunit=dir_unit, file='/tmp/lf_list.txt', status='old', action='read', &
             iostat=dir_io)
        if (dir_io /= 0) return

        do
            read(dir_unit, '(A)', iostat=io_stat) line
            if (io_stat /= 0) exit
            if (len_trim(line) == 0) cycle

            full_path = trim(dir)//trim(line)
            exe_path = '/tmp/ffc_lf_scan_exe'
            call execute_command_line('rm -f '//exe_path)

            total_count = total_count + 1

            if (.not. compile_and_run(full_path, exe_path, error_msg)) then
                unsupported_count = unsupported_count + 1
                if (len_trim(unsupported_list) > 0) unsupported_list = trim(unsupported_list)//CHAR(10)
                unsupported_list = trim(unsupported_list)//trim(line)
            else
                supported_count = supported_count + 1
                if (len_trim(supported_list) > 0) supported_list = trim(supported_list)//CHAR(10)
                supported_list = trim(supported_list)//trim(line)
            end if
        end do
        close(dir_unit)
        call execute_command_line('rm -f /tmp/lf_list.txt')
    end subroutine scan_directory

    logical function compile_and_run(source, exe, error_msg)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: exe
        character(len=:), allocatable, intent(out) :: error_msg
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: result
        integer :: exit_stat

        options = compiler_frontend_options_t()
        options%run_semantics = .true.
        options%input_mode = INPUT_MODE_LAZY

        call compile_frontend_from_file(source, result, options)
        if (.not. result%success()) then
            error_msg = 'FortFront: '//trim(result%diagnostic_text)
            compile_and_run = .false.
            return
        end if

        call lower_program_to_liric_exe(result%arena, result%root_index, exe, error_msg)
        if (len_trim(error_msg) > 0) then
            compile_and_run = .false.
            return
        end if

        call execute_command_line('timeout 5 '//exe//' > /dev/null 2>&1', exitstat=exit_stat)
        compile_and_run = (exit_stat == 0)
    end function compile_and_run

end program scan_lf
