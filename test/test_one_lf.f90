program test_one
    use fortfront, only: compiler_frontend_options_t, &
                         compiler_frontend_result_t, &
                         compile_frontend_from_file, INPUT_MODE_LAZY
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    character(len=*), parameter :: TEST_FILE = '../fortfront/examples/lf/array_element_assignment.lf'
    character(len=:), allocatable :: error_msg
    type(compiler_frontend_options_t) :: options
    type(compiler_frontend_result_t) :: result
    integer :: exit_stat

    options = compiler_frontend_options_t()
    options%run_semantics = .true.
    options%input_mode = INPUT_MODE_LAZY

    print *, 'Testing:', trim(TEST_FILE)
    call compile_frontend_from_file(TEST_FILE, result, options)
    if (.not. result%success()) then
        print *, 'FortFront failed:'
        print '(A)', trim(result%diagnostic_text)
    else
        print *, 'FortFront succeeded'
        call lower_program_to_liric_exe(result%arena, result%root_index, '/tmp/test_one_exe', error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'Lowering failed:'
            print '(A)', trim(error_msg)
        else
            print *, 'Lowering succeeded'
            call execute_command_line('timeout 5 /tmp/test_one_exe > /dev/null 2>&1', exitstat=exit_stat)
            print '(A,I0)', 'Exit code: ', exit_stat
        end if
    end if
end program test_one
