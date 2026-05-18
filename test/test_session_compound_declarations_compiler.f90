program test_session_compound_declarations_compiler
    use fortfront, only: compiler_frontend_options_t, &
                         compiler_frontend_result_t, &
                         compile_frontend_from_string, INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    call check_case('two_integers', &
                    'program main'//new_line('a')// &
                    '  integer :: i, j'//new_line('a')// &
                    '  i = 7'//new_line('a')// &
                    '  j = 5'//new_line('a')// &
                    '  stop i + j'//new_line('a')// &
                    'end program main', 12, &
                    '/tmp/ffc_compound_two_int_test')

    call check_case('three_integers', &
                    'program main'//new_line('a')// &
                    '  integer :: a, b, c'//new_line('a')// &
                    '  a = 10'//new_line('a')// &
                    '  b = 3'//new_line('a')// &
                    '  c = 4'//new_line('a')// &
                    '  stop a + b + c'//new_line('a')// &
                    'end program main', 17, &
                    '/tmp/ffc_compound_three_int_test')

    call check_case('logicals', &
                    'program main'//new_line('a')// &
                    '  implicit none'//new_line('a')// &
                    '  logical :: x, y'//new_line('a')// &
                    '  x = .true.'//new_line('a')// &
                    '  y = .false.'//new_line('a')// &
                    '  if (x) then'//new_line('a')// &
                    '    stop 1'//new_line('a')// &
                    '  else'//new_line('a')// &
                    '    stop 2'//new_line('a')// &
                    '  end if'//new_line('a')// &
                    'end program main', 1, &
                    '/tmp/ffc_compound_logicals_test')

    print *, 'PASS: compound declarations lower through direct LIRIC session'
contains

    subroutine check_case(label, source, expected_exit, exe_path)
        character(len=*), intent(in) :: label
        character(len=*), intent(in) :: source
        integer, intent(in) :: expected_exit
        character(len=*), intent(in) :: exe_path
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result
        character(len=:), allocatable :: error_msg
        integer :: exit_stat, cmd_stat

        print *, '=== compound declaration case: '//label//' ==='

        options = compiler_frontend_options_t()
        options%run_semantics = .true.
        options%input_mode = INPUT_MODE_STANDARD

        call compile_frontend_from_string(source, frontend_result, options)
        if (.not. frontend_result%success()) then
            print *, 'FAIL ['//label//']: FortFront rejected source: ', &
                trim(frontend_result%diagnostic_text)
            stop 1
        end if

        call execute_command_line('rm -f '//exe_path)
        call lower_program_to_liric_exe(frontend_result%arena, &
                                        frontend_result%root_index, &
                                        exe_path, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL ['//label//']: lowering failed: ', trim(error_msg)
            stop 1
        end if

        call execute_command_line(exe_path, exitstat=exit_stat, cmdstat=cmd_stat)
        call execute_command_line('rm -f '//exe_path)
        if (cmd_stat /= 0) then
            print *, 'FAIL ['//label//']: emitted executable did not run'
            stop 1
        end if
        if (exit_stat /= expected_exit) then
            print *, 'FAIL ['//label//']: exit status ', exit_stat, &
                ' expected ', expected_exit
            stop 1
        end if
    end subroutine check_case

end program test_session_compound_declarations_compiler
