program test_counted_do_compiler
    use fortfront, only: compiler_frontend_options_t, &
                         compiler_frontend_result_t, &
                         compile_frontend_from_string, INPUT_MODE_STANDARD
    use empty_program_lowering, only: lower_empty_program_to_llvm
    use liric_bindings, only: liric_compile_ll_to_exe
    implicit none

    type(compiler_frontend_options_t) :: options
    type(compiler_frontend_result_t) :: frontend_result
    character(len=:), allocatable :: llvm_ir
    character(len=:), allocatable :: error_msg
    character(len=64) :: first_line
    character(len=64) :: second_line
    character(len=64) :: third_line
    character(len=*), parameter :: exe_path = '/tmp/ffc_counted_do_test'
    character(len=*), parameter :: out_path = '/tmp/ffc_counted_do_test.out'
    integer :: exit_stat
    integer :: io_stat
    integer :: unit
    character(len=*), parameter :: source = &
        'program main'//new_line('a')// &
        '  integer :: i'//new_line('a')// &
        '  do i = 1, 3'//new_line('a')// &
        '    print *, i'//new_line('a')// &
        '  end do'//new_line('a')// &
        'end program main'

    print *, '=== counted do compiler test ==='

    options = compiler_frontend_options_t()
    options%run_semantics = .true.
    options%input_mode = INPUT_MODE_STANDARD

    call compile_frontend_from_string(source, frontend_result, options)
    if (.not. frontend_result%success()) then
        print *, 'FAIL: FortFront rejected source: ', &
            trim(frontend_result%diagnostic_text)
        stop 1
    end if

    call lower_empty_program_to_llvm(frontend_result%arena, &
                                     frontend_result%root_index, llvm_ir, &
                                     error_msg)
    if (len_trim(error_msg) > 0) then
        print *, 'FAIL: lowering failed: ', trim(error_msg)
        stop 1
    end if

    call execute_command_line('rm -f '//exe_path//' '//out_path)
    if (.not. liric_compile_ll_to_exe(llvm_ir, exe_path, error_msg)) then
        print *, 'FAIL: LIRIC executable emission failed: ', trim(error_msg)
        stop 1
    end if

    call execute_command_line(exe_path//' > '//out_path, exitstat=exit_stat)
    if (exit_stat /= 0) then
        print *, 'FAIL: executable exit status ', exit_stat
        call execute_command_line('rm -f '//exe_path//' '//out_path)
        stop 1
    end if

    open (newunit=unit, file=out_path, status='old', action='read', &
          iostat=io_stat)
    if (io_stat /= 0) then
        print *, 'FAIL: could not open captured output'
        call execute_command_line('rm -f '//exe_path//' '//out_path)
        stop 1
    end if
    read (unit, '(A)', iostat=io_stat) first_line
    read (unit, '(A)', iostat=io_stat) second_line
    read (unit, '(A)', iostat=io_stat) third_line
    close (unit)
    call execute_command_line('rm -f '//exe_path//' '//out_path)

    if (trim(adjustl(first_line)) /= '1' .or. &
        trim(adjustl(second_line)) /= '2' .or. &
        trim(adjustl(third_line)) /= '3') then
        print *, 'FAIL: expected output 1,2,3'
        stop 1
    end if

    print *, 'PASS: counted do loop'
end program test_counted_do_compiler
