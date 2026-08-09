program test_session_runtime_len_expr_result_compiler
    ! The executable checks both the runtime LEN and every returned byte;
    ! expect_no_leaks independently checks the descriptor's ownership.
    use ffc_test_support, only: expect_exit_status, expect_no_leaks
    implicit none

    logical :: all_passed
    character(len=:), allocatable :: source

    source = &
        'program main'//new_line('a')// &
        '  character(len=:), allocatable :: r'//new_line('a')// &
        '  r = greet("Ada")'//new_line('a')// &
        '  if (len(r) /= 10) stop 11'//new_line('a')// &
        '  if (r /= "Hello, Ada") stop 12'//new_line('a')// &
        '  deallocate(r)'//new_line('a')// &
        '  stop 0'//new_line('a')// &
        'contains'//new_line('a')// &
        '  function greet(name) result(s)'//new_line('a')// &
        '    character(len=*), intent(in) :: name'//new_line('a')// &
        '    character(len=len(name)+7) :: s'//new_line('a')// &
        '    s = "Hello, " // name'//new_line('a')// &
        '  end function greet'//new_line('a')// &
        'end program main'

    all_passed = .true.
    if (.not. expect_exit_status(source, 0, &
            '/tmp/ffc_runtime_length_expression_result_exit')) all_passed = .false.
    if (.not. expect_no_leaks(source, &
            '/tmp/ffc_runtime_length_expression_result_leak')) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: runtime length expression result has exact LEN/value and clean ownership'
end program test_session_runtime_len_expr_result_compiler
