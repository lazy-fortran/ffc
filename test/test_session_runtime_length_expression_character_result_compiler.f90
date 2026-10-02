! fo: dispatcher
module ffc_case_7f5152b0a55651ce7b58b5e4
    implicit none
    private
    public :: case_7f5152b0a55651ce7b58b5e4
    interface
        subroutine case_7f5152b0a55651ce7b58b5e4()
        end subroutine case_7f5152b0a55651ce7b58b5e4
    end interface
end module ffc_case_7f5152b0a55651ce7b58b5e4

subroutine case_7f5152b0a55651ce7b58b5e4()
    ! The executable checks both the runtime LEN and every returned byte;
    ! expect_no_leaks independently checks the descriptor's ownership.
    use ffc_test_support, only: expect_exit_status, expect_no_leaks
    implicit none
    save

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
end subroutine case_7f5152b0a55651ce7b58b5e4
