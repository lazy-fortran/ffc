program test_session_runtime_character_result_compiler
    ! A runtime-length contained result must transfer its heap descriptor to a
    ! deferred caller, which then owns and can deallocate the returned value.
    ! The executable's own LEN/value assertions are the behavioral oracle;
    ! expect_no_leaks independently checks the transfer and deallocation.
    use ffc_test_support, only: expect_exit_status, expect_no_leaks
    implicit none

    logical :: all_passed
    character(len=:), allocatable :: source

    source = &
        'program main'//new_line('a')// &
        '  character(len=:), allocatable :: r'//new_line('a')// &
        '  r = make(4)'//new_line('a')// &
        '  if (len(r) /= 4) stop 11'//new_line('a')// &
        '  if (r /= "ZZZZ") stop 12'//new_line('a')// &
        '  deallocate(r)'//new_line('a')// &
        '  stop 0'//new_line('a')// &
        'contains'//new_line('a')// &
        '  function make(k) result(s)'//new_line('a')// &
        '    integer, intent(in) :: k'//new_line('a')// &
        '    character(len=k) :: s'//new_line('a')// &
        '    s = repeat("Z", k)'//new_line('a')// &
        '  end function make'//new_line('a')// &
        'end program main'

    all_passed = .true.
    if (.not. expect_exit_status(source, 0, &
            '/tmp/ffc_runtime_character_result_exit')) all_passed = .false.
    if (.not. expect_no_leaks(source, &
            '/tmp/ffc_runtime_character_result_leak')) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: runtime-length character result transfers ownership'
end program test_session_runtime_character_result_compiler
