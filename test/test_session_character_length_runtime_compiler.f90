program test_session_character_length_runtime_compiler
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session character runtime length compiler test ==='

    all_passed = .true.
    if (.not. test_local_length_from_assumed_length_dummy()) all_passed = .false.
    if (.not. test_local_length_tracks_different_call_sites()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: character(len=len(other)) local lowers through direct '// &
        'LIRIC session'

contains

    logical function test_local_length_from_assumed_length_dummy()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  print *, describe("hello")'//new_line('a')// &
            'contains'//new_line('a')// &
            '  function describe(s) result(n)'//new_line('a')// &
            '    character(len=*), intent(in) :: s'//new_line('a')// &
            '    character(len=len(s)) :: local'//new_line('a')// &
            '    integer :: n'//new_line('a')// &
            '    local = s'//new_line('a')// &
            '    n = len(local)'//new_line('a')// &
            '  end function describe'//new_line('a')// &
            'end program main'

        test_local_length_from_assumed_length_dummy = expect_output( &
            source, '           5'//new_line('a'), &
            '/tmp/ffc_session_char_runtime_len_test')
    end function test_local_length_from_assumed_length_dummy

    logical function test_local_length_tracks_different_call_sites()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  print *, describe("hi")'//new_line('a')// &
            '  print *, describe("hello!")'//new_line('a')// &
            'contains'//new_line('a')// &
            '  function describe(s) result(n)'//new_line('a')// &
            '    character(len=*), intent(in) :: s'//new_line('a')// &
            '    character(len=len(s)) :: local'//new_line('a')// &
            '    integer :: n'//new_line('a')// &
            '    local = s'//new_line('a')// &
            '    n = len(local)'//new_line('a')// &
            '  end function describe'//new_line('a')// &
            'end program main'

        test_local_length_tracks_different_call_sites = expect_output( &
            source, '           2'//new_line('a')// &
            '           6'//new_line('a'), &
            '/tmp/ffc_session_char_runtime_len_sites_test')
    end function test_local_length_tracks_different_call_sites

end program test_session_character_length_runtime_compiler
