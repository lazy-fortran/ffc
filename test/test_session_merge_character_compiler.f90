program test_session_merge_character_compiler
    use ffc_test_support, only: expect_output
    implicit none

    character(len=*), parameter :: source = &
        'program main'//new_line('a')// &
        '  character(len=5) :: a, b, c'//new_line('a')// &
        '  logical :: mask4'//new_line('a')// &
        '  logical(1) :: mask1'//new_line('a')// &
        '  a = "hello"'//new_line('a')// &
        '  b = "world"'//new_line('a')// &
        '  mask4 = .true.'//new_line('a')// &
        '  c = merge(a, b, mask4)'//new_line('a')// &
        '  if (c /= "hello") error stop 1'//new_line('a')// &
        '  mask1 = .false.'//new_line('a')// &
        '  c = merge(a, b, mask1)'//new_line('a')// &
        '  if (c /= "world") error stop 2'//new_line('a')// &
        '  mask1 = .true.'//new_line('a')// &
        '  c = merge(a, b, mask1)'//new_line('a')// &
        '  if (c /= "hello") error stop 3'//new_line('a')// &
        '  print *, "ok"'//new_line('a')// &
        'end program main'

    if (.not. expect_output(source, ' ok'//new_line('a'), &
            '/tmp/ffc_session_merge_character')) stop 1
end program test_session_merge_character_compiler
