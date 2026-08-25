program test_session_character_array_compiler
    use ffc_test_support, only: expect_output
    implicit none

    character(len=*), parameter :: source = &
        'program main'//new_line('a')// &
        '  character(len=4) :: a(2), b(2,2), c(2), d(2,2)'//new_line('a')// &
        '  character(len=2) :: short'//new_line('a')// &
        '  character(len=6) :: long'//new_line('a')// &
        '  short = "xy"'//new_line('a')// &
        '  long = "abcdef"'//new_line('a')// &
        '  a = short'//new_line('a')// &
        '  b = long // "!"'//new_line('a')// &
        '  c = "x"'//new_line('a')// &
        '  d = "yz"'//new_line('a')// &
        '  if (a(2) /= "xy  " .or. b(2,2) /= "abcd") error stop 1'//new_line('a')// &
        '  if (c(2) /= "x   " .or. d(2,2) /= "yz  ") error stop 1'//new_line('a')// &
        '  print *, "[", a(1), a(2), b(1,1), b(2,2), c(1), c(2), d(1,1), d(2,2), "]"'//new_line('a')// &
        'end program main'

    if (.not. expect_output(source, ' [xy  xy  abcdabcdx   x   yz  yz  ]'//new_line('a'), &
            '/tmp/ffc_session_character_array_broadcast_test')) stop 1
    print *, 'PASS: fixed character array scalar assignment lowers correctly'
end program test_session_character_array_compiler
