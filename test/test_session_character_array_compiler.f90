program test_session_character_array_compiler
    use ffc_test_support, only: expect_output
    implicit none

    character(len=*), parameter :: source = &
        'program main'//new_line('a')// &
        '  character(len=4) :: a(2), b(2,2)'//new_line('a')// &
        '  a = "x"'//new_line('a')// &
        '  b = "yz"'//new_line('a')// &
        '  if (a(2) /= "x" .or. b(2,2) /= "yz") error stop 1'//new_line('a')// &
        '  print *, "[", a(1), a(2), b(1,1), b(2,2), "]"'//new_line('a')// &
        'end program main'

    if (.not. expect_output(source, ' [x   x   yz  yz  ]'//new_line('a'), &
            '/tmp/ffc_session_character_array_broadcast_test')) stop 1
    print *, 'PASS: fixed character array scalar assignment lowers correctly'
end program test_session_character_array_compiler
