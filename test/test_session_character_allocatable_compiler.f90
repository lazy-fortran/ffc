program test_session_character_allocatable_compiler
    ! Fixed-length character 1-D allocatable arrays through the direct LIRIC
    ! session: declare, allocate(a(n)) with blank-fill, element write/read,
    ! whole-array print, and deallocate.
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session character allocatable compiler test ==='

    all_passed = .true.
    if (.not. test_element_write_print()) all_passed = .false.
    if (.not. test_whole_array_print()) all_passed = .false.
    if (.not. test_blank_fill_unwritten_slot()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: character allocatable arrays lower through direct LIRIC session'

contains

    logical function test_element_write_print()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=10), allocatable :: a(:)'//new_line('a')// &
            '  allocate(a(2))'//new_line('a')// &
            '  a(1) = "hello"'//new_line('a')// &
            '  a(2) = "world"'//new_line('a')// &
            '  print *, a(1)'//new_line('a')// &
            '  print *, a(2)'//new_line('a')// &
            '  deallocate(a)'//new_line('a')// &
            'end program main'

        test_element_write_print = expect_output( &
            source, ' hello     '//new_line('a')// &
            ' world     '//new_line('a'), '/tmp/ffc_char_alloc_elem')
    end function test_element_write_print

    logical function test_whole_array_print()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=5), allocatable :: a(:)'//new_line('a')// &
            '  allocate(a(3))'//new_line('a')// &
            '  a(1) = "ab"'//new_line('a')// &
            '  a(2) = "cd"'//new_line('a')// &
            '  a(3) = "ef"'//new_line('a')// &
            '  print *, a'//new_line('a')// &
            '  deallocate(a)'//new_line('a')// &
            'end program main'

        test_whole_array_print = expect_output( &
            source, ' ab   cd   ef   '//new_line('a'), '/tmp/ffc_char_alloc_whole')
    end function test_whole_array_print

    logical function test_blank_fill_unwritten_slot()
        ! An unwritten slot prints as blanks, not garbage or a null crash.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=4), allocatable :: a(:)'//new_line('a')// &
            '  allocate(a(2))'//new_line('a')// &
            '  a(1) = "hi"'//new_line('a')// &
            '  print *, a(2)'//new_line('a')// &
            '  deallocate(a)'//new_line('a')// &
            'end program main'

        test_blank_fill_unwritten_slot = expect_output( &
            source, ' '//repeat(' ', 4)//new_line('a'), '/tmp/ffc_char_alloc_blank')
    end function test_blank_fill_unwritten_slot

end program test_session_character_allocatable_compiler
