program test_session_deferred_char_array_compiler
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session deferred-length character array compiler test ==='

    all_passed = .true.
    if (.not. test_declare_allocate_assign_print()) all_passed = .false.
    if (.not. test_element_read_and_whole_print()) all_passed = .false.
    if (.not. test_allocated_and_deallocate()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: deferred-length character allocatable arrays lower correctly'

contains

    logical function test_declare_allocate_assign_print()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=:), allocatable :: c(:)'//new_line('a')// &
            '  allocate(character(len=5) :: c(3))'//new_line('a')// &
            '  c(1) = "apple"'//new_line('a')// &
            '  c(2) = "mango"'//new_line('a')// &
            '  c(3) = "peach"'//new_line('a')// &
            '  print *, c(2)'//new_line('a')// &
            'end program main'

        test_declare_allocate_assign_print = expect_output( &
            source, ' mango'//new_line('a'), &
            '/tmp/ffc_session_deferred_char_array_assign_test')
    end function test_declare_allocate_assign_print

    logical function test_element_read_and_whole_print()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(:), allocatable :: c(:)'//new_line('a')// &
            '  allocate(character(5) :: c(3))'//new_line('a')// &
            '  c(1) = "apple"'//new_line('a')// &
            '  c(2) = "mango"'//new_line('a')// &
            '  c(3) = "peach"'//new_line('a')// &
            '  if (c(1) /= "apple") error stop 2'//new_line('a')// &
            '  if (c(3) /= "peach") error stop 3'//new_line('a')// &
            '  print *, c'//new_line('a')// &
            'end program main'

        test_element_read_and_whole_print = expect_output( &
            source, ' applemangopeach'//new_line('a'), &
            '/tmp/ffc_session_deferred_char_array_read_test')
    end function test_element_read_and_whole_print

    logical function test_allocated_and_deallocate()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(:), allocatable :: c(:)'//new_line('a')// &
            '  allocate(character(5) :: c(3))'//new_line('a')// &
            '  c(1) = "apple"'//new_line('a')// &
            '  if (.not. allocated(c)) error stop 1'//new_line('a')// &
            '  deallocate(c)'//new_line('a')// &
            '  if (allocated(c)) error stop 2'//new_line('a')// &
            '  print *, "OK"'//new_line('a')// &
            'end program main'

        test_allocated_and_deallocate = expect_output( &
            source, ' OK'//new_line('a'), &
            '/tmp/ffc_session_deferred_char_array_alloc_test')
    end function test_allocated_and_deallocate

end program test_session_deferred_char_array_compiler
