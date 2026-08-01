program test_session_deferred_char_array_compiler
    use ffc_test_support, only: expect_output, expect_error_contains
    implicit none

    logical :: all_passed

    print *, '=== direct session deferred-length character array compiler test ==='

    all_passed = .true.
    if (.not. test_declare_allocate_assign_print()) all_passed = .false.
    if (.not. test_element_read_and_whole_print()) all_passed = .false.
    if (.not. test_allocated_and_deallocate()) all_passed = .false.
    if (.not. test_deferred_elements_report_allocated_length()) &
        all_passed = .false.
    if (.not. test_conforming_whole_array_assignment()) all_passed = .false.
    if (.not. test_nonconformable_assignment_reports_shape()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: deferred-length character allocatable arrays lower correctly'

contains

    logical function test_deferred_elements_report_allocated_length()
        ! A deferred-length character array takes its element length from the
        ! allocate type-spec, and every element carries that length with the
        ! usual blank padding.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=:), allocatable :: c(:)'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  allocate(character(len=4) :: c(3))'//new_line('a')// &
            '  c(1) = "ab"'//new_line('a')// &
            '  c(2) = "cdef"'//new_line('a')// &
            '  c(3) = "gh"'//new_line('a')// &
            '  do i = 1, 3'//new_line('a')// &
            '    print *, i, len(c(i)), "[", c(i), "]"'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  print *, size(c)'//new_line('a')// &
            '  deallocate(c)'//new_line('a')// &
            'end program main'

        test_deferred_elements_report_allocated_length = expect_output( &
            source, &
            '           1           4 [ab  ]'//new_line('a')// &
            '           2           4 [cdef]'//new_line('a')// &
            '           3           4 [gh  ]'//new_line('a')// &
            '           3'//new_line('a'), &
            '/tmp/ffc_session_deferred_char_array_len_test')
    end function test_deferred_elements_report_allocated_length

    logical function test_conforming_whole_array_assignment()
        ! Whole-array assignment between conforming character arrays copies
        ! every element, which contiguous element storage makes a single
        ! block copy rather than a walk over per-element pointers.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=5) :: c(3), d(3)'//new_line('a')// &
            '  c(1) = "apple"'//new_line('a')// &
            '  c(2) = "mango"'//new_line('a')// &
            '  c(3) = "peach"'//new_line('a')// &
            '  d = c'//new_line('a')// &
            '  print *, "[", d(1), d(2), d(3), "]"'//new_line('a')// &
            'end program main'

        test_conforming_whole_array_assignment = expect_output( &
            source, ' [applemangopeach]'//new_line('a'), &
            '/tmp/ffc_session_char_array_whole_assign_test')
    end function test_conforming_whole_array_assignment

    logical function test_nonconformable_assignment_reports_shape()
        ! Assigning between character arrays of different extent is a shape
        ! error, reported as one rather than misread as some other kind of
        ! value.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=4) :: a(3)'//new_line('a')// &
            '  character(len=4) :: b(2)'//new_line('a')// &
            '  a(1) = "xx"'//new_line('a')// &
            '  a(2) = "yy"'//new_line('a')// &
            '  a(3) = "zz"'//new_line('a')// &
            '  b = a'//new_line('a')// &
            '  print *, b(1)'//new_line('a')// &
            'end program main'

        test_nonconformable_assignment_reports_shape = expect_error_contains( &
            source, 'conforming shapes', &
            '/tmp/ffc_session_char_array_nonconform_test')
    end function test_nonconformable_assignment_reports_shape

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
