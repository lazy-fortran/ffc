program test_session_derived_alloc_char_component_compiler
    ! Deferred-length allocatable character components (#402) through the
    ! direct LIRIC session. The component holds the canonical character
    ! descriptor (data pointer + i64 length) inline, so assignment allocates to
    ! the RHS length and deep-copies the bytes, whole-derived assignment leaves
    ! the destination independent of the source, and deallocate frees the owned
    ! data exactly once and restores the unallocated state.
    use ffc_test_support, only: expect_output, expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== direct session derived allocatable character component test ==='

    all_passed = .true.
    if (.not. test_assign_lengths()) all_passed = .false.
    if (.not. test_copy_independence()) all_passed = .false.
    if (.not. test_deallocate_lifecycle()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: deferred-length allocatable character components'

contains

    logical function test_assign_lengths()
        ! Two derived values carry lengths 2 and 7; each reports its own length
        ! and text, and reassignment re-allocates to the new length.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: box_t'//new_line('a')// &
            '    character(len=:), allocatable :: s'//new_line('a')// &
            '  end type box_t'//new_line('a')// &
            '  type(box_t) :: b, c'//new_line('a')// &
            '  if (allocated(b%s)) error stop 1'//new_line('a')// &
            '  b%s = "ab"'//new_line('a')// &
            '  c%s = "seventy"'//new_line('a')// &
            '  if (.not. allocated(b%s)) error stop 2'//new_line('a')// &
            '  if (len(b%s) /= 2) error stop 3'//new_line('a')// &
            '  if (len(c%s) /= 7) error stop 4'//new_line('a')// &
            '  if (b%s /= "ab") error stop 5'//new_line('a')// &
            '  print *, b%s, "|", c%s'//new_line('a')// &
            '  b%s = "abcde"'//new_line('a')// &
            '  if (len(b%s) /= 5) error stop 6'//new_line('a')// &
            '  print *, b%s'//new_line('a')// &
            'end program main'

        test_assign_lengths = expect_output( &
            source, ' ab|seventy'//new_line('a')//' abcde'//new_line('a'), &
            '/tmp/ffc_derived_alloc_char_lengths')
    end function test_assign_lengths

    logical function test_copy_independence()
        ! c = b deep-copies the component: a later reassignment of b%s leaves
        ! c%s at its own length and text.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: box_t'//new_line('a')// &
            '    character(len=:), allocatable :: s'//new_line('a')// &
            '  end type box_t'//new_line('a')// &
            '  type(box_t) :: b, c'//new_line('a')// &
            '  b%s = "ab"'//new_line('a')// &
            '  c = b'//new_line('a')// &
            '  if (len(c%s) /= 2) error stop 1'//new_line('a')// &
            '  b%s = "seventy"'//new_line('a')// &
            '  if (len(c%s) /= 2) error stop 2'//new_line('a')// &
            '  if (c%s /= "ab") error stop 3'//new_line('a')// &
            '  if (len(b%s) /= 7) error stop 4'//new_line('a')// &
            '  print *, c%s, "|", b%s'//new_line('a')// &
            'end program main'

        test_copy_independence = expect_output( &
            source, ' ab|seventy'//new_line('a'), &
            '/tmp/ffc_derived_alloc_char_copy')
    end function test_copy_independence

    logical function test_deallocate_lifecycle()
        ! deallocate frees the owned data once and clears the descriptor; a
        ! second deallocate of the now-unallocated component is a no-op rather
        ! than a double free, and the component can be assigned again.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: box_t'//new_line('a')// &
            '    character(len=:), allocatable :: s'//new_line('a')// &
            '  end type box_t'//new_line('a')// &
            '  type(box_t) :: b'//new_line('a')// &
            '  b%s = "hello"'//new_line('a')// &
            '  deallocate(b%s)'//new_line('a')// &
            '  if (allocated(b%s)) error stop 1'//new_line('a')// &
            '  deallocate(b%s)'//new_line('a')// &
            '  if (allocated(b%s)) error stop 2'//new_line('a')// &
            '  b%s = "again"'//new_line('a')// &
            '  if (len(b%s) /= 5) error stop 3'//new_line('a')// &
            '  print *, b%s'//new_line('a')// &
            'end program main'

        test_deallocate_lifecycle = expect_output( &
            source, ' again'//new_line('a'), &
            '/tmp/ffc_derived_alloc_char_dealloc')
    end function test_deallocate_lifecycle

end program test_session_derived_alloc_char_component_compiler
