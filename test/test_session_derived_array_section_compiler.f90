program test_session_derived_array_section_compiler
    use ffc_test_support, only: expect_error_contains, expect_output
    implicit none

    logical :: all_passed

    all_passed = .true.

    write(*,'(A)') ' === derived array section compiler test ==='

    if (.not. test_rank2_derived_array_element()) all_passed = .false.
    if (.not. test_rank2_derived_component_element()) all_passed = .false.
    if (.not. test_whole_derived_array_actual()) all_passed = .false.
    if (.not. test_rank1_derived_section_actual()) all_passed = .false.
    if (.not. test_rank2_derived_section_actual()) all_passed = .false.
    if (.not. test_genuine_duplicate_rejected()) all_passed = .false.

    if (all_passed) then
        write(*,'(A)') ' PASS: rank-2 derived arrays and section actuals lower'
    else
        write(*,'(A)') ' FAIL: derived array section lowering regressed'
        stop 1
    end if

contains

    logical function test_rank2_derived_array_element()
        ! A rank-2 derived array is one declaration; its elements address
        ! column-major within the flat slot storage.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: t'//new_line('a')// &
            '    integer :: a'//new_line('a')// &
            '  end type t'//new_line('a')// &
            '  type(t) :: g(2,4)'//new_line('a')// &
            '  g(1,1)%a = 11'//new_line('a')// &
            '  g(2,3)%a = 23'//new_line('a')// &
            '  print *, g(1,1)%a + g(2,3)%a'//new_line('a')// &
            'end program main'

        test_rank2_derived_array_element = expect_output( &
            source, '          34'//new_line('a'), &
            '/tmp/ffc_session_derived_rank2_elem_test')
    end function test_rank2_derived_array_element

    logical function test_rank2_derived_component_element()
        ! The same declaration reached as a component of an enclosing type.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: field_t'//new_line('a')// &
            '    integer :: k'//new_line('a')// &
            '  end type field_t'//new_line('a')// &
            '  type :: holder_t'//new_line('a')// &
            '    type(field_t) :: fieldset(2,4)'//new_line('a')// &
            '  end type holder_t'//new_line('a')// &
            '  type(holder_t) :: myfields'//new_line('a')// &
            '  myfields%fieldset(2,3)%k = 7'//new_line('a')// &
            '  myfields%fieldset(1,4)%k = 5'//new_line('a')// &
            '  print *, myfields%fieldset(2,3)%k*myfields%fieldset(1,4)%k'// &
            new_line('a')// &
            'end program main'

        test_rank2_derived_component_element = expect_output( &
            source, '          35'//new_line('a'), &
            '/tmp/ffc_session_derived_rank2_comp_test')
    end function test_rank2_derived_component_element

    logical function test_whole_derived_array_actual()
        ! The dummy declaration of a derived array is the same symbol as the
        ! parameter, not a second, conflicting declaration.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: t'//new_line('a')// &
            '    integer :: a'//new_line('a')// &
            '  end type t'//new_line('a')// &
            '  type(t) :: g(4)'//new_line('a')// &
            '  g(1)%a = 1'//new_line('a')// &
            '  call bump(g)'//new_line('a')// &
            '  print *, g(1)%a'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine bump(x)'//new_line('a')// &
            '    type(t), intent(inout) :: x(1:4)'//new_line('a')// &
            '    x(1)%a = x(1)%a + 41'//new_line('a')// &
            '  end subroutine bump'//new_line('a')// &
            'end program main'

        test_whole_derived_array_actual = expect_output( &
            source, '          42'//new_line('a'), &
            '/tmp/ffc_session_derived_whole_actual_test')
    end function test_whole_derived_array_actual

    logical function test_rank1_derived_section_actual()
        ! A section of a rank-1 derived array is a view of the same storage:
        ! the callee's writes land in the caller's array.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: t'//new_line('a')// &
            '    integer :: a'//new_line('a')// &
            '  end type t'//new_line('a')// &
            '  type(t) :: g(4)'//new_line('a')// &
            '  g(2)%a = 3'//new_line('a')// &
            '  call bump(g(2:3))'//new_line('a')// &
            '  print *, g(2)%a'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine bump(x)'//new_line('a')// &
            '    type(t), intent(inout) :: x(1:2)'//new_line('a')// &
            '    x(1)%a = x(1)%a + 4'//new_line('a')// &
            '  end subroutine bump'//new_line('a')// &
            'end program main'

        test_rank1_derived_section_actual = expect_output( &
            source, '           7'//new_line('a'), &
            '/tmp/ffc_session_derived_rank1_section_test')
    end function test_rank1_derived_section_actual

    logical function test_rank2_derived_section_actual()
        ! A contiguous column section g(1:2,3) of a rank-2 derived array binds
        ! a rank-1 dummy in place; elements outside the section keep their
        ! values. Expected text matches gfortran.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: t'//new_line('a')// &
            '    integer :: a'//new_line('a')// &
            '    integer :: b'//new_line('a')// &
            '  end type t'//new_line('a')// &
            '  type(t) :: g(2,4)'//new_line('a')// &
            '  integer :: i, j'//new_line('a')// &
            '  do j = 1, 4'//new_line('a')// &
            '    do i = 1, 2'//new_line('a')// &
            '      g(i,j)%a = 10*i + j'//new_line('a')// &
            '      g(i,j)%b = 0'//new_line('a')// &
            '    end do'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  call setfields(g(1:2,3))'//new_line('a')// &
            '  print *, g(1,3)%b, g(2,3)%b, g(1,4)%b'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine setfields(fieldset)'//new_line('a')// &
            '    type(t), intent(inout) :: fieldset(1:2)'//new_line('a')// &
            '    fieldset(1)%b = fieldset(1)%a + 100'//new_line('a')// &
            '    fieldset(2)%b = fieldset(2)%a + 200'//new_line('a')// &
            '  end subroutine setfields'//new_line('a')// &
            'end program main'

        test_rank2_derived_section_actual = expect_output( &
            source, '         113         223           0'//new_line('a'), &
            '/tmp/ffc_session_derived_rank2_section_test')
    end function test_rank2_derived_section_actual

    logical function test_genuine_duplicate_rejected()
        ! Two genuine declarations of the same name in one scope are still two
        ! declarations, so the duplicate stays rejected.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: t'//new_line('a')// &
            '    integer :: a'//new_line('a')// &
            '  end type t'//new_line('a')// &
            '  type(t) :: g(2)'//new_line('a')// &
            '  type(t) :: g(3)'//new_line('a')// &
            '  g(1)%a = 1'//new_line('a')// &
            '  print *, g(1)%a'//new_line('a')// &
            'end program main'

        test_genuine_duplicate_rejected = expect_error_contains( &
            source, 'g', '/tmp/ffc_session_derived_dup_decl_test')
    end function test_genuine_duplicate_rejected

end program test_session_derived_array_section_compiler
