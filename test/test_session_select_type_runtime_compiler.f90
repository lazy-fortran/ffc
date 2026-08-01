program test_session_select_type_runtime_compiler
    ! SELECT TYPE on a scalar class(t) selector dispatches from the runtime
    ! dynamic type identity carried by the class descriptor (#419): the same
    ! construct, compiled once, picks a different arm for a base actual and for
    ! an extension actual.
    use ffc_test_support, only: expect_output, expect_error_contains
    implicit none

    logical :: all_passed

    print *, '=== runtime SELECT TYPE on class(t) scalars ==='

    all_passed = .true.
    if (.not. test_type_is_picks_exact_dynamic_type()) all_passed = .false.
    if (.not. test_class_is_matches_extension()) all_passed = .false.
    if (.not. test_class_default_when_no_guard_matches()) all_passed = .false.
    if (.not. test_most_specific_class_is_wins()) all_passed = .false.
    if (.not. test_arm_reads_narrowed_components()) all_passed = .false.
    if (.not. test_duplicate_type_is_guard_is_rejected()) all_passed = .false.
    if (.not. test_impossible_guard_is_rejected()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: runtime SELECT TYPE on class(t) scalars'

contains

    character(len=:) function hierarchy() result(text)
        ! base_t <- mid_t <- leaf_t, plus an unrelated type.
        allocatable :: text
        text = 'module m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type :: base_t'//new_line('a')// &
            '    integer :: x'//new_line('a')// &
            '  end type base_t'//new_line('a')// &
            '  type, extends(base_t) :: mid_t'//new_line('a')// &
            '    integer :: y'//new_line('a')// &
            '  end type mid_t'//new_line('a')// &
            '  type, extends(mid_t) :: leaf_t'//new_line('a')// &
            '    integer :: z'//new_line('a')// &
            '  end type leaf_t'//new_line('a')
    end function hierarchy

    logical function test_type_is_picks_exact_dynamic_type()
        ! One compiled construct, two actuals: TYPE IS compares exact identity,
        ! so the base actual takes the base arm and the extension actual takes
        ! the extension arm. A static choice could not produce both.
        character(len=:), allocatable :: source

        source = hierarchy()// &
            'contains'//new_line('a')// &
            '  subroutine tag(self)'//new_line('a')// &
            '    class(base_t), intent(in) :: self'//new_line('a')// &
            '    select type (self)'//new_line('a')// &
            '    type is (base_t)'//new_line('a')// &
            '      print *, 1'//new_line('a')// &
            '    type is (mid_t)'//new_line('a')// &
            '      print *, 2'//new_line('a')// &
            '    class default'//new_line('a')// &
            '      print *, 9'//new_line('a')// &
            '    end select'//new_line('a')// &
            '  end subroutine tag'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type(base_t) :: b'//new_line('a')// &
            '  type(mid_t) :: e'//new_line('a')// &
            '  b%x = 0'//new_line('a')// &
            '  e%x = 0'//new_line('a')// &
            '  e%y = 0'//new_line('a')// &
            '  call tag(b)'//new_line('a')// &
            '  call tag(e)'//new_line('a')// &
            'end program main'

        test_type_is_picks_exact_dynamic_type = expect_output(source, &
            '           1'//new_line('a')//'           2'//new_line('a'), &
            '/tmp/ffc_st_runtime_type_is')
    end function test_type_is_picks_exact_dynamic_type

    logical function test_class_is_matches_extension()
        ! CLASS IS matches the named type and any extension of it, so a leaf_t
        ! actual reaches a class is (mid_t) guard.
        character(len=:), allocatable :: source

        source = hierarchy()// &
            'contains'//new_line('a')// &
            '  subroutine tag(self)'//new_line('a')// &
            '    class(base_t), intent(in) :: self'//new_line('a')// &
            '    select type (self)'//new_line('a')// &
            '    class is (mid_t)'//new_line('a')// &
            '      print *, 4'//new_line('a')// &
            '    class default'//new_line('a')// &
            '      print *, 9'//new_line('a')// &
            '    end select'//new_line('a')// &
            '  end subroutine tag'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type(leaf_t) :: l'//new_line('a')// &
            '  l%x = 0'//new_line('a')// &
            '  l%y = 0'//new_line('a')// &
            '  l%z = 0'//new_line('a')// &
            '  call tag(l)'//new_line('a')// &
            'end program main'

        test_class_is_matches_extension = expect_output(source, &
            '           4'//new_line('a'), '/tmp/ffc_st_runtime_class_is')
    end function test_class_is_matches_extension

    logical function test_class_default_when_no_guard_matches()
        ! A base actual reaches CLASS DEFAULT when every guard names a proper
        ! extension.
        character(len=:), allocatable :: source

        source = hierarchy()// &
            'contains'//new_line('a')// &
            '  subroutine tag(self)'//new_line('a')// &
            '    class(base_t), intent(in) :: self'//new_line('a')// &
            '    select type (self)'//new_line('a')// &
            '    type is (mid_t)'//new_line('a')// &
            '      print *, 2'//new_line('a')// &
            '    class is (leaf_t)'//new_line('a')// &
            '      print *, 3'//new_line('a')// &
            '    class default'//new_line('a')// &
            '      print *, 9'//new_line('a')// &
            '    end select'//new_line('a')// &
            '  end subroutine tag'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type(base_t) :: b'//new_line('a')// &
            '  b%x = 0'//new_line('a')// &
            '  call tag(b)'//new_line('a')// &
            'end program main'

        test_class_default_when_no_guard_matches = expect_output(source, &
            '           9'//new_line('a'), '/tmp/ffc_st_runtime_default')
    end function test_class_default_when_no_guard_matches

    logical function test_most_specific_class_is_wins()
        ! F2018 11.1.11.2: when several CLASS IS guards match, the most
        ! specific one is selected regardless of source order. Here the less
        ! specific guard is written first, so source order would pick the
        ! wrong arm.
        character(len=:), allocatable :: source

        source = hierarchy()// &
            'contains'//new_line('a')// &
            '  subroutine tag(self)'//new_line('a')// &
            '    class(base_t), intent(in) :: self'//new_line('a')// &
            '    select type (self)'//new_line('a')// &
            '    class is (base_t)'//new_line('a')// &
            '      print *, 1'//new_line('a')// &
            '    class is (mid_t)'//new_line('a')// &
            '      print *, 2'//new_line('a')// &
            '    end select'//new_line('a')// &
            '  end subroutine tag'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type(leaf_t) :: l'//new_line('a')// &
            '  l%x = 0'//new_line('a')// &
            '  l%y = 0'//new_line('a')// &
            '  l%z = 0'//new_line('a')// &
            '  call tag(l)'//new_line('a')// &
            'end program main'

        test_most_specific_class_is_wins = expect_output(source, &
            '           2'//new_line('a'), '/tmp/ffc_st_runtime_specific')
    end function test_most_specific_class_is_wins

    logical function test_arm_reads_narrowed_components()
        ! Inside a TYPE IS arm the associate name has the guard's type, so a
        ! component that exists only on the extension is readable, and it
        ! aliases the selector's storage rather than a copy.
        character(len=:), allocatable :: source

        source = hierarchy()// &
            'contains'//new_line('a')// &
            '  subroutine bump(self)'//new_line('a')// &
            '    class(base_t), intent(inout) :: self'//new_line('a')// &
            '    select type (s => self)'//new_line('a')// &
            '    type is (mid_t)'//new_line('a')// &
            '      s%y = s%y + s%x'//new_line('a')// &
            '    class default'//new_line('a')// &
            '      print *, 9'//new_line('a')// &
            '    end select'//new_line('a')// &
            '  end subroutine bump'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type(mid_t) :: e'//new_line('a')// &
            '  e%x = 5'//new_line('a')// &
            '  e%y = 30'//new_line('a')// &
            '  call bump(e)'//new_line('a')// &
            '  print *, e%y'//new_line('a')// &
            'end program main'

        test_arm_reads_narrowed_components = expect_output(source, &
            '          35'//new_line('a'), '/tmp/ffc_st_runtime_narrowed')
    end function test_arm_reads_narrowed_components

    logical function test_duplicate_type_is_guard_is_rejected()
        ! Two TYPE IS guards naming the same type can never both be reachable
        ! (F2018 C1163); the construct stays rejected.
        character(len=:), allocatable :: source

        source = hierarchy()// &
            'contains'//new_line('a')// &
            '  subroutine tag(self)'//new_line('a')// &
            '    class(base_t), intent(in) :: self'//new_line('a')// &
            '    select type (self)'//new_line('a')// &
            '    type is (mid_t)'//new_line('a')// &
            '      print *, 2'//new_line('a')// &
            '    type is (mid_t)'//new_line('a')// &
            '      print *, 3'//new_line('a')// &
            '    end select'//new_line('a')// &
            '  end subroutine tag'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type(mid_t) :: e'//new_line('a')// &
            '  e%x = 0'//new_line('a')// &
            '  e%y = 0'//new_line('a')// &
            '  call tag(e)'//new_line('a')// &
            'end program main'

        test_duplicate_type_is_guard_is_rejected = expect_error_contains( &
            source, 'duplicate', '/tmp/ffc_st_runtime_dup')
    end function test_duplicate_type_is_guard_is_rejected

    logical function test_impossible_guard_is_rejected()
        ! A guard naming a type outside the selector's declared hierarchy can
        ! never match (F2018 C1162) and is rejected rather than silently
        ! compiled into dead code.
        character(len=:), allocatable :: source

        source = hierarchy()// &
            '  type :: other_t'//new_line('a')// &
            '    integer :: w'//new_line('a')// &
            '  end type other_t'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine tag(self)'//new_line('a')// &
            '    class(mid_t), intent(in) :: self'//new_line('a')// &
            '    select type (self)'//new_line('a')// &
            '    type is (other_t)'//new_line('a')// &
            '      print *, 7'//new_line('a')// &
            '    class default'//new_line('a')// &
            '      print *, 9'//new_line('a')// &
            '    end select'//new_line('a')// &
            '  end subroutine tag'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type(mid_t) :: e'//new_line('a')// &
            '  e%x = 0'//new_line('a')// &
            '  e%y = 0'//new_line('a')// &
            '  call tag(e)'//new_line('a')// &
            'end program main'

        test_impossible_guard_is_rejected = expect_error_contains(source, &
            'cannot match', '/tmp/ffc_st_runtime_impossible')
    end function test_impossible_guard_is_rejected

end program test_session_select_type_runtime_compiler
