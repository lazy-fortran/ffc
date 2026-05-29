program test_session_if_merge_character_compiler
    ! Regression test for IF-merge across character variables.
    ! Deferred-length character symbols share their (data_ptr, length)
    ! allocas across branches, so the merge is a no-op on the symbol;
    ! fixed-length character symbols carry a fresh materialised string
    ! pointer per assignment and need a real phi on the pointer.
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== IF merge character compiler test ==='

    all_passed = .true.
    if (.not. test_deferred_char_then_branch()) all_passed = .false.
    if (.not. test_deferred_char_else_branch()) all_passed = .false.
    if (.not. test_fixed_char_then_branch()) all_passed = .false.
    if (.not. test_fixed_char_else_branch()) all_passed = .false.
    if (.not. all_passed) stop 1
    print *, 'PASS: IF merge across character variables lowers through direct LIRIC'

contains

    logical function test_deferred_char_then_branch()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=:), allocatable :: s'//new_line('a')// &
            '  integer :: flag'//new_line('a')// &
            '  flag = 1'//new_line('a')// &
            '  s = "init"'//new_line('a')// &
            '  if (flag == 1) then'//new_line('a')// &
            '    s = "hello"'//new_line('a')// &
            '  else'//new_line('a')// &
            '    s = "world"'//new_line('a')// &
            '  end if'//new_line('a')// &
            '  print *, s'//new_line('a')// &
            'end program main'

        test_deferred_char_then_branch = expect_output( &
            source, ' hello'//new_line('a'), &
            '/tmp/ffc_session_if_def_char_then_test')
    end function test_deferred_char_then_branch

    logical function test_deferred_char_else_branch()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=:), allocatable :: s'//new_line('a')// &
            '  integer :: flag'//new_line('a')// &
            '  flag = 0'//new_line('a')// &
            '  s = "init"'//new_line('a')// &
            '  if (flag == 1) then'//new_line('a')// &
            '    s = "hello"'//new_line('a')// &
            '  else'//new_line('a')// &
            '    s = "world"'//new_line('a')// &
            '  end if'//new_line('a')// &
            '  print *, s'//new_line('a')// &
            'end program main'

        test_deferred_char_else_branch = expect_output( &
            source, ' world'//new_line('a'), &
            '/tmp/ffc_session_if_def_char_else_test')
    end function test_deferred_char_else_branch

    logical function test_fixed_char_then_branch()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=5) :: s'//new_line('a')// &
            '  integer :: flag'//new_line('a')// &
            '  flag = 1'//new_line('a')// &
            '  if (flag == 1) then'//new_line('a')// &
            '    s = "hello"'//new_line('a')// &
            '  else'//new_line('a')// &
            '    s = "world"'//new_line('a')// &
            '  end if'//new_line('a')// &
            '  print *, s'//new_line('a')// &
            'end program main'

        test_fixed_char_then_branch = expect_output( &
            source, ' hello'//new_line('a'), &
            '/tmp/ffc_session_if_fix_char_then_test')
    end function test_fixed_char_then_branch

    logical function test_fixed_char_else_branch()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=5) :: s'//new_line('a')// &
            '  integer :: flag'//new_line('a')// &
            '  flag = 0'//new_line('a')// &
            '  if (flag == 1) then'//new_line('a')// &
            '    s = "hello"'//new_line('a')// &
            '  else'//new_line('a')// &
            '    s = "world"'//new_line('a')// &
            '  end if'//new_line('a')// &
            '  print *, s'//new_line('a')// &
            'end program main'

        test_fixed_char_else_branch = expect_output( &
            source, ' world'//new_line('a'), &
            '/tmp/ffc_session_if_fix_char_else_test')
    end function test_fixed_char_else_branch

end program test_session_if_merge_character_compiler
