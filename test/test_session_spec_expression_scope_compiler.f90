program test_session_spec_expression_scope_compiler
    ! A specification expression - an array bound, a character length - is
    ! evaluated in the scope where its declaration appears, so a name in one
    ! obeys host association and BLOCK shadowing like any other reference.
    ! Resolving such a name by spelling picks whichever same-named constant
    ! the lowering symbol table happens to hold, which is a silent miscompile
    ! rather than a diagnostic: the array or string simply gets the wrong
    ! width. Each expectation below is an exact width that differs from the
    ! width the shadowed outer constant would give (#329).
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session specification expression scope test ==='

    all_passed = .true.
    if (.not. test_block_shadowed_character_length()) all_passed = .false.
    if (.not. test_later_declared_character_length()) all_passed = .false.
    if (.not. test_host_shadowed_character_length()) all_passed = .false.
    if (.not. test_block_shadowed_array_bound()) all_passed = .false.
    if (.not. test_later_declared_array_bound()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: specification expressions resolve by binding identity'

contains

    logical function test_block_shadowed_character_length()
        ! The BLOCK-local n = 6 hides the host n = 3 for the whole BLOCK, so
        ! the string is 6 wide. The assigned literal is longer than either
        ! candidate width, so len(s) reports the declared width and not the
        ! value's. Resolving `n` by spelling yields the host 3.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, parameter :: n = 3'//new_line('a')// &
            '  block'//new_line('a')// &
            '    integer, parameter :: n = 6'//new_line('a')// &
            '    character(len=n) :: s'//new_line('a')// &
            '    s = "abcdefghij"'//new_line('a')// &
            '    print *, len(s)'//new_line('a')// &
            '  end block'//new_line('a')// &
            'end program main'

        test_block_shadowed_character_length = expect_output( &
            source, '           6'//new_line('a'), &
            '/tmp/ffc_spec_expr_block_char_len_test')
    end function test_block_shadowed_character_length

    logical function test_later_declared_character_length()
        ! The constant is declared after the declaration that uses it. It has
        ! no lowering symbol at that point, so a symbol-table lookup finds
        ! nothing at all; the binding still names the declaration.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=n) :: s'//new_line('a')// &
            '  integer, parameter :: n = 4'//new_line('a')// &
            '  s = "abcdefghij"'//new_line('a')// &
            '  print *, len(s)'//new_line('a')// &
            'end program main'

        test_later_declared_character_length = expect_output( &
            source, '           4'//new_line('a'), &
            '/tmp/ffc_spec_expr_later_char_len_test')
    end function test_later_declared_character_length

    logical function test_host_shadowed_character_length()
        ! Both traps at once: the contained procedure declares its own n = 9
        ! after the declaration that names n, and the host declares n = 3. The
        ! local constant shadows the host throughout the procedure, so the
        ! width is 9.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, parameter :: n = 3'//new_line('a')// &
            '  call show()'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine show()'//new_line('a')// &
            '    character(len=n) :: s'//new_line('a')// &
            '    integer, parameter :: n = 9'//new_line('a')// &
            '    s = "abcdefghijkl"'//new_line('a')// &
            '    print *, len(s)'//new_line('a')// &
            '  end subroutine show'//new_line('a')// &
            'end program main'

        test_host_shadowed_character_length = expect_output( &
            source, '           9'//new_line('a'), &
            '/tmp/ffc_spec_expr_host_char_len_test')
    end function test_host_shadowed_character_length

    logical function test_block_shadowed_array_bound()
        ! The array-bound counterpart, printing both widths so a resolution
        ! that leaks the BLOCK constant outward is caught as well as one that
        ! never sees it.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, parameter :: n = 3'//new_line('a')// &
            '  integer :: outer(n)'//new_line('a')// &
            '  block'//new_line('a')// &
            '    integer, parameter :: n = 7'//new_line('a')// &
            '    integer :: inner(n)'//new_line('a')// &
            '    inner = 1'//new_line('a')// &
            '    print *, size(inner)'//new_line('a')// &
            '  end block'//new_line('a')// &
            '  outer = 1'//new_line('a')// &
            '  print *, size(outer)'//new_line('a')// &
            'end program main'

        test_block_shadowed_array_bound = expect_output( &
            source, '           7'//new_line('a')// &
            '           3'//new_line('a'), &
            '/tmp/ffc_spec_expr_block_array_bound_test')
    end function test_block_shadowed_array_bound

    logical function test_later_declared_array_bound()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(n)'//new_line('a')// &
            '  integer, parameter :: n = 4'//new_line('a')// &
            '  a = 1'//new_line('a')// &
            '  print *, size(a)'//new_line('a')// &
            'end program main'

        test_later_declared_array_bound = expect_output( &
            source, '           4'//new_line('a'), &
            '/tmp/ffc_spec_expr_later_array_bound_test')
    end function test_later_declared_array_bound

end program test_session_spec_expression_scope_compiler
