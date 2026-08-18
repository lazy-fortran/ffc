program test_session_block_shadow_compiler
    ! #280: a variable declared inside a BLOCK shadows an identically named
    ! outer variable. Writes to the inner name must not touch the outer storage,
    ! and the outer value must be visible again after the block ends.
    use ffc_test_support, only: expect_error_contains, expect_output
    implicit none
    logical :: all_passed

    all_passed = .true.
    print *, '=== BLOCK variable shadowing compiler test ==='

    if (.not. test_shadow_restores_outer()) all_passed = .false.
    if (.not. test_outer_write_persists()) all_passed = .false.
    if (.not. test_associate_shadow_restores_outer()) all_passed = .false.
    if (.not. test_block_local_is_not_visible_after_end()) all_passed = .false.

    if (all_passed) then
        print *, 'PASS: BLOCK shadowing lowers through direct LIRIC session'
    else
        print *, 'FAIL: BLOCK shadowing test failed'
    end if
    if (.not. all_passed) stop 1

contains

    logical function test_shadow_restores_outer()
        ! The outer value remains visible before and after the BLOCK while the
        ! inner declaration shadows it only inside: 1, 2, 1.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: value'//new_line('a')// &
            '  value = 1'//new_line('a')// &
            '  print *, value'//new_line('a')// &
            '  block'//new_line('a')// &
            '    integer :: value'//new_line('a')// &
            '    value = 2'//new_line('a')// &
            '    print *, value'//new_line('a')// &
            '  end block'//new_line('a')// &
            '  print *, value'//new_line('a')// &
            'end program main'
        test_shadow_restores_outer = expect_output( &
            source, '           1'//new_line('a')//'           2'//new_line('a')// &
            '           1'//new_line('a'), &
            '/tmp/ffc_block_shadow_test')
    end function test_shadow_restores_outer

    logical function test_outer_write_persists()
        ! A non-shadowed outer variable written inside the BLOCK keeps the new
        ! value after the block (no spurious restore).
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  x = 5'//new_line('a')// &
            '  block'//new_line('a')// &
            '    integer :: y'//new_line('a')// &
            '    y = x + 3'//new_line('a')// &
            '    x = y * 2'//new_line('a')// &
            '  end block'//new_line('a')// &
            '  print *, x'//new_line('a')// &
            'end program main'
        test_outer_write_persists = expect_output( &
            source, '          16'//new_line('a'), '/tmp/ffc_block_outer_test2')
    end function test_outer_write_persists

    logical function test_associate_shadow_restores_outer()
        ! An ASSOCIATE name is a construct-local binding even when it has the
        ! same spelling as the selector's host variable. The host storage must
        ! be unchanged after END ASSOCIATE.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: value'//new_line('a')// &
            '  value = 1'//new_line('a')// &
            '  associate (value => 2)'//new_line('a')// &
            '    print *, value'//new_line('a')// &
            '  end associate'//new_line('a')// &
            '  print *, value'//new_line('a')// &
            'end program main'
        test_associate_shadow_restores_outer = expect_output( &
            source, '           2'//new_line('a')//'           1'//new_line('a'), &
            '/tmp/ffc_associate_shadow_test')
    end function test_associate_shadow_restores_outer

    logical function test_block_local_is_not_visible_after_end()
        ! A BLOCK declaration must be removed from storage lookup at the
        ! construct boundary; FortFront therefore cannot be bypassed by a stale
        ! text-keyed symbol after END BLOCK.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: outer'//new_line('a')// &
            '  outer = 1'//new_line('a')// &
            '  block'//new_line('a')// &
            '    integer :: inside'//new_line('a')// &
            '    inside = 2'//new_line('a')// &
            '  end block'//new_line('a')// &
            '  print *, inside'//new_line('a')// &
            'end program main'
        test_block_local_is_not_visible_after_end = expect_error_contains( &
            source, 'Name ''inside'' is not declared', &
            '/tmp/ffc_block_scope_error_test')
    end function test_block_local_is_not_visible_after_end

end program test_session_block_shadow_compiler
