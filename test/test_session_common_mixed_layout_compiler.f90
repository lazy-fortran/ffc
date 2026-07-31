program test_session_common_mixed_layout_compiler
    ! #351: COMMON association is by storage sequence (F2018 8.10.3). Two units
    ! naming the same block with reordered mixed-width members must land on the
    ! same bytes: a real(8) consumes eight bytes and must not overlap the
    ! integers that follow it, and a real(8) written through the second unit's
    ! ordering must not run past the block into unrelated storage.
    use ffc_test_support, only: expect_output
    implicit none
    logical :: all_passed

    all_passed = .true.
    print *, '=== COMMON mixed-width layout compiler test ==='

    if (.not. test_reordered_mixed_widths()) all_passed = .false.
    if (.not. test_double_over_two_integers()) all_passed = .false.
    if (.not. test_integer_halves_of_double()) all_passed = .false.

    if (all_passed) then
        print *, 'PASS: COMMON preserves mixed-width byte layout'
    else
        print *, 'FAIL: COMMON mixed-width layout test failed'
    end if
    if (.not. all_passed) stop 1

contains

    logical function test_reordered_mixed_widths()
        ! Reproduces lfortran integration_tests/common_18.f90: the subroutine
        ! writes an eight-byte real over the two integers at bytes 8..15. If the
        ! layout were computed per element rather than per byte, that store
        ! would run past the block and clobber neighbouring storage.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  real(8) :: r8'//new_line('a')// &
            '  integer :: i1, i2'//new_line('a')// &
            '  common /data/ r8, i1, i2'//new_line('a')// &
            '  r8 = 3.14159d0'//new_line('a')// &
            '  i1 = 42'//new_line('a')// &
            '  i2 = 99'//new_line('a')// &
            '  call sub_reordered()'//new_line('a')// &
            '  print *, "PASS: common_18"'//new_line('a')// &
            'end program main'//new_line('a')// &
            'subroutine sub_reordered()'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: j1, j2'//new_line('a')// &
            '  real(8) :: s8'//new_line('a')// &
            '  common /data/ j1, j2, s8'//new_line('a')// &
            '  j1 = 1'//new_line('a')// &
            '  j2 = 2'//new_line('a')// &
            '  s8 = 1.0d0'//new_line('a')// &
            'end subroutine sub_reordered'
        test_reordered_mixed_widths = expect_output( &
            source, ' PASS: common_18'//new_line('a'), &
            '/tmp/ffc_common_mixed_reorder_test')
    end function test_reordered_mixed_widths

    logical function test_double_over_two_integers()
        ! The subroutine's real(8) starts at byte 8, exactly where the main
        ! program's first integer starts, so writing 1.0d0 there is observable
        ! as its two 32-bit halves: 0 then 1072693248 (0x3FF00000).
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  real(8) :: r8'//new_line('a')// &
            '  integer :: i1, i2'//new_line('a')// &
            '  common /data/ r8, i1, i2'//new_line('a')// &
            '  i1 = 7'//new_line('a')// &
            '  i2 = 8'//new_line('a')// &
            '  call sub_reordered()'//new_line('a')// &
            '  print *, i1'//new_line('a')// &
            '  print *, i2'//new_line('a')// &
            'end program main'//new_line('a')// &
            'subroutine sub_reordered()'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: j1, j2'//new_line('a')// &
            '  real(8) :: s8'//new_line('a')// &
            '  common /data/ j1, j2, s8'//new_line('a')// &
            '  s8 = 1.0d0'//new_line('a')// &
            'end subroutine sub_reordered'
        test_double_over_two_integers = expect_output( &
            source, '           0'//new_line('a')// &
            '  1072693248'//new_line('a'), &
            '/tmp/ffc_common_mixed_double_test')
    end function test_double_over_two_integers

    logical function test_integer_halves_of_double()
        ! The mirror direction: the main program's real(8) at byte 0 is seen by
        ! the subroutine as two integers at bytes 0 and 4.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  real(8) :: r8'//new_line('a')// &
            '  integer :: i1, i2'//new_line('a')// &
            '  common /data/ r8, i1, i2'//new_line('a')// &
            '  r8 = 1.0d0'//new_line('a')// &
            '  call sub_reordered()'//new_line('a')// &
            'end program main'//new_line('a')// &
            'subroutine sub_reordered()'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: j1, j2'//new_line('a')// &
            '  real(8) :: s8'//new_line('a')// &
            '  common /data/ j1, j2, s8'//new_line('a')// &
            '  print *, j1'//new_line('a')// &
            '  print *, j2'//new_line('a')// &
            'end subroutine sub_reordered'
        test_integer_halves_of_double = expect_output( &
            source, '           0'//new_line('a')// &
            '  1072693248'//new_line('a'), &
            '/tmp/ffc_common_mixed_halves_test')
    end function test_integer_halves_of_double

end program test_session_common_mixed_layout_compiler
