program test_session_alloc_array_component_compiler
    ! Rank-1 allocatable array components (integer/real/logical) through the
    ! direct LIRIC session: default-unallocated state, allocate(obj%v(n)) with a
    ! runtime extent, element read and write obj%v(i), allocated(obj%v),
    ! size(obj%v), and deallocate(obj%v). The component's inline 16-byte
    ! descriptor (data pointer + i64 extent) starts null and returns to null
    ! after deallocate. Covers a type mixing a scalar field with two allocatable
    ! array components so the inline slot layout is exercised.
    use ffc_test_support, only: expect_output, expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== direct session allocatable array component compiler test ==='

    all_passed = .true.
    if (.not. test_integer_fill_sum()) all_passed = .false.
    if (.not. test_mixed_components()) all_passed = .false.
    if (.not. test_lifecycle_exit_status()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: allocatable array components lower through direct LIRIC session'

contains

    logical function test_integer_fill_sum()
        ! allocate with a runtime extent, fill and read elements in loops, and
        ! print the sum.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: vec_t'//new_line('a')// &
            '    integer, allocatable :: v(:)'//new_line('a')// &
            '  end type vec_t'//new_line('a')// &
            '  type(vec_t) :: a'//new_line('a')// &
            '  integer :: i, s, n'//new_line('a')// &
            '  n = 3'//new_line('a')// &
            '  allocate(a%v(n))'//new_line('a')// &
            '  do i = 1, n'//new_line('a')// &
            '    a%v(i) = i * 10'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  s = 0'//new_line('a')// &
            '  do i = 1, size(a%v)'//new_line('a')// &
            '    s = s + a%v(i)'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  print *, s'//new_line('a')// &
            'end program main'

        test_integer_fill_sum = expect_output( &
            source, '          60'//new_line('a'), &
            '/tmp/ffc_alloc_array_comp_sum')
    end function test_integer_fill_sum

    logical function test_mixed_components()
        ! A scalar field alongside integer and real allocatable array components;
        ! the scalar field and both descriptors coexist in the inline layout. The
        ! real component is checked through arithmetic and an error stop so the
        ! test does not depend on real list-directed formatting.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  type :: box_t'//new_line('a')// &
            '    integer :: tag'//new_line('a')// &
            '    integer, allocatable :: v(:)'//new_line('a')// &
            '    real, allocatable :: r(:)'//new_line('a')// &
            '  end type box_t'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  type(box_t) :: a'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  a%tag = 7'//new_line('a')// &
            '  allocate(a%v(4))'//new_line('a')// &
            '  allocate(a%r(3))'//new_line('a')// &
            '  do i = 1, 4'//new_line('a')// &
            '    a%v(i) = i * i'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  do i = 1, 3'//new_line('a')// &
            '    a%r(i) = real(i) * 0.5'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  if (abs(a%r(2) - 1.0) > 1.0e-6) error stop 8'//new_line('a')// &
            '  if (size(a%v) /= 4) error stop 9'//new_line('a')// &
            '  print *, a%v(3), a%tag'//new_line('a')// &
            'end program main'

        test_mixed_components = expect_output( &
            source, &
            '           9           7'//new_line('a'), &
            '/tmp/ffc_alloc_array_comp_mixed')
    end function test_mixed_components

    logical function test_lifecycle_exit_status()
        ! allocated is false before allocate and after deallocate, true in
        ! between; size matches the requested extent. The program exits 0.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: box_t'//new_line('a')// &
            '    integer, allocatable :: v(:)'//new_line('a')// &
            '  end type box_t'//new_line('a')// &
            '  type(box_t) :: b'//new_line('a')// &
            '  if (allocated(b%v)) error stop 1'//new_line('a')// &
            '  allocate(b%v(5))'//new_line('a')// &
            '  if (.not. allocated(b%v)) error stop 2'//new_line('a')// &
            '  if (size(b%v) /= 5) error stop 3'//new_line('a')// &
            '  b%v(5) = 99'//new_line('a')// &
            '  if (b%v(5) /= 99) error stop 4'//new_line('a')// &
            '  deallocate(b%v)'//new_line('a')// &
            '  if (allocated(b%v)) error stop 5'//new_line('a')// &
            'end program main'

        test_lifecycle_exit_status = expect_exit_status( &
            source, 0, '/tmp/ffc_alloc_array_comp_life')
    end function test_lifecycle_exit_status

end program test_session_alloc_array_component_compiler
