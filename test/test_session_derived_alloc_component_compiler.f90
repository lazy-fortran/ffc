program test_session_derived_alloc_component_compiler
    ! Scalar allocatable components (integer/real/logical) through the direct
    ! LIRIC session: default-unallocated state, allocate(x%c), component read
    ! and write, allocated(x%c) in a conditional, and deallocate(x%c). The
    ! component's inline data pointer starts null and is restored to null after
    ! deallocate. Covers a module-defined type mixing a defaulted scalar with
    ! allocatable components.
    use ffc_test_support, only: expect_output, expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== direct session derived allocatable component compiler test ==='

    all_passed = .true.
    if (.not. test_integer_lifecycle()) all_passed = .false.
    if (.not. test_module_mixed_components()) all_passed = .false.
    if (.not. test_unallocated_after_deallocate()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: scalar allocatable components lower through direct LIRIC session'

contains

    logical function test_integer_lifecycle()
        ! allocate, write, arithmetic read/write, print, deallocate.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: box_t'//new_line('a')// &
            '    integer, allocatable :: v'//new_line('a')// &
            '  end type box_t'//new_line('a')// &
            '  type(box_t) :: b'//new_line('a')// &
            '  if (allocated(b%v)) error stop 1'//new_line('a')// &
            '  allocate(b%v)'//new_line('a')// &
            '  if (.not. allocated(b%v)) error stop 2'//new_line('a')// &
            '  b%v = 42'//new_line('a')// &
            '  b%v = b%v + 1'//new_line('a')// &
            '  print *, b%v'//new_line('a')// &
            '  deallocate(b%v)'//new_line('a')// &
            'end program main'

        test_integer_lifecycle = expect_output( &
            source, '          43'//new_line('a'), &
            '/tmp/ffc_derived_alloc_int')
    end function test_integer_lifecycle

    logical function test_module_mixed_components()
        ! A module type mixing a defaulted integer with allocatable real and
        ! logical components; the defaulted field and the null pointers coexist.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  type :: rec_t'//new_line('a')// &
            '    integer :: id = 7'//new_line('a')// &
            '    real, allocatable :: x'//new_line('a')// &
            '    logical, allocatable :: flag'//new_line('a')// &
            '  end type rec_t'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  type(rec_t) :: r'//new_line('a')// &
            '  if (allocated(r%x)) error stop 1'//new_line('a')// &
            '  if (r%id /= 7) error stop 2'//new_line('a')// &
            '  allocate(r%x)'//new_line('a')// &
            '  allocate(r%flag)'//new_line('a')// &
            '  r%x = 2.5'//new_line('a')// &
            '  r%flag = .true.'//new_line('a')// &
            '  r%x = r%x * 2.0'//new_line('a')// &
            '  print *, r%x, r%flag, r%id'//new_line('a')// &
            'end program main'

        test_module_mixed_components = expect_output( &
            source, '   5.00000000     T           7'//new_line('a'), &
            '/tmp/ffc_derived_alloc_mixed')
    end function test_module_mixed_components

    logical function test_unallocated_after_deallocate()
        ! allocated is false again after deallocate; the program exits 0.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: box_t'//new_line('a')// &
            '    real, allocatable :: v'//new_line('a')// &
            '  end type box_t'//new_line('a')// &
            '  type(box_t) :: b'//new_line('a')// &
            '  allocate(b%v)'//new_line('a')// &
            '  b%v = 1.0'//new_line('a')// &
            '  deallocate(b%v)'//new_line('a')// &
            '  if (allocated(b%v)) error stop 3'//new_line('a')// &
            'end program main'

        test_unallocated_after_deallocate = expect_exit_status( &
            source, 0, '/tmp/ffc_derived_alloc_dealloc')
    end function test_unallocated_after_deallocate

end program test_session_derived_alloc_component_compiler
