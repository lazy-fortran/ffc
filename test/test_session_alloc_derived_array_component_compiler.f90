program test_session_alloc_derived_array_component_compiler
    ! Rank-1 allocatable arrays of a derived element type through the direct
    ! LIRIC session: allocate(obj%arr(n)) heap-allocates n inner instances via
    ! calloc, allocated/size read the inline 16-byte descriptor, per-element
    ! component read and write obj%arr(i)%field address the heap element through
    ! the component data pointer, and nested allocatable-derived-array components
    ! stack (obj%z(i)%arr(j)) reaching deeper heap elements. Element storage is
    ! zero-initialised so nested allocatable component descriptors start
    ! unallocated. Covers the allocate_34 nesting shape from the lfortran corpus.
    use ffc_test_support, only: expect_output, expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== direct session allocatable derived array component test ==='

    all_passed = .true.
    if (.not. test_element_read_write()) all_passed = .false.
    if (.not. test_lifecycle_exit_status()) all_passed = .false.
    if (.not. test_nested_allocatable_stack()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: allocatable derived array components lower through session'

contains

    logical function test_element_read_write()
        ! allocate a derived array component, write and read scalar fields of
        ! individual heap elements, and print a field sum.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: vertex_t'//new_line('a')// &
            '    integer :: id'//new_line('a')// &
            '  end type vertex_t'//new_line('a')// &
            '  type :: graph_t'//new_line('a')// &
            '    type(vertex_t), allocatable :: v(:)'//new_line('a')// &
            '  end type graph_t'//new_line('a')// &
            '  type(graph_t) :: g'//new_line('a')// &
            '  integer :: i, s'//new_line('a')// &
            '  allocate(g%v(4))'//new_line('a')// &
            '  do i = 1, 4'//new_line('a')// &
            '    g%v(i)%id = i * 10'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  s = 0'//new_line('a')// &
            '  do i = 1, size(g%v)'//new_line('a')// &
            '    s = s + g%v(i)%id'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  print *, s'//new_line('a')// &
            'end program main'

        test_element_read_write = expect_output( &
            source, '         100'//new_line('a'), &
            '/tmp/ffc_alloc_derived_arr_rw')
    end function test_element_read_write

    logical function test_lifecycle_exit_status()
        ! Default-unallocated state, allocate, allocated/size, element write and
        ! read, then deallocate returns the descriptor to unallocated.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: item_t'//new_line('a')// &
            '    integer :: a'//new_line('a')// &
            '    real :: b'//new_line('a')// &
            '  end type item_t'//new_line('a')// &
            '  type :: bag_t'//new_line('a')// &
            '    type(item_t), allocatable :: items(:)'//new_line('a')// &
            '  end type bag_t'//new_line('a')// &
            '  type(bag_t) :: bag'//new_line('a')// &
            '  if (allocated(bag%items)) error stop 1'//new_line('a')// &
            '  allocate(bag%items(3))'//new_line('a')// &
            '  if (.not. allocated(bag%items)) error stop 2'//new_line('a')// &
            '  if (size(bag%items) /= 3) error stop 3'//new_line('a')// &
            '  bag%items(2)%a = 42'//new_line('a')// &
            '  bag%items(2)%b = 2.5'//new_line('a')// &
            '  if (bag%items(2)%a /= 42) error stop 4'//new_line('a')// &
            '  if (abs(bag%items(2)%b - 2.5) > 1.0e-6) error stop 5'//new_line('a')// &
            '  deallocate(bag%items)'//new_line('a')// &
            '  if (allocated(bag%items)) error stop 6'//new_line('a')// &
            'end program main'

        test_lifecycle_exit_status = expect_exit_status( &
            source, 0, '/tmp/ffc_alloc_derived_arr_life')
    end function test_lifecycle_exit_status

    logical function test_nested_allocatable_stack()
        ! Nested allocatable derived array components: an allocatable derived
        ! array whose element type itself holds an allocatable derived array and,
        ! deeper, an intrinsic allocatable array. Zero-initialised heap elements
        ! keep inner descriptors unallocated until explicitly allocated, matching
        ! the allocate_34 corpus shape.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: t1'//new_line('a')// &
            '    integer, allocatable :: arr(:)'//new_line('a')// &
            '  end type t1'//new_line('a')// &
            '  type :: t2'//new_line('a')// &
            '    type(t1), allocatable :: arr(:)'//new_line('a')// &
            '  end type t2'//new_line('a')// &
            '  type :: t3'//new_line('a')// &
            '    type(t2), allocatable :: z(:)'//new_line('a')// &
            '  end type t3'//new_line('a')// &
            '  type(t3) :: obj'//new_line('a')// &
            '  if (allocated(obj%z)) error stop 1'//new_line('a')// &
            '  allocate(obj%z(2))'//new_line('a')// &
            '  if (allocated(obj%z(1)%arr)) error stop 2'//new_line('a')// &
            '  allocate(obj%z(1)%arr(3))'//new_line('a')// &
            '  if (size(obj%z(1)%arr) /= 3) error stop 3'//new_line('a')// &
            '  if (allocated(obj%z(1)%arr(1)%arr)) error stop 4'//new_line('a')// &
            '  allocate(obj%z(1)%arr(1)%arr(5))'//new_line('a')// &
            '  if (size(obj%z(1)%arr(1)%arr) /= 5) error stop 5'//new_line('a')// &
            'end program main'

        test_nested_allocatable_stack = expect_exit_status( &
            source, 0, '/tmp/ffc_alloc_derived_arr_nested')
    end function test_nested_allocatable_stack

end program test_session_alloc_derived_array_component_compiler
