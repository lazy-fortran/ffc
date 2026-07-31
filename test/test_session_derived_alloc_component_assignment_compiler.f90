program test_session_derived_alloc_component_assignment_compiler
    ! Intrinsic assignment of a derived scalar deep-copies its allocatable
    ! components: after y = x the destination owns its own storage, so writing
    ! through the source afterwards leaves the destination unchanged. Covers a
    ! scalar allocatable component, a rank-1 allocatable array component, an
    ! unallocated component (which stays unallocated in the destination), and a
    ! generic function result assigned into a declared variable.
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session derived allocatable component assignment test ==='

    all_passed = .true.
    if (.not. test_scalar_component_deep_copy()) all_passed = .false.
    if (.not. test_array_component_deep_copy()) all_passed = .false.
    if (.not. test_unallocated_component_stays_unallocated()) all_passed = .false.
    if (.not. test_function_result_assignment()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: derived assignment deep-copies allocatable components'

contains

    logical function test_scalar_component_deep_copy()
        ! b = a copies the scalar allocatable component's value into storage
        ! owned by b; a%v = 7 afterwards must not reach b%v.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: box_t'//new_line('a')// &
            '    integer, allocatable :: v'//new_line('a')// &
            '  end type box_t'//new_line('a')// &
            '  type(box_t) :: a, b'//new_line('a')// &
            '  allocate(a%v)'//new_line('a')// &
            '  a%v = 5'//new_line('a')// &
            '  b = a'//new_line('a')// &
            '  if (.not. allocated(b%v)) error stop 1'//new_line('a')// &
            '  a%v = 7'//new_line('a')// &
            '  print *, b%v'//new_line('a')// &
            '  print *, a%v'//new_line('a')// &
            'end program main'

        test_scalar_component_deep_copy = expect_output( &
            source, '           5'//new_line('a')//'           7'//new_line('a'), &
            '/tmp/ffc_derived_assign_scalar')
    end function test_scalar_component_deep_copy

    logical function test_array_component_deep_copy()
        ! Whole-derived copy of a rank-1 allocatable array component: b owns its
        ! own elements, so a%v(1) = 99 afterwards leaves b%v(1) at 1.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: box_t'//new_line('a')// &
            '    integer, allocatable :: v(:)'//new_line('a')// &
            '  end type box_t'//new_line('a')// &
            '  type(box_t) :: a, b'//new_line('a')// &
            '  allocate(a%v(3))'//new_line('a')// &
            '  a%v(1) = 1'//new_line('a')// &
            '  a%v(2) = 2'//new_line('a')// &
            '  a%v(3) = 3'//new_line('a')// &
            '  b = a'//new_line('a')// &
            '  if (.not. allocated(b%v)) error stop 1'//new_line('a')// &
            '  if (size(b%v) /= 3) error stop 2'//new_line('a')// &
            '  a%v(1) = 99'//new_line('a')// &
            '  print *, b%v(1), b%v(2), b%v(3)'//new_line('a')// &
            '  print *, a%v(1)'//new_line('a')// &
            'end program main'

        test_array_component_deep_copy = expect_output( &
            source, &
            '           1           2           3'//new_line('a')// &
            '          99'//new_line('a'), &
            '/tmp/ffc_derived_assign_array')
    end function test_array_component_deep_copy

    logical function test_unallocated_component_stays_unallocated()
        ! An unallocated source component leaves the destination unallocated;
        ! the destination can then be allocated independently.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: box_t'//new_line('a')// &
            '    integer :: id'//new_line('a')// &
            '    integer, allocatable :: v(:)'//new_line('a')// &
            '  end type box_t'//new_line('a')// &
            '  type(box_t) :: a, b'//new_line('a')// &
            '  a%id = 11'//new_line('a')// &
            '  b = a'//new_line('a')// &
            '  if (allocated(b%v)) error stop 1'//new_line('a')// &
            '  allocate(b%v(2))'//new_line('a')// &
            '  b%v(1) = 4'//new_line('a')// &
            '  b%v(2) = 6'//new_line('a')// &
            '  if (allocated(a%v)) error stop 2'//new_line('a')// &
            '  print *, b%id, b%v(1), b%v(2)'//new_line('a')// &
            'end program main'

        test_unallocated_component_stays_unallocated = expect_output( &
            source, '          11           4           6'//new_line('a'), &
            '/tmp/ffc_derived_assign_unalloc')
    end function test_unallocated_component_stays_unallocated

    logical function test_function_result_assignment()
        ! A module function result carrying an allocatable array component
        ! assigns into a declared variable with its own storage.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  type :: box_t'//new_line('a')// &
            '    integer, allocatable :: v(:)'//new_line('a')// &
            '  end type box_t'//new_line('a')// &
            'contains'//new_line('a')// &
            '  function make() result(r)'//new_line('a')// &
            '    type(box_t) :: r'//new_line('a')// &
            '    allocate(r%v(2))'//new_line('a')// &
            '    r%v(1) = 4'//new_line('a')// &
            '    r%v(2) = 5'//new_line('a')// &
            '  end function make'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  type(box_t) :: b'//new_line('a')// &
            '  b = make()'//new_line('a')// &
            '  print *, b%v(1), b%v(2)'//new_line('a')// &
            'end program main'

        test_function_result_assignment = expect_output( &
            source, '           4           5'//new_line('a'), &
            '/tmp/ffc_derived_assign_result')
    end function test_function_result_assignment

end program test_session_derived_alloc_component_assignment_compiler
