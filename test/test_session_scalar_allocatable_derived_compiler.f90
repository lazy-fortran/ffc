program test_session_scalar_allocatable_derived_compiler
    ! Scalar allocatable derived variables: declare, allocate (bare and
    ! type-spec forms), read/write components through the heap instance,
    ! allocated(), deallocate(), default component init, whole-scalar copy,
    ! and auto-allocation on assignment to an unallocated target.
    use ffc_test_support, only: expect_exit_status, expect_output
    implicit none

    logical :: all_passed

    print *, '=== scalar allocatable derived compiler test ==='

    all_passed = .true.
    if (.not. test_allocate_component_rw()) all_passed = .false.
    if (.not. test_allocated_lifecycle()) all_passed = .false.
    if (.not. test_default_component_init()) all_passed = .false.
    if (.not. test_typespec_allocate()) all_passed = .false.
    if (.not. test_whole_scalar_copy()) all_passed = .false.
    if (.not. test_auto_allocate_on_assign()) all_passed = .false.
    if (.not. test_scalar_move_alloc()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: scalar allocatable derived lower through direct LIRIC'

contains

    logical function test_allocate_component_rw()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: point_t'//new_line('a')// &
            '    integer :: x'//new_line('a')// &
            '    integer :: y'//new_line('a')// &
            '  end type'//new_line('a')// &
            '  type(point_t), allocatable :: p'//new_line('a')// &
            '  allocate(p)'//new_line('a')// &
            '  p%x = 7'//new_line('a')// &
            '  p%y = 11'//new_line('a')// &
            '  print *, p%x + p%y'//new_line('a')// &
            'end program main'

        test_allocate_component_rw = expect_output( &
            source, '          18'//new_line('a'), &
            '/tmp/ffc_sad_rw_test')
    end function test_allocate_component_rw

    logical function test_allocated_lifecycle()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: box_t'//new_line('a')// &
            '    integer :: v'//new_line('a')// &
            '  end type'//new_line('a')// &
            '  type(box_t), allocatable :: b'//new_line('a')// &
            '  if (allocated(b)) stop 1'//new_line('a')// &
            '  allocate(b)'//new_line('a')// &
            '  if (.not. allocated(b)) stop 2'//new_line('a')// &
            '  deallocate(b)'//new_line('a')// &
            '  if (allocated(b)) stop 3'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_allocated_lifecycle = expect_exit_status( &
            source, 0, '/tmp/ffc_sad_lifecycle_test')
    end function test_allocated_lifecycle

    logical function test_default_component_init()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: val_t'//new_line('a')// &
            '    integer :: origin = 3'//new_line('a')// &
            '  end type'//new_line('a')// &
            '  type(val_t), allocatable :: val'//new_line('a')// &
            '  allocate(val)'//new_line('a')// &
            '  if (val%origin /= 3) stop 1'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_default_component_init = expect_exit_status( &
            source, 0, '/tmp/ffc_sad_default_test')
    end function test_default_component_init

    logical function test_typespec_allocate()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: pair_t'//new_line('a')// &
            '    integer :: a'//new_line('a')// &
            '    integer :: b'//new_line('a')// &
            '  end type'//new_line('a')// &
            '  type(pair_t), allocatable :: q'//new_line('a')// &
            '  allocate(pair_t :: q)'//new_line('a')// &
            '  q%a = 4'//new_line('a')// &
            '  q%b = 5'//new_line('a')// &
            '  print *, q%a * q%b'//new_line('a')// &
            'end program main'

        test_typespec_allocate = expect_output( &
            source, '          20'//new_line('a'), &
            '/tmp/ffc_sad_typespec_test')
    end function test_typespec_allocate

    logical function test_whole_scalar_copy()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: rec_t'//new_line('a')// &
            '    integer :: n'//new_line('a')// &
            '  end type'//new_line('a')// &
            '  type(rec_t), allocatable :: s, t'//new_line('a')// &
            '  allocate(s)'//new_line('a')// &
            '  s%n = 42'//new_line('a')// &
            '  allocate(t)'//new_line('a')// &
            '  t = s'//new_line('a')// &
            '  print *, t%n'//new_line('a')// &
            'end program main'

        test_whole_scalar_copy = expect_output( &
            source, '          42'//new_line('a'), &
            '/tmp/ffc_sad_copy_test')
    end function test_whole_scalar_copy

    logical function test_auto_allocate_on_assign()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: myint_t'//new_line('a')// &
            '    integer :: m'//new_line('a')// &
            '  end type'//new_line('a')// &
            '  type(myint_t), allocatable :: ins'//new_line('a')// &
            '  ins = myint_t(44)'//new_line('a')// &
            '  if (.not. allocated(ins)) stop 1'//new_line('a')// &
            '  if (ins%m /= 44) stop 2'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_auto_allocate_on_assign = expect_exit_status( &
            source, 0, '/tmp/ffc_sad_autoalloc_test')
    end function test_auto_allocate_on_assign

    logical function test_scalar_move_alloc()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: node_t'//new_line('a')// &
            '    integer :: k'//new_line('a')// &
            '  end type'//new_line('a')// &
            '  type(node_t), allocatable :: from, to'//new_line('a')// &
            '  allocate(from)'//new_line('a')// &
            '  from%k = 99'//new_line('a')// &
            '  call move_alloc(from, to)'//new_line('a')// &
            '  if (allocated(from)) stop 1'//new_line('a')// &
            '  if (.not. allocated(to)) stop 2'//new_line('a')// &
            '  if (to%k /= 99) stop 3'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_scalar_move_alloc = expect_exit_status( &
            source, 0, '/tmp/ffc_sad_move_alloc_test')
    end function test_scalar_move_alloc

end program test_session_scalar_allocatable_derived_compiler
