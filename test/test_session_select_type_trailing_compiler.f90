program test_session_select_type_trailing_compiler
    ! Issue #2811 / #273: a local class(*) allocatable, a typed allocate, and a
    ! select type whose arms reassign an SSA scalar that a trailing statement
    ! reads back. The trailing print must execute as a sibling of the construct
    ! and see the value the chosen arm wrote.
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed
    character(len=*), parameter :: nl = new_line('a')

    print *, '=== select type trailing-statement compiler test ==='

    all_passed = .true.
    if (.not. test_type_is_integer_then_print()) all_passed = .false.
    if (.not. test_class_default_then_print()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: select type binds arm values into the trailing statement'

contains

    logical function test_type_is_integer_then_print()
        ! allocate(integer :: x) selects the type is (integer) arm, which sets
        ! i = 1; the print after end select reads the merged i.
        character(len=*), parameter :: source = &
            'program main'//nl// &
            '  class(*), allocatable :: x'//nl// &
            '  integer :: i'//nl// &
            '  i = 0'//nl// &
            '  allocate (integer :: x)'//nl// &
            '  select type (x)'//nl// &
            '  type is (integer)'//nl// &
            '    i = 1'//nl// &
            '  class default'//nl// &
            '    i = 2'//nl// &
            '  end select'//nl// &
            '  print *, i'//nl// &
            'end program main'

        test_type_is_integer_then_print = expect_output( &
            source, '           1'//nl, '/tmp/ffc_select_type_trailing_int')
    end function test_type_is_integer_then_print

    logical function test_class_default_then_print()
        ! allocate(real :: x) matches no type is (integer) guard, so the class
        ! default arm sets i = 2; the trailing print reads the merged i.
        character(len=*), parameter :: source = &
            'program main'//nl// &
            '  class(*), allocatable :: x'//nl// &
            '  integer :: i'//nl// &
            '  i = 0'//nl// &
            '  allocate (real :: x)'//nl// &
            '  select type (x)'//nl// &
            '  type is (integer)'//nl// &
            '    i = 1'//nl// &
            '  class default'//nl// &
            '    i = 2'//nl// &
            '  end select'//nl// &
            '  print *, i'//nl// &
            'end program main'

        test_class_default_then_print = expect_output( &
            source, '           2'//nl, '/tmp/ffc_select_type_trailing_def')
    end function test_class_default_then_print

end program test_session_select_type_trailing_compiler
