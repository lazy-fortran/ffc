program test_session_corpus_gaps_609_compiler
    !! Behavioural coverage for the ffc#609 corpus gaps that this change
    !! closes: the positional character(N) type-spec, ALLOCATE on a scalar
    !! POINTER, a list-directed I/O implied-do, and SELECT TYPE on a
    !! class(t) scalar allocatable.
    use ffc_test_support, only: expect_exit_status, expect_output
    implicit none

    logical :: all_passed

    print *, '=== ffc#609 corpus gap compiler test ==='

    all_passed = .true.
    if (.not. test_character_positional_length_function()) all_passed = .false.
    if (.not. test_character_positional_length_variable()) all_passed = .false.
    if (.not. test_allocate_scalar_pointer_result()) all_passed = .false.
    if (.not. test_io_implied_do_print()) all_passed = .false.
    if (.not. test_io_implied_do_print_with_step()) all_passed = .false.
    if (.not. test_select_type_allocatable_class_child()) all_passed = .false.
    if (.not. test_select_type_allocatable_class_base()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: ffc#609 corpus gaps compile and run'

contains

    logical function test_character_positional_length_function()
        ! character(1) function f(): the length is positional, not len=N.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  print *, f()'//new_line('a')// &
            'contains'//new_line('a')// &
            '  character(1) function f()'//new_line('a')// &
            "     f = 'a'"//new_line('a')// &
            '  end function f'//new_line('a')// &
            'end program main'

        test_character_positional_length_function = expect_output( &
            source, ' a'//new_line('a'), '/tmp/ffc_session_609_char_fn')
    end function test_character_positional_length_function

    logical function test_character_positional_length_variable()
        ! character(3) :: s declares a length-3 scalar, exactly as len=3 does.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(3) :: s'//new_line('a')// &
            "  s = 'abc'"//new_line('a')// &
            '  print *, s'//new_line('a')// &
            'end program main'

        test_character_positional_length_variable = expect_output( &
            source, ' abc'//new_line('a'), '/tmp/ffc_session_609_char_var')
    end function test_character_positional_length_variable

    logical function test_allocate_scalar_pointer_result()
        ! allocate(res) on a scalar POINTER gives it fresh heap storage.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, pointer :: res'//new_line('a')// &
            '  allocate (res)'//new_line('a')// &
            '  res = 42'//new_line('a')// &
            '  stop res'//new_line('a')// &
            'end program main'

        test_allocate_scalar_pointer_result = expect_exit_status( &
            source, 42, '/tmp/ffc_session_609_ptr_alloc')
    end function test_allocate_scalar_pointer_result

    logical function test_io_implied_do_print()
        ! print *, (i, i = 1, 4) writes one list-directed item per iteration.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  print *, (i, i = 1, 4)'//new_line('a')// &
            'end program main'

        test_io_implied_do_print = expect_output( &
            source, '           1           2           3           4'// &
            new_line('a'), &
            '/tmp/ffc_session_609_implied_do')
    end function test_io_implied_do_print

    logical function test_io_implied_do_print_with_step()
        ! A non-unit stride selects the same trip count gfortran does.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  print *, (i * 2, i = 1, 5, 2)'//new_line('a')// &
            'end program main'

        test_io_implied_do_print_with_step = expect_output( &
            source, '           2           6          10'//new_line('a'), &
            '/tmp/ffc_session_609_implied_do_step')
    end function test_io_implied_do_print_with_step

    logical function test_select_type_allocatable_class_child()
        ! SELECT TYPE on class(t), allocatable resolves the allocated
        ! extension type at run time, not the declared type.
        test_select_type_allocatable_class_child = expect_exit_status( &
            allocatable_class_source('child'), 2, &
            '/tmp/ffc_session_609_st_alloc_child')
    end function test_select_type_allocatable_class_child

    logical function test_select_type_allocatable_class_base()
        ! The same program allocating the declared type takes the base arm.
        test_select_type_allocatable_class_base = expect_exit_status( &
            allocatable_class_source('base'), 1, &
            '/tmp/ffc_session_609_st_alloc_base')
    end function test_select_type_allocatable_class_base

    function allocatable_class_source(allocated_type) result(source)
        character(len=*), intent(in) :: allocated_type
        character(len=:), allocatable :: source

        source = &
            'program main'//new_line('a')// &
            '  type :: base'//new_line('a')// &
            '     integer :: i'//new_line('a')// &
            '  end type base'//new_line('a')// &
            '  type, extends(base) :: child'//new_line('a')// &
            '     integer :: j'//new_line('a')// &
            '  end type child'//new_line('a')// &
            '  class(base), allocatable :: x'//new_line('a')// &
            '  integer :: r'//new_line('a')// &
            '  allocate ('//trim(allocated_type)//' :: x)'//new_line('a')// &
            '  r = 0'//new_line('a')// &
            '  select type (x)'//new_line('a')// &
            '  type is (child)'//new_line('a')// &
            '     r = 2'//new_line('a')// &
            '  type is (base)'//new_line('a')// &
            '     r = 1'//new_line('a')// &
            '  class default'//new_line('a')// &
            '     r = 9'//new_line('a')// &
            '  end select'//new_line('a')// &
            '  stop r'//new_line('a')// &
            'end program main'
    end function allocatable_class_source

end program test_session_corpus_gaps_609_compiler
