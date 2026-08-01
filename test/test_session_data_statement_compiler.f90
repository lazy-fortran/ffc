program test_session_data_statement
    ! DATA statement lowering through direct LIRIC (#2349, #2251). DATA gives
    ! variables their initial value before execution, independent of textual
    ! position. Covers scalar and array initialisation, an executable
    ! assignment overriding DATA regardless of source order, a real array
    ! implied-do, and a hexadecimal BOZ constant.
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session DATA statement compiler test ==='

    all_passed = .true.
    if (.not. test_scalar_init()) all_passed = .false.
    if (.not. test_array_init()) all_passed = .false.
    if (.not. test_assignment_overrides_data()) all_passed = .false.
    if (.not. test_real_array_implied_do()) all_passed = .false.
    if (.not. test_boz_constant()) all_passed = .false.
    if (.not. test_strided_section_init()) all_passed = .false.
    if (.not. test_rank_two_section_and_repeat()) all_passed = .false.
    if (.not. test_repeated_value_init()) all_passed = .false.
    if (.not. test_nested_derived_constructor()) all_passed = .false.
    if (.not. test_empty_derived_constructor_data()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: DATA statements lower through direct LIRIC session'

contains

    logical function test_scalar_init()
        ! Multiple scalar objects take values in list order.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a, b'//new_line('a')// &
            '  data a, b /10, 20/'//new_line('a')// &
            '  print *, a, b'//new_line('a')// &
            'end program main'

        test_scalar_init = expect_output( &
            source, '          10          20'//new_line('a'), &
            '/tmp/ffc_data_scalar')
    end function test_scalar_init

    logical function test_array_init()
        ! An array object consumes one value per element in storage order.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: arr(3)'//new_line('a')// &
            '  data arr /1, 2, 3/'//new_line('a')// &
            '  print *, arr(1), arr(2), arr(3)'//new_line('a')// &
            'end program main'

        test_array_init = expect_output( &
            source, '           1           2           3'//new_line('a'), &
            '/tmp/ffc_data_array')
    end function test_array_init

    logical function test_assignment_overrides_data()
        ! DATA initialises before execution; an executable assignment that
        ! precedes the DATA statement textually still wins at run time.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x(2)'//new_line('a')// &
            '  x = 7'//new_line('a')// &
            '  data x /4, 5/'//new_line('a')// &
            '  print *, x(1), x(2)'//new_line('a')// &
            'end program main'

        test_assignment_overrides_data = expect_output( &
            source, '           7           7'//new_line('a'), &
            '/tmp/ffc_data_override')
    end function test_assignment_overrides_data

    logical function test_real_array_implied_do()
        ! Implied-do object: coeff(i)/coeff(i+2) resolve against the unrolled
        ! control value, matching gfortran storage order.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: coeff(4)'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  data (coeff(i), coeff(i+2), i=1,2) /1.0, 2.0, 3.0, 4.0/'// &
            new_line('a')// &
            '  print *, coeff(1), coeff(2), coeff(3), coeff(4)'//new_line('a')// &
            'end program main'

        test_real_array_implied_do = expect_output( &
            source, &
            '   1.00000000       3.00000000       2.00000000       4.00000000    '// &
            new_line('a'), &
            '/tmp/ffc_data_implied_do')
    end function test_real_array_implied_do

    logical function test_boz_constant()
        ! A hexadecimal BOZ initialiser decodes by radix (z'10' = 16).
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: k'//new_line('a')// &
            "  data k /z'10'/"//new_line('a')// &
            '  print *, k'//new_line('a')// &
            'end program main'

        test_boz_constant = expect_output( &
            source, '          16'//new_line('a'), &
            '/tmp/ffc_data_boz')
    end function test_boz_constant

    logical function test_strided_section_init()
        ! Array-section objects with a stride consume one value per selected
        ! element in element order; two sections interleave in one array.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: arr(6)'//new_line('a')// &
            '  data arr(1:5:2) /1, 2, 3/'//new_line('a')// &
            '  data arr(2:4:2) /7, 8/'//new_line('a')// &
            '  print *, arr(1), arr(2), arr(3), arr(4), arr(5)'//new_line('a')// &
            'end program main'

        test_strided_section_init = expect_output( &
            source, &
            '           1           7           2           8           3'// &
            new_line('a'), &
            '/tmp/ffc_data_section_stride')
    end function test_strided_section_init

    logical function test_rank_two_section_and_repeat()
        ! A section of a rank-2 array walks the declared column-major layout,
        ! and a repeated value expands over a real section.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: m(2,3)'//new_line('a')// &
            '  real :: r(4)'//new_line('a')// &
            '  data m(1,1:3) /10, 20, 30/'//new_line('a')// &
            '  data r(2:3) /2*1.5/'//new_line('a')// &
            '  print *, m(1,1), m(1,2), m(1,3)'//new_line('a')// &
            '  print *, r(2), r(3)'//new_line('a')// &
            'end program main'

        test_rank_two_section_and_repeat = expect_output( &
            source, &
            '          10          20          30'//new_line('a')// &
            '   1.50000000       1.50000000    '//new_line('a'), &
            '/tmp/ffc_data_section_rank2')
    end function test_rank_two_section_and_repeat

    logical function test_repeated_value_init()
        ! A repeat count supplies the same value to every element once.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(4)'//new_line('a')// &
            '  data a /4*3/'//new_line('a')// &
            '  print *, a(1), a(2), a(3), a(4)'//new_line('a')// &
            'end program main'

        test_repeated_value_init = expect_output( &
            source, &
            '           3           3           3           3'//new_line('a'), &
            '/tmp/ffc_data_repeat')
    end function test_repeated_value_init

    logical function test_nested_derived_constructor()
        ! A nested structure constructor initialises the derived object's
        ! component layout instead of reaching the scalar store path.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: inner_t'//new_line('a')// &
            '    integer :: k'//new_line('a')// &
            '  end type inner_t'//new_line('a')// &
            '  type :: outer_t'//new_line('a')// &
            '    type(inner_t) :: i'//new_line('a')// &
            '    integer :: j'//new_line('a')// &
            '  end type outer_t'//new_line('a')// &
            '  type(outer_t) :: o'//new_line('a')// &
            '  data o /outer_t(inner_t(3), 4)/'//new_line('a')// &
            '  print *, o%i%k, o%j'//new_line('a')// &
            'end program main'

        test_nested_derived_constructor = expect_output( &
            source, '           3           4'//new_line('a'), &
            '/tmp/ffc_data_derived_ctor')
    end function test_nested_derived_constructor

    logical function test_empty_derived_constructor_data()
        ! An empty structure constructor is a valid DATA value and leaves the
        ! zero-component object well-defined without emitting a scalar store.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: t'//new_line('a')// &
            '  end type t'//new_line('a')// &
            '  type(t) :: y'//new_line('a')// &
            '  integer :: marker'//new_line('a')// &
            '  data y /t()/'//new_line('a')// &
            '  marker = 17'//new_line('a')// &
            '  print *, marker'//new_line('a')// &
            'end program main'

        test_empty_derived_constructor_data = expect_output( &
            source, '          17'//new_line('a'), &
            '/tmp/ffc_data_empty_derived')
    end function test_empty_derived_constructor_data

end program test_session_data_statement
