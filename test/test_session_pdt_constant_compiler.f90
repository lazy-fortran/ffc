program test_session_pdt_constant_compiler
    ! Parameterized derived types with constant integer KIND and LEN type
    ! parameters (#411). Each distinct tuple of actual type parameters is a
    ! distinct concrete type: the actuals are folded at compile time,
    ! substituted into component character lengths, array bounds, and kind
    ! selectors, and the resulting layout is cached under a mangled instance
    ! name so two declarations with the same actuals share it.
    ! Missing, nonconstant, and excess actual parameters are diagnosed.
    use ffc_test_support, only: expect_output, expect_exit_status, &
        expect_error_contains
    implicit none

    logical :: all_passed

    print *, '=== direct session parameterized derived type compiler test ==='

    all_passed = .true.
    if (.not. test_two_lengths_and_one_kind()) all_passed = .false.
    if (.not. test_module_pdt_instances()) all_passed = .false.
    if (.not. test_shared_instance_layout()) all_passed = .false.
    if (.not. test_nonconstant_actual_rejected()) all_passed = .false.
    if (.not. test_missing_actual_rejected()) all_passed = .false.
    if (.not. test_excess_actuals_rejected()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: constant PDT type parameters lower through direct LIRIC session'

contains

    logical function test_two_lengths_and_one_kind()
        ! Two instances of one PDT with different LEN and KIND actuals. LEN
        ! drives a character component length and an array component bound;
        ! KIND drives a real component kind. Output matches gfortran.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: box_t(n, k)'//new_line('a')// &
            '    integer, len :: n'//new_line('a')// &
            '    integer, kind :: k = 4'//new_line('a')// &
            '    character(len=n) :: label'//new_line('a')// &
            '    real(k) :: value'//new_line('a')// &
            '    integer :: items(n)'//new_line('a')// &
            '  end type box_t'//new_line('a')// &
            '  type(box_t(3, 4)) :: a'//new_line('a')// &
            '  type(box_t(5, 8)) :: b'//new_line('a')// &
            '  a%label = "abc"'//new_line('a')// &
            '  b%label = "hello"'//new_line('a')// &
            '  a%value = 0.5'//new_line('a')// &
            '  b%value = 0.25'//new_line('a')// &
            '  a%items(3) = 30'//new_line('a')// &
            '  b%items(5) = 50'//new_line('a')// &
            '  print *, len(a%label), len(b%label)'//new_line('a')// &
            '  print *, a%label, " ", b%label'//new_line('a')// &
            '  print *, a%items(3) + b%items(5)'//new_line('a')// &
            '  if (abs(a%value - 0.5) > 1.0e-6) error stop 3'//new_line('a')// &
            '  if (abs(b%value - 0.25d0) > 1.0d-12) error stop 4'//new_line('a')// &
            'end program main'

        test_two_lengths_and_one_kind = expect_output( &
            source, &
            '           3           5'//new_line('a')// &
            ' abc hello'//new_line('a')// &
            '          80'//new_line('a'), &
            '/tmp/ffc_pdt_const_lengths')
    end function test_two_lengths_and_one_kind

    logical function test_module_pdt_instances()
        ! A PDT defined in a module and instantiated twice in the program with
        ! different LEN actuals.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  type :: buf_t(n)'//new_line('a')// &
            '    integer, len :: n'//new_line('a')// &
            '    character(len=n) :: text'//new_line('a')// &
            '    integer :: count'//new_line('a')// &
            '  end type buf_t'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  type(buf_t(4)) :: p'//new_line('a')// &
            '  type(buf_t(2)) :: r'//new_line('a')// &
            '  p%text = "abcd"'//new_line('a')// &
            '  r%text = "hi"'//new_line('a')// &
            '  p%count = 1'//new_line('a')// &
            '  r%count = 2'//new_line('a')// &
            '  print *, len(p%text), len(r%text), p%count + r%count'// &
            new_line('a')// &
            '  print *, p%text, r%text'//new_line('a')// &
            'end program main'

        test_module_pdt_instances = expect_output( &
            source, &
            '           4           2           3'//new_line('a')// &
            ' abcdhi'//new_line('a'), &
            '/tmp/ffc_pdt_const_module')
    end function test_module_pdt_instances

    logical function test_shared_instance_layout()
        ! Two declarations with the same actuals share one cached layout, and a
        ! third with different actuals keeps its own; assignment between the
        ! matching pair is well formed and the program exits 0.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: pair_t(n)'//new_line('a')// &
            '    integer, len :: n'//new_line('a')// &
            '    integer :: items(n)'//new_line('a')// &
            '  end type pair_t'//new_line('a')// &
            '  type(pair_t(2)) :: a'//new_line('a')// &
            '  type(pair_t(2)) :: b'//new_line('a')// &
            '  type(pair_t(4)) :: c'//new_line('a')// &
            '  a%items(1) = 11'//new_line('a')// &
            '  a%items(2) = 22'//new_line('a')// &
            '  b = a'//new_line('a')// &
            '  c%items(4) = 44'//new_line('a')// &
            '  if (b%items(2) /= 22) error stop 1'//new_line('a')// &
            '  if (c%items(4) /= 44) error stop 2'//new_line('a')// &
            'end program main'

        test_shared_instance_layout = expect_exit_status( &
            source, 0, '/tmp/ffc_pdt_const_shared')
    end function test_shared_instance_layout

    logical function test_nonconstant_actual_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: box_t(n)'//new_line('a')// &
            '    integer, len :: n'//new_line('a')// &
            '    integer :: items(n)'//new_line('a')// &
            '  end type box_t'//new_line('a')// &
            '  integer :: v'//new_line('a')// &
            '  type(box_t(v)) :: a'//new_line('a')// &
            '  a%items(1) = 1'//new_line('a')// &
            'end program main'

        test_nonconstant_actual_rejected = expect_error_contains( &
            source, 'must be a compile-time integer constant', &
            '/tmp/ffc_pdt_const_nonconst')
    end function test_nonconstant_actual_rejected

    logical function test_missing_actual_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: box_t(n)'//new_line('a')// &
            '    integer, len :: n'//new_line('a')// &
            '    integer :: items(n)'//new_line('a')// &
            '  end type box_t'//new_line('a')// &
            '  type(box_t) :: a'//new_line('a')// &
            '  a%items(1) = 1'//new_line('a')// &
            'end program main'

        test_missing_actual_rejected = expect_error_contains( &
            source, 'has no actual value and no default', &
            '/tmp/ffc_pdt_const_missing')
    end function test_missing_actual_rejected

    logical function test_excess_actuals_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: box_t(n)'//new_line('a')// &
            '    integer, len :: n'//new_line('a')// &
            '    integer :: items(n)'//new_line('a')// &
            '  end type box_t'//new_line('a')// &
            '  type(box_t(3, 4)) :: a'//new_line('a')// &
            '  a%items(1) = 1'//new_line('a')// &
            'end program main'

        test_excess_actuals_rejected = expect_error_contains( &
            source, 'too many actual type parameters', &
            '/tmp/ffc_pdt_const_excess')
    end function test_excess_actuals_rejected

end program test_session_pdt_constant_compiler
