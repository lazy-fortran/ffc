program test_session_undefined_source_oracles_compiler
    ! Oracle set for workspace-plan Phase 0 "wrong output" claims that turned
    ! out not to be compiler defects. Three of the six cited corpus files
    ! cannot serve as byte-exact oracles as written:
    !
    !   control_flow_associate_construct.f90 declares `real :: x(10), y(10)`
    !   and never assigns them, so `z => x + y` reads undefined values
    !   (F2018 8.5.16) and every output conforms -- gfortran's own numbers are
    !   uninitialized stack garbage.
    !   issue_1324_if_inside_do_loop.f90 and do_concurrent_3_valid.f90 call
    !   random_number, so they are nondeterministic run to run even under
    !   gfortran alone; the apparent "branch merge inside DO" and "DO
    !   CONCURRENT locality" defects were random threshold crossings.
    !
    ! Each case here keeps the construct the corpus file was built to exercise
    ! -- an ASSOCIATE with an expression selector, nested IF inside DO plus a
    ! nested inner loop, and DO CONCURRENT with a scalar locality source -- but
    ! drives it from defined values, so both frontends must agree byte for
    ! byte. The fourth case pins issue_2349_data_implied_do, which is defined
    ! and now matches after the DATA zero-fill chain (ffc 32b358e, 5d1ad4d,
    ! 0aa1de7).
    !
    ! Nothing here asserts the *values* the undefined sources happen to print.
    use ffc_test_support, only: expect_output_matches_gfortran
    implicit none

    logical :: all_passed

    print *, '=== defined-source oracles for retired wrong-output claims ==='

    all_passed = .true.
    if (.not. test_associate_expression_selector_defined()) all_passed = .false.
    if (.not. test_nested_if_in_do_defined()) all_passed = .false.
    if (.not. test_do_concurrent_scalar_locality_defined()) all_passed = .false.
    if (.not. test_data_implied_do_partial_init()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: associate, nested-if-in-do, do concurrent and DATA ' // &
        'implied-do agree with gfortran on defined sources'

contains

    logical function test_associate_expression_selector_defined()
        ! ASSOCIATE binding a whole-array expression; x and y are defined so
        ! the printed sums are specified rather than undefined.
        character(len=*), parameter :: source = &
            'program assoc_expr'//new_line('a')// &
            '  real :: x(10), y(10)'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  do i = 1, 10'//new_line('a')// &
            '    x(i) = real(i)'//new_line('a')// &
            '    y(i) = real(i) * 0.5'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  associate (z => x + y)'//new_line('a')// &
            '    print *, z(1), z(5), z(10)'//new_line('a')// &
            '  end associate'//new_line('a')// &
            'end program assoc_expr'//new_line('a')

        test_associate_expression_selector_defined = &
            expect_output_matches_gfortran(source, 'assoc_expr_selector')
    end function test_associate_expression_selector_defined

    logical function test_nested_if_in_do_defined()
        ! The control-flow shape of issue_1324: a guard-print inside DO, a
        ! block IF wrapping an inline IF, and a nested inner loop with a
        ! guard. x is a deterministic function of the iteration, so the
        ! threshold crossings are fixed.
        character(len=*), parameter :: source = &
            'program if_inside_do'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  real :: x'//new_line('a')// &
            '  integer :: i, j, n'//new_line('a')// &
            '  n = 3'//new_line('a')// &
            '  do i = 1, n'//new_line('a')// &
            '    x = real(i) * 0.11'//new_line('a')// &
            '    print*, "x =", x'//new_line('a')// &
            '    if (x > 0.3) print*, "x larger than 0.3"'//new_line('a')// &
            '    if (x > 0.2) then'//new_line('a')// &
            '      if (x > 0.1) print*, "nested inline if"'//new_line('a')// &
            '      print*, "x larger than 0.2"'//new_line('a')// &
            '    end if'//new_line('a')// &
            '    do j = 1, 2'//new_line('a')// &
            '      if (j == 1) print*, "nested iteration", j'//new_line('a')// &
            '    end do'//new_line('a')// &
            '  end do'//new_line('a')// &
            'end program if_inside_do'//new_line('a')

        test_nested_if_in_do_defined = &
            expect_output_matches_gfortran(source, 'if_inside_do_defined')
    end function test_nested_if_in_do_defined

    logical function test_do_concurrent_scalar_locality_defined()
        ! DO CONCURRENT whose body only touches array(i) and a scalar read
        ! inside the loop bound region. val is constant, so array(i) is
        ! specified for every i and the print is a real locality check.
        character(len=*), parameter :: source = &
            'program do_concurrent_locality'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  real :: array(123), val'//new_line('a')// &
            '  val = 2.0'//new_line('a')// &
            '  do concurrent(i=1:123)'//new_line('a')// &
            '    array(i) = val*real(i)'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  print *, array(1), array(62), array(123)'//new_line('a')// &
            'end program do_concurrent_locality'//new_line('a')

        test_do_concurrent_scalar_locality_defined = &
            expect_output_matches_gfortran(source, 'do_concurrent_locality')
    end function test_do_concurrent_scalar_locality_defined

    logical function test_data_implied_do_partial_init()
        ! issue_2349 verbatim: a triangular implied-do object list takes 6
        ! values for 9 scalar objects, so the untouched tail must read zero
        ! (the DATA whole-array zero-fill). This one is defined output and is
        ! the only cited file that ever had a valid oracle.
        character(len=*), parameter :: source = &
            'program test_data_implied_do'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  real :: arr(3, 3)'//new_line('a')// &
            '  integer :: i, j'//new_line('a')// &
            '  data ((arr(i, j), i = 1, j), j = 1, 3) /6 * 1.0/'//new_line('a')// &
            '  print *, arr'//new_line('a')// &
            'end program test_data_implied_do'//new_line('a')

        test_data_implied_do_partial_init = &
            expect_output_matches_gfortran(source, 'data_implied_do_2349')
    end function test_data_implied_do_partial_init

end program test_session_undefined_source_oracles_compiler
