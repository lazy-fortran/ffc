! fo: dispatcher
module ffc_case_test_session_io_implied_do_print_compiler
    implicit none
    private
    public :: case_test_session_io_implied_do_print_compiler
    interface
        subroutine case_test_session_io_implied_do_print_compiler()
        end subroutine case_test_session_io_implied_do_print_compiler
    end interface
end module ffc_case_test_session_io_implied_do_print_compiler

subroutine case_test_session_io_implied_do_print_compiler()
    ! List-directed print of an I/O implied-do, print *, (obj, ..., i = lo, hi).
    ! One dispatch path owns an implied-do item whether or not it is the only
    ! item in the statement, so the element type and the list-directed
    ! separator rule are decided in exactly one place. Every expectation is
    ! gfortran's output.
    use ffc_test_support, only: expect_output, expect_output_matches_gfortran, &
        expect_error_contains
    implicit none
    save

    logical :: all_passed

    print *, '=== direct session I/O implied-do print compiler test ==='

    all_passed = .true.
    if (.not. test_character_elements_print_as_characters()) all_passed = .false.
    if (.not. test_integer_elements_still_print_as_integers()) all_passed = .false.
    if (.not. test_implied_do_among_other_items()) all_passed = .false.
    if (.not. test_implied_do_with_step()) all_passed = .false.
    if (.not. test_array_constructor_implied_do()) all_passed = .false.
    if (.not. test_multiple_objects()) all_passed = .false.
    if (.not. test_nested_objects()) all_passed = .false.
    if (.not. test_formatted_objects_refused()) all_passed = .false.
    if (.not. test_other_io_objects_refused()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: I/O implied-do print lowers through one dispatch path'

contains

    logical function test_multiple_objects()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer :: i'//new_line('a')// &
            'print *, (i, -i, i=2,1,-1)'//new_line('a')// &
            'write(*, *) (i, -i, i=2,1,-1)'//new_line('a')// &
            'end program main'

        test_multiple_objects = expect_output_matches_gfortran(source, &
            'io_multiple_objects')
    end function test_multiple_objects

    logical function test_nested_objects()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer :: i, j'//new_line('a')// &
            'print *, ((i, -i, i=2,1,-1), j=1,2)'//new_line('a')// &
            'end program main'

        test_nested_objects = expect_output_matches_gfortran(source, &
            'io_nested_objects')
    end function test_nested_objects

    logical function test_formatted_objects_refused()
        ! Multi-object, nested and unit-* controls lowered byte-exactly by
        ! the flattened walk (73e8a92): pinned as outputs. Empty formats and
        ! implied-do beside other items stay refused, each named.
        test_formatted_objects_refused = expect_output( &
            'program main'//new_line('a')// &
            'integer :: i, j'//new_line('a')// &
            'print "(I0)", (i, -i, i=1,2)'//new_line('a')// &
            'end program main', '1'//new_line('a')//'-1'//new_line('a')// &
            '2'//new_line('a')//'-2'//new_line('a'), &
            '/tmp/ffc_fmt_multi_objects')
        if (.not. expect_output( &
            'program main'//new_line('a')// &
            'integer :: i, j'//new_line('a')// &
            'print "(I0)", ((i, i=1,2), j=1,2)'//new_line('a')// &
            'end program main', '1'//new_line('a')//'2'//new_line('a')// &
            '1'//new_line('a')//'2'//new_line('a'), &
            '/tmp/ffc_fmt_nested_objects')) then
            test_formatted_objects_refused = .false.
        end if
        if (.not. refuse_formatted('print "()", (i, -i, i=1,2)', 'empty', &
            'no data descriptor')) then
            test_formatted_objects_refused = .false.
        end if
        if (.not. refuse_formatted('print "(I0)", 9, (i, -i, i=1,2)', &
            'mixed', 'formatted I/O implied-do')) then
            test_formatted_objects_refused = .false.
        end if
        if (.not. expect_output( &
            'program main'//new_line('a')// &
            'integer :: i, j'//new_line('a')// &
            'write(*, "(I0)") (i, -i, i=1,2)'//new_line('a')// &
            'end program main', '1'//new_line('a')//'-1'//new_line('a')// &
            '2'//new_line('a')//'-2'//new_line('a'), &
            '/tmp/ffc_fmt_write_objects')) then
            test_formatted_objects_refused = .false.
        end if
    end function test_formatted_objects_refused

    logical function refuse_formatted(statement, stem, message) result(refused)
        character(len=*), intent(in) :: statement, stem, message
        character(len=:), allocatable :: source

        source = 'program main'//new_line('a')// &
            'integer :: i, j'//new_line('a')//statement//new_line('a')// &
            'end program main'
        refused = expect_error_contains(source, message, &
            '/tmp/ffc_formatted_'//stem//'_objects')
    end function refuse_formatted

    logical function test_other_io_objects_refused()
        character(len=*), parameter :: read_source = &
            'program main'//new_line('a')// &
            'integer :: i, a(2)'//new_line('a')// &
            'read *, (a(i), i=1,2)'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: write_source = &
            'program main'//new_line('a')// &
            'integer :: i'//new_line('a')// &
            'write(10, *) (i, -i, i=1,2)'//new_line('a')// &
            'end program main'

        test_other_io_objects_refused = expect_error_contains(read_source, &
            'expected identifier assignment target', '/var/tmp/ffc_read_io_objects')
        if (.not. expect_error_contains(write_source, &
            'only supports integer expressions', '/var/tmp/ffc_write_io_objects')) then
            test_other_io_objects_refused = .false.
        end if
    end function test_other_io_objects_refused

    logical function test_character_elements_print_as_characters()
        ! A character array element is a character value, so consecutive
        ! elements print concatenated with no separating blank between them.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=2) :: c(3)'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  c(1) = "ab"'//new_line('a')// &
            '  c(2) = "cd"'//new_line('a')// &
            '  c(3) = "ef"'//new_line('a')// &
            '  print *, (c(i), i=1,3)'//new_line('a')// &
            'end program main'

        test_character_elements_print_as_characters = expect_output( &
            source, ' abcdef'//new_line('a'), &
            '/tmp/ffc_session_implied_do_char_test')
    end function test_character_elements_print_as_characters

    logical function test_integer_elements_still_print_as_integers()
        ! Regression guard for the path this shares with characters: an
        ! integer implied-do keeps a separating blank before every value.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(3)'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  a(1) = 10'//new_line('a')// &
            '  a(2) = 20'//new_line('a')// &
            '  a(3) = 30'//new_line('a')// &
            '  print *, (a(i), i=1,3)'//new_line('a')// &
            'end program main'

        test_integer_elements_still_print_as_integers = expect_output( &
            source, &
            '          10          20          30'//new_line('a'), &
            '/tmp/ffc_session_implied_do_int_test')
    end function test_integer_elements_still_print_as_integers

    logical function test_implied_do_among_other_items()
        ! The same item in a statement with other items: the separator rule
        ! spans the boundary, so a character item either side of the
        ! implied-do joins its elements without a blank.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=3) :: c(2)'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  c(1) = "abc"'//new_line('a')// &
            '  c(2) = "def"'//new_line('a')// &
            '  print *, "tag", (c(i), i=1,2), "end"'//new_line('a')// &
            'end program main'

        test_implied_do_among_other_items = expect_output( &
            source, ' tagabcdefend'//new_line('a'), &
            '/tmp/ffc_session_implied_do_mixed_test')
    end function test_implied_do_among_other_items

    logical function test_implied_do_with_step()
        ! A non-unit step selects a subset of the elements, in loop order.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(5)'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  do i = 1, 5'//new_line('a')// &
            '    a(i) = i * 100'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  print *, (a(i), i=1,5,2)'//new_line('a')// &
            'end program main'

        test_implied_do_with_step = expect_output( &
            source, &
            '         100         300         500'//new_line('a'), &
            '/tmp/ffc_session_implied_do_step_test')
    end function test_implied_do_with_step

    logical function test_array_constructor_implied_do()
        ! Legacy (/ ... /) constructors wrap an implied-do in an array literal,
        ! rather than using the I/O implied-do node used by print *, ( ... ).
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  print *, (/(i, i=1,4)/)'//new_line('a')// &
            'end program main'

        test_array_constructor_implied_do = expect_output( &
            source, '           1           2           3           4'// &
            new_line('a'), '/tmp/ffc_session_array_ctor_implied_do')
    end function test_array_constructor_implied_do

end subroutine case_test_session_io_implied_do_print_compiler
