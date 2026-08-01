program test_session_accept_reject_false_01_compiler
    !! Valid programs that ffc used to reject with a spurious semantic error
    !! (lazy-fortran/ffc#581). gfortran compiles and runs all of them.
    use ffc_test_support, only: expect_exit_status, expect_error_contains
    implicit none

    logical :: all_passed

    print *, '=== spurious semantic rejection regression test ==='

    all_passed = .true.
    if (.not. test_block_data_equivalenced_into_common()) all_passed = .false.
    if (.not. test_block_data_local_data_object_rejected()) all_passed = .false.
    if (.not. test_empty_bind_c_type_accepted()) all_passed = .false.
    if (.not. test_continued_format_statement_accepted()) all_passed = .false.
    if (.not. test_unbalanced_format_still_rejected()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: valid specification parts and statements accepted'

contains

    logical function test_block_data_equivalenced_into_common()
        !! F2018 C8105 is satisfied through EQUIVALENCE association: 'la'
        !! shares storage with the COMMON member 'lb'.
        character(len=*), parameter :: source = &
            'block data bd_l'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  logical :: lb'//new_line('a')// &
            '  common /blockl/ lb(2)'//new_line('a')// &
            '  logical :: la(2)'//new_line('a')// &
            '  equivalence (la, lb)'//new_line('a')// &
            '  data la(2) / .false. /'//new_line('a')// &
            'end'//new_line('a')// &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  logical :: la(2)'//new_line('a')// &
            '  common /blockl/ la'//new_line('a')// &
            '  if (la(2)) stop 3'//new_line('a')// &
            '  stop 4'//new_line('a')// &
            'end program main'

        test_block_data_equivalenced_into_common = expect_exit_status( &
            source, 4, '/tmp/ffc_accept_581_blockdata')
    end function test_block_data_equivalenced_into_common

    logical function test_block_data_local_data_object_rejected()
        !! Negative control: a DATA object with no COMMON association at all
        !! is still rejected.
        character(len=*), parameter :: source = &
            'block data bd_l'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  logical :: lb'//new_line('a')// &
            '  common /blockl/ lb(2)'//new_line('a')// &
            '  logical :: la(2)'//new_line('a')// &
            '  data la(2) / .false. /'//new_line('a')// &
            'end'//new_line('a')// &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  stop 4'//new_line('a')// &
            'end program main'

        test_block_data_local_data_object_rejected = expect_error_contains( &
            source, 'must be in COMMON', '/tmp/ffc_accept_581_blockdata_neg')
    end function test_block_data_local_data_object_rejected

    logical function test_empty_bind_c_type_accepted()
        !! A component-less BIND(C) type is legal; gfortran only warns.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type, bind(C) :: junk'//new_line('a')// &
            '  end type junk'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  stop 5'//new_line('a')// &
            'end program main'

        test_empty_bind_c_type_accepted = expect_exit_status( &
            source, 5, '/tmp/ffc_accept_581_bindc')
    end function test_empty_bind_c_type_accepted

    logical function test_continued_format_statement_accepted()
        !! The format specification is complete once the continuation lines
        !! are joined, so it must not be diagnosed as unbalanced.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '35043 format (" ",16X,"COMPUTED: ",22X,1P/26X,F5.4,3X," ",&'// &
            new_line('a')// &
            '           (23X,F6.2),3X)'//new_line('a')// &
            '  stop 6'//new_line('a')// &
            'end program main'

        test_continued_format_statement_accepted = expect_exit_status( &
            source, 6, '/tmp/ffc_accept_581_format')
    end function test_continued_format_statement_accepted

    logical function test_unbalanced_format_still_rejected()
        !! Negative control: joining continuations must not hide a genuinely
        !! unbalanced format specification.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '100 format ("a",&'//new_line('a')// &
            '           "b"'//new_line('a')// &
            '  stop 6'//new_line('a')// &
            'end program main'

        test_unbalanced_format_still_rejected = expect_error_contains( &
            source, 'unbalanced parenthes', '/tmp/ffc_accept_581_format_neg')
    end function test_unbalanced_format_still_rejected

end program test_session_accept_reject_false_01_compiler
