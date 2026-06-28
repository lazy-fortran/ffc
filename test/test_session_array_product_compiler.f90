program test_session_array_product
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session array product compiler test ==='

    all_passed = .true.
    if (.not. test_rank1_product()) all_passed = .false.
    if (.not. test_rank2_product()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: array product lowers through direct LIRIC'

contains

    logical function test_rank1_product()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(1:4)'//new_line('a')// &
            '  a = [2, 3, 4, 5]'//new_line('a')// &
            '  print *, product(a)'//new_line('a')// &
            'end program main'

        test_rank1_product = expect_output( &
            source, '         120'//new_line('a'), &
            '/tmp/ffc_session_array_product_rank1_test')
    end function test_rank1_product

    logical function test_rank2_product()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(1:2, 1:2)'//new_line('a')// &
            '  a = [1, 2, 3, 4]'//new_line('a')// &
            '  print *, product(a)'//new_line('a')// &
            'end program main'

        test_rank2_product = expect_output( &
            source, '          24'//new_line('a'), &
            '/tmp/ffc_session_array_product_rank2_test')
    end function test_rank2_product

end program test_session_array_product
