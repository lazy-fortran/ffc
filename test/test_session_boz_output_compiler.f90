! fo: dispatcher
module ffc_case_test_session_boz_output_compiler
    implicit none
    private
    public :: case_test_session_boz_output_compiler
    interface
        subroutine case_test_session_boz_output_compiler()
        end subroutine case_test_session_boz_output_compiler
    end interface
end module ffc_case_test_session_boz_output_compiler

subroutine case_test_session_boz_output_compiler()
    use ffc_test_support, only: expect_output
    implicit none
    save

    logical :: all_passed

    all_passed = .true.
    if (.not. test_mixed_fields()) all_passed = .false.
    if (.not. test_zero_and_overflow()) all_passed = .false.
    if (.not. test_negative_storage_width()) all_passed = .false.
    if (.not. test_wide_expression()) all_passed = .false.
    if (.not. all_passed) stop 1
    print *, 'PASS: B/O/Z integer output matches measured gfortran bytes'

contains

    logical function test_mixed_fields()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  print "(B8,1X,O5,1X,Z4)", 5, 15, 15'//new_line('a')// &
            '  write(*,"(2(Z4))") 15, 255'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '     101    17    F'//new_line('a')// &
            '   F  FF'//new_line('a')

        test_mixed_fields = expect_output(source, expected, &
            '/var/tmp/ffc_boz_mixed')
    end function test_mixed_fields

    logical function test_zero_and_overflow()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  print "(B0.0,O4.0,Z0.5)", 0, 0, 0'//new_line('a')// &
            '  print "(B3,O1,Z1)", 15, 8, 16'//new_line('a')// &
            '  print "(Z8.5)", 15'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '     00000'//new_line('a')// &
            '*****'//new_line('a')// &
            '   0000F'//new_line('a')

        test_zero_and_overflow = expect_output(source, expected, &
            '/var/tmp/ffc_boz_zero_overflow')
    end function test_zero_and_overflow

    logical function test_negative_storage_width()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer(1) :: a'//new_line('a')// &
            '  integer(2) :: b'//new_line('a')// &
            '  integer(4) :: c'//new_line('a')// &
            '  integer(8) :: d'//new_line('a')// &
            '  a=-1_1'//new_line('a')// &
            '  b=-1_2'//new_line('a')// &
            '  c=-1_4'//new_line('a')// &
            '  d=-1_8'//new_line('a')// &
            '  print "(Z2,1X,Z4,1X,Z8,1X,Z16)", a, b, c, d'//new_line('a')// &
            '  print "(B8,1X,O3,1X,Z4)", a, a, c'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            'FF FFFF FFFFFFFF FFFFFFFFFFFFFFFF'//new_line('a')// &
            '11111111 377 ****'//new_line('a')

        test_negative_storage_width = expect_output(source, expected, &
            '/var/tmp/ffc_boz_negative_kinds')
    end function test_negative_storage_width

    logical function test_wide_expression()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer(8) :: n'//new_line('a')// &
            '  n=4294967296_8'//new_line('a')// &
            '  print "(Z0)", n+1_8'//new_line('a')// &
            '  print "(Z2,Z16)", -1_1, -1_8'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '100000001'//new_line('a')// &
            'FFFFFFFFFFFFFFFFFF'//new_line('a')

        test_wide_expression = expect_output(source, expected, &
            '/var/tmp/ffc_boz_wide_expression')
    end function test_wide_expression

end subroutine case_test_session_boz_output_compiler
