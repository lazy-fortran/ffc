program test_session_e_descriptor_compiler
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session E/EN descriptor compiler test ==='

    all_passed = .true.
    if (.not. test_plain_e_descriptors()) all_passed = .false.
    if (.not. test_en_descriptors()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: E and EN descriptors lower through direct LIRIC session'

contains

    logical function test_plain_e_descriptors()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  print ''(E15.4)'', 12345.678'//new_line('a')// &
            '  print ''(E12.3)'', 0.000123'//new_line('a')// &
            '  print ''(E13.5)'', 0.0'//new_line('a')// &
            '  print ''(E15.4)'', -12345.678'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '     0.1235E+05'//new_line('a')// &
            '   0.123E-03'//new_line('a')// &
            '  0.00000E+00'//new_line('a')// &
            '    -0.1235E+05'//new_line('a')

        test_plain_e_descriptors = expect_output(source, expected, &
                                                 '/tmp/ffc_fmt_e_test')
    end function test_plain_e_descriptors

    logical function test_en_descriptors()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  print ''(EN15.4)'', 12345.678'//new_line('a')// &
            '  print ''(EN15.4)'', 12345678.0'//new_line('a')// &
            '  print ''(EN12.3)'', 0.000123'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '    12.3457E+03'//new_line('a')// &
            '    12.3457E+06'//new_line('a')// &
            ' 123.000E-06'//new_line('a')

        test_en_descriptors = expect_output(source, expected, &
                                            '/tmp/ffc_fmt_en_test')
    end function test_en_descriptors

end program test_session_e_descriptor_compiler
