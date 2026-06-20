program test_session_fixed_concat_compiler
    ! Runtime concatenation of a // chain with variable and intrinsic operands
    ! into a fixed-length character target. FortFront's lazy standardization
    ! sizes the target to the exact concatenated length, so each leaf operand
    ! contributes its full declared length, matching gfortran.
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session fixed-length concat compiler test ==='

    all_passed = .true.
    if (.not. test_variable_concat()) all_passed = .false.
    if (.not. test_literal_variable_chain()) all_passed = .false.
    if (.not. test_trim_concat()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: fixed-length // chains lower through direct LIRIC session'

contains

    logical function test_variable_concat()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=5) :: a'//new_line('a')// &
            '  character(len=5) :: b'//new_line('a')// &
            '  character(len=10) :: r'//new_line('a')// &
            '  a = "Hello"'//new_line('a')// &
            '  b = "World"'//new_line('a')// &
            '  r = a // b'//new_line('a')// &
            '  print *, r'//new_line('a')// &
            'end program main'

        test_variable_concat = expect_output( &
            source, ' HelloWorld'//new_line('a'), &
            '/tmp/ffc_fixed_concat_var_test')
    end function test_variable_concat

    logical function test_literal_variable_chain()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=5) :: name'//new_line('a')// &
            '  character(len=13) :: greeting'//new_line('a')// &
            '  name = "World"'//new_line('a')// &
            '  greeting = "Hello, " // name // "!"'//new_line('a')// &
            '  print *, greeting'//new_line('a')// &
            'end program main'

        test_literal_variable_chain = expect_output( &
            source, ' Hello, World!'//new_line('a'), &
            '/tmp/ffc_fixed_concat_chain_test')
    end function test_literal_variable_chain

    logical function test_trim_concat()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=7) :: name'//new_line('a')// &
            '  character(len=2) :: version'//new_line('a')// &
            '  character(len=10) :: full'//new_line('a')// &
            '  name = "Fortran"'//new_line('a')// &
            '  version = "90"'//new_line('a')// &
            '  full = trim(name) // " " // trim(version)'//new_line('a')// &
            '  print *, full'//new_line('a')// &
            'end program main'

        test_trim_concat = expect_output( &
            source, ' Fortran 90'//new_line('a'), &
            '/tmp/ffc_fixed_concat_trim_test')
    end function test_trim_concat

end program test_session_fixed_concat_compiler
