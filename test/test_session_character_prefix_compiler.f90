program test_session_character_prefix_compiler
    use ffc_test_support, only: expect_output_matches_gfortran, expect_no_leaks, &
                                expect_error_contains
    implicit none
    character(len=:), allocatable :: source

    ! Prefix and body declarations must use the same declared-width ABI.
    ! The dummy n deliberately shadows the host n to check binding identity.
    source = &
        'program main'//new_line('a')// &
        '  integer :: n'//new_line('a')// &
        '  character(:), allocatable :: s'//new_line('a')// &
        '  n = 5'//new_line('a')// &
        '  s = prefix()'//new_line('a')// &
        '  print *, len(s), "[", s, "]"'//new_line('a')// &
        '  s = declared()'//new_line('a')// &
        '  print *, len(s), "[", s, "]"'//new_line('a')// &
        '  n = 2'//new_line('a')// &
        '  s = prefix()'//new_line('a')// &
        '  print *, len(s), "[", s, "]"'//new_line('a')// &
        '  s = declared()'//new_line('a')// &
        '  print *, len(s), "[", s, "]"'//new_line('a')// &
        '  s = implicit_result(4)'//new_line('a')// &
        '  print *, len(s), "[", s, "]"'//new_line('a')// &
        '  s = implicit_result(0)'//new_line('a')// &
        '  print *, len(s), "[", s, "]"'//new_line('a')// &
        '  s = fixed()'//new_line('a')// &
        '  print *, len(s), "[", s, "]"'//new_line('a')// &
        '  deallocate(s)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  character(n) function prefix() result(r)'//new_line('a')// &
        '    r = "abc"'//new_line('a')// &
        '  end function prefix'//new_line('a')// &
        '  function declared() result(r)'//new_line('a')// &
        '    character(len=n) :: r'//new_line('a')// &
        '    r = "abc"'//new_line('a')// &
        '  end function declared'//new_line('a')// &
        '  CHARACTER ( LEN = n ) FUNCTION implicit_result(n)'//new_line('a')// &
        '    integer, intent(in) :: n'//new_line('a')// &
        '    implicit_result = "xyz"'//new_line('a')// &
        '  end function implicit_result'//new_line('a')// &
        '  character(5) function fixed() result(r)'//new_line('a')// &
        '    r = "ab"'//new_line('a')// &
        '  end function fixed'//new_line('a')// &
        'end program main'

    if (.not. expect_output_matches_gfortran(source, &
                                             'character_function_prefix')) stop 1
    if (.not. expect_no_leaks(source, &
                              'test/ffc_character_prefix_ownership')) stop 1
    source = constant_width_source('5_8')
    if (.not. expect_output_matches_gfortran(source, &
        'character_prefix_constant_width')) stop 1
    source = constant_width_source('-4294967295_8')
    if (.not. expect_error_contains(source, 'character length', &
        'test/ffc_character_prefix_negative_constant_width')) stop 1
    source = constant_width_source('0_8')
    if (.not. expect_error_contains(source, 'character length', &
        'test/ffc_character_prefix_zero_constant_width')) stop 1
    source = constant_width_source('4294967297_8')
    if (.not. expect_error_contains(source, &
        'character length exceeds supported maximum 2147483647', &
        'test/ffc_character_prefix_constant_width_limit')) stop 1
    print *, 'PASS: character function prefixes match explicit declarations'

contains

    function constant_width_source(width) result(source)
        character(len=*), intent(in) :: width
        character(len=:), allocatable :: source

        source = &
            'module m'//new_line('a')// &
            '  integer(8), parameter :: n = '//width//new_line('a')// &
            'contains'//new_line('a')// &
            '  character(n) function prefix() result(r)'//new_line('a')// &
            '    r = "x"'//new_line('a')// &
            '  end function prefix'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  print *, len(prefix())'//new_line('a')// &
            'end program main'
    end function constant_width_source
end program test_session_character_prefix_compiler
