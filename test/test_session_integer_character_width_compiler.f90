program test_session_integer_character_width_compiler
    use ffc_test_support, only: expect_output_matches_gfortran, &
                                expect_stderr_and_exit, expect_error_contains
    implicit none
    character(len=:), allocatable :: source

    source = width_source('integer', '-2')
    if (.not. expect_output_matches_gfortran(source, &
                                             'integer_character_width')) stop 1
    source = width_source('integer(8)', '-4294967295_8')
    if (.not. expect_output_matches_gfortran(source, &
                                             'integer8_character_width')) stop 1
    source = &
        'program main'//new_line('a')// &
        '  character(:), allocatable :: r'//new_line('a')// &
        '  r = make(4294967297_8)'//new_line('a')// &
        '  print *, len(r)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  function make(k) result(s)'//new_line('a')// &
        '    integer(8), intent(in) :: k'//new_line('a')// &
        '    character(len=k) :: s'//new_line('a')// &
        '    s = "a"'//new_line('a')// &
        '  end function make'//new_line('a')// &
        'end program main'
    if (.not. expect_stderr_and_exit(source, &
        'Fortran runtime error: Character length exceeds supported maximum '// &
        '2147483647'//new_line('a'), 2, &
        'test/ffc_integer_character_width_limit')) stop 1
    source = &
        'program main'//new_line('a')// &
        '  integer :: n'//new_line('a')// &
        '  character(len=n) :: s'//new_line('a')// &
        '  n = 5'//new_line('a')// &
        '  s = "a"'//new_line('a')// &
        '  print *, s'//new_line('a')// &
        'end program main'
    if (.not. expect_error_contains(source, 'character length', &
        'test/ffc_invalid_main_character_width')) stop 1
    print *, 'PASS: integer character widths match gfortran'


contains

    function width_source(integer_type, negative_length) result(source)
        character(len=*), intent(in) :: integer_type, negative_length
        character(len=:), allocatable :: source

        ! The specification width is independent of the assigned value and is
        ! captured before the dummy is changed. Check padding, truncation,
        ! zero/negative lengths, and storage for an initial substring write.
        source = &
            'program main'//new_line('a')// &
            '  '//integer_type//' :: n'//new_line('a')// &
            '  character(:), allocatable :: r'//new_line('a')// &
            '  n = 5'//new_line('a')// &
            '  r = make(n)'//new_line('a')// &
            '  print *, len(r), "[", r, "]", n'//new_line('a')// &
            '  n = 2'//new_line('a')// &
            '  r = make(n)'//new_line('a')// &
            '  print *, len(r), "[", r, "]", n'//new_line('a')// &
            '  n = 0'//new_line('a')// &
            '  r = make(n)'//new_line('a')// &
            '  print *, len(r), "[", r, "]"'//new_line('a')// &
            '  n = '//negative_length//new_line('a')// &
            '  r = make(n)'//new_line('a')// &
            '  print *, len(r), "[", r, "]"'//new_line('a')// &
            '  n = 3'//new_line('a')// &
            '  r = fill(n)'//new_line('a')// &
            '  print *, len(r), "[", r, "]"'//new_line('a')// &
            '  deallocate(r)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  function make(k) result(s)'//new_line('a')// &
            '    '//integer_type//', intent(inout) :: k'//new_line('a')// &
            '    character(len=k) :: s'//new_line('a')// &
            '    k = 1'//new_line('a')// &
            '    s = "abc"'//new_line('a')// &
            '  end function make'//new_line('a')// &
            '  function fill(k) result(s)'//new_line('a')// &
            '    '//integer_type//', intent(in) :: k'//new_line('a')// &
            '    character(len=k) :: s'//new_line('a')// &
            '    integer :: i'//new_line('a')// &
            '    do i = 1, k'//new_line('a')// &
            '      s(i:i) = "z"'//new_line('a')// &
            '    end do'//new_line('a')// &
            '  end function fill'//new_line('a')// &
            'end program main'
    end function width_source
end program test_session_integer_character_width_compiler
