program test_session_character_locate_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none
    character(len=*), parameter :: source = &
        'program main'//new_line('a')// &
        '  character(len=3) :: a(4)'//new_line('a')// &
        '  logical :: mask(4)'//new_line('a')// &
        "  a(1) = 'bbb'; a(2) = 'aaa'; a(3) = 'bbb'; a(4) = 'ccc'"// &
        new_line('a')// &
        '  mask = [.false., .true., .false., .false.]'//new_line('a')// &
        '  if (maxloc(a, dim=1) /= 4) error stop 1'//new_line('a')// &
        '  if (minloc(a, dim=1) /= 2) error stop 2'//new_line('a')// &
        '  if (maxloc(a, dim=1, mask=mask) /= 2) error stop 3'//new_line('a')// &
        '  if (minloc(a, dim=1, mask=mask) /= 2) error stop 4'//new_line('a')// &
        'end program main'

    print *, '=== direct session character MINLOC/MAXLOC test ==='
    if (.not. expect_exit_status(source, 0, &
            '/tmp/ffc_session_character_locate_test')) stop 1
    print *, 'PASS: character MINLOC/MAXLOC use lexical ordering and masks'
end program test_session_character_locate_compiler
