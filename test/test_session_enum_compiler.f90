program test_session_enum_compiler
    use ffc_test_support, only: expect_output
    implicit none

    print *, '=== direct session enum compiler test ==='

    ! Explicit initializers bind as integer named constants.
    if (.not. expect_output( &
         'program main'//new_line('a')// &
         '    enum, bind(c)'//new_line('a')// &
         '        enumerator :: RED = 1'//new_line('a')// &
         '        enumerator :: GREEN = 2'//new_line('a')// &
         '    end enum'//new_line('a')// &
         '    integer :: color'//new_line('a')// &
         '    color = RED'//new_line('a')// &
         '    print *, color, GREEN'//new_line('a')// &
         'end program main', &
         '           1           2'//new_line('a'), &
         '/tmp/ffc_session_enum_explicit_test')) stop 1
    print *, 'PASS: explicit enumerator values bind as constants'

    ! Implicit values start at 0 and increment; multiple names per line.
    if (.not. expect_output( &
         'program main'//new_line('a')// &
         '    enum, bind(c)'//new_line('a')// &
         '        enumerator :: a, b, c'//new_line('a')// &
         '    end enum'//new_line('a')// &
         '    print *, a, b, c'//new_line('a')// &
         'end program main', &
         '           0           1           2'//new_line('a'), &
         '/tmp/ffc_session_enum_implicit_test')) stop 1
    print *, 'PASS: implicit enumerator values count from zero'

    ! Mixed explicit and implicit values resume from the last explicit value.
    if (.not. expect_output( &
         'program main'//new_line('a')// &
         '    enum, bind(c)'//new_line('a')// &
         '        enumerator :: x = 10, y, z = 20, w'//new_line('a')// &
         '    end enum'//new_line('a')// &
         '    print *, x, y, z, w'//new_line('a')// &
         'end program main', &
         '          10          11          20          21'//new_line('a'), &
         '/tmp/ffc_session_enum_mixed_test')) stop 1
    print *, 'PASS: mixed enumerator values resume from explicit value'

    ! Enumerators work as select case selectors and in expressions.
    if (.not. expect_output( &
         'program main'//new_line('a')// &
         '    enum, bind(c)'//new_line('a')// &
         '        enumerator :: mon = 1, tue, wed'//new_line('a')// &
         '    end enum'//new_line('a')// &
         '    integer :: d'//new_line('a')// &
         '    d = tue'//new_line('a')// &
         '    select case (d)'//new_line('a')// &
         '    case (tue)'//new_line('a')// &
         '        print *, wed + mon'//new_line('a')// &
         '    case default'//new_line('a')// &
         '        print *, -1'//new_line('a')// &
         '    end select'//new_line('a')// &
         'end program main', &
         '           4'//new_line('a'), &
         '/tmp/ffc_session_enum_select_test')) stop 1
    print *, 'PASS: enumerators drive select case and expressions'

    print *, 'PASS: enum/enumerator lower through direct LIRIC session'
end program test_session_enum_compiler
