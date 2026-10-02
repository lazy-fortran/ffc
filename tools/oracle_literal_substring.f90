program o3018
    implicit none
    character(len=3) :: c
    character(len=9) :: pad
    character(len=8) :: over
    print *, 'abcdef'(2:4)
    print *, 'abc'(1:3)
    pad = 'abcdefghi'(3:8)
    print '(a,i0)', pad, 1
    print *, 'abc'(1:1)
    print *, 'x'//'abcdef'(2:3)//'y'
    print *, len('abcdef'(2:4))
    c = 'hello'(2:4)
    print '(a,i0)', c, 2
    pad = 'abcdefghi'(2:9)
    print '(a,i0)', pad, 3
    if ('abcdef'(2:4) == 'bcd') print *, 'EQ_OK'
    print '(a,i0)', "dqtest"(2:3), 4
    over = 'abcdefgh'(1:8)
    print '(a,i0)', over, 5
    print '(a,i0)', 'xyzw'(4:4), 6
end program o3018
