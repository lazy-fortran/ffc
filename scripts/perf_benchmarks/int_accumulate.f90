program int_accumulate
    integer :: i
    integer :: acc
    acc = 0
    do i = 1, 200000000
        acc = acc + mod(i, 7)
    end do
    print *, acc
end program int_accumulate
