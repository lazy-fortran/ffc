program nested_loop
    integer :: i, j
    integer :: acc
    acc = 0
    do i = 1, 20000
        do j = 1, 20000
            acc = acc + mod(i * j, 13)
        end do
    end do
    print *, acc
end program nested_loop
