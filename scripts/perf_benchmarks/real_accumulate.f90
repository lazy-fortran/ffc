program real_accumulate
    integer :: i
    real(8) :: acc
    acc = 0.0d0
    do i = 1, 100000000
        acc = acc + 1.0d0 / real(i, 8)
    end do
    print *, acc
end program real_accumulate
