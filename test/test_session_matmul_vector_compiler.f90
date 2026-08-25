program test_session_matmul_vector_compiler
    implicit none
    integer :: a(3), b(3, 2), c(2), d(2, 3), e(2)
    logical :: la(3), lb(3, 2), lc(2)

    a = [1, 2, 3]
    b = reshape([1, 4, 2, 5, 3, 6], [3, 2])
    c = matmul(a, b)
    if (any(c /= [15, 29])) error stop 1

    d = reshape([1, 2, 3, 4, 5, 6], [2, 3])
    e = matmul(d, a)
    if (any(e /= [22, 28])) error stop 2

    la = [.true., .false., .true.]
    lb = reshape([.true., .false., .true., .false., .true., .true.], [3, 2])
    lc = matmul(la, lb)
    if (any(lc .neqv. [.true., .true.])) error stop 3
end program test_session_matmul_vector_compiler
