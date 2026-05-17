module backend_constants
    implicit none
    private

    ! Export constants
    public :: BACKEND_FORTRAN, BACKEND_MLIR, BACKEND_LLVM, BACKEND_C

    ! Backend type constants
    integer, parameter :: BACKEND_FORTRAN = 1
    integer, parameter :: BACKEND_MLIR = 2
    integer, parameter :: BACKEND_LLVM = 3
    integer, parameter :: BACKEND_C = 4

end module backend_constants
