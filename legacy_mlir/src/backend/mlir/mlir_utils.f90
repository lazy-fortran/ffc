module mlir_utils
    implicit none
    private

    public :: int_to_str, string_to_int

contains

    function int_to_str(n) result(str)
        integer, intent(in) :: n
        character(len=:), allocatable :: str
        character(len=20) :: tmp

        write (tmp, '(I0)') n
        str = trim(tmp)
    end function int_to_str

    function string_to_int(str) result(n)
        character(len=*), intent(in) :: str
        integer :: n

        read (str, *) n
    end function string_to_int

end module mlir_utils
