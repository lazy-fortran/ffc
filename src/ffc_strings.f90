module ffc_strings
    implicit none
    private

    public :: set_empty

contains

    subroutine set_empty(value)
        character(len=:), allocatable, intent(out) :: value

        allocate (character(len=0) :: value)
    end subroutine set_empty

end module ffc_strings
