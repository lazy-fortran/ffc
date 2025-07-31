module string_type
    implicit none
    private

    public :: string_t

    ! Minimal string type to replace fpm_strings dependency
    type :: string_t
        character(len=:), allocatable :: s
    end type string_t

end module string_type