module session_program_lowering
    use session_program_lowering_impl, only: lower_program_to_liric_exe, &
        lower_program_to_liric_object
    implicit none
    private
    public :: lower_program_to_liric_exe
    public :: lower_program_to_liric_object
end module session_program_lowering
