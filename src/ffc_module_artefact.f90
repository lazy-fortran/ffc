module ffc_module_artefact
    ! Serialises a module's exported interface to a sibling `.fmod` artefact so
    ! a later compilation unit can resolve `use <module>` without reparsing the
    ! source. The format is a minimal, line-oriented TOML subset: one [module]
    ! header, then [[parameter]] and [[derived_type]] tables. It carries no
    ! source locations, comments, or prose.
    implicit none
    private

    public :: fmod_parameter_t
    public :: fmod_component_t
    public :: fmod_derived_type_t
    public :: module_info_t
    public :: write_fmod

    character(len=*), parameter, public :: FFC_FMOD_VERSION = '0.1.0'

    type :: fmod_parameter_t
        character(len=:), allocatable :: name
        character(len=:), allocatable :: kind
        character(len=:), allocatable :: value
    end type fmod_parameter_t

    type :: fmod_component_t
        character(len=:), allocatable :: name
        character(len=:), allocatable :: kind
    end type fmod_component_t

    type :: fmod_derived_type_t
        character(len=:), allocatable :: name
        type(fmod_component_t), allocatable :: components(:)
    end type fmod_derived_type_t

    type :: module_info_t
        character(len=:), allocatable :: name
        type(fmod_parameter_t), allocatable :: parameters(:)
        type(fmod_derived_type_t), allocatable :: derived_types(:)
    end type module_info_t

contains

    subroutine write_fmod(path, info, error_msg)
        character(len=*), intent(in) :: path
        type(module_info_t), intent(in) :: info
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: unit, io_stat, i, j

        allocate (character(len=0) :: error_msg)
        open (newunit=unit, file=path, status='replace', action='write', &
              iostat=io_stat)
        if (io_stat /= 0) then
            error_msg = 'could not open .fmod artefact for writing: '//trim(path)
            return
        end if

        write (unit, '(A)') '[module]'
        write (unit, '(A)') 'name = "'//mod_name(info)//'"'
        write (unit, '(A)') 'ffc_version = "'//FFC_FMOD_VERSION//'"'

        if (allocated(info%parameters)) then
            do i = 1, size(info%parameters)
                write (unit, '(A)') ''
                write (unit, '(A)') '[[parameter]]'
                write (unit, '(A)') 'name = "'//field(info%parameters(i)%name)//'"'
                write (unit, '(A)') 'kind = "'//field(info%parameters(i)%kind)//'"'
                write (unit, '(A)') 'value = '//field(info%parameters(i)%value)
            end do
        end if

        if (allocated(info%derived_types)) then
            do i = 1, size(info%derived_types)
                write (unit, '(A)') ''
                write (unit, '(A)') '[[derived_type]]'
                write (unit, '(A)') 'name = "'// &
                    field(info%derived_types(i)%name)//'"'
                write (unit, '(A)') 'components = ['
                if (allocated(info%derived_types(i)%components)) then
                    do j = 1, size(info%derived_types(i)%components)
                        write (unit, '(A)') '    { name = "'// &
                            field(info%derived_types(i)%components(j)%name)// &
                            '", kind = "'// &
                            field(info%derived_types(i)%components(j)%kind)//'" },'
                    end do
                end if
                write (unit, '(A)') ']'
            end do
        end if

        close (unit, iostat=io_stat)
        if (io_stat /= 0) then
            error_msg = 'could not close .fmod artefact: '//trim(path)
            return
        end if
    end subroutine write_fmod

    pure function mod_name(info) result(name)
        type(module_info_t), intent(in) :: info
        character(len=:), allocatable :: name

        if (allocated(info%name)) then
            name = info%name
        else
            name = ''
        end if
    end function mod_name

    pure function field(text) result(out)
        character(len=:), allocatable, intent(in) :: text
        character(len=:), allocatable :: out

        if (allocated(text)) then
            out = text
        else
            out = ''
        end if
    end function field

end module ffc_module_artefact
