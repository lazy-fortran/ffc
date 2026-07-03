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
    public :: fmod_variable_t
    public :: fmod_procedure_t
    public :: fmod_generic_t
    public :: module_info_t
    public :: write_fmod
    public :: read_fmod

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

    type :: fmod_variable_t
        character(len=:), allocatable :: name
        character(len=:), allocatable :: kind
    end type fmod_variable_t

    ! A module procedure exported for separate compilation: its Fortran name,
    ! result kind ('integer' for an integer function, 'subroutine' for a
    ! subroutine), argument count, and the space-joined per-argument scalar-kind
    ! tokens (e.g. "integer real"). Only procedures whose arguments are all
    ! supported by-reference scalars are recorded, so a using unit can call them
    ! by reference and link against the module object (#284).
    type :: fmod_procedure_t
        character(len=:), allocatable :: name
        character(len=:), allocatable :: kind
        character(len=:), allocatable :: arg_kinds
        integer :: nargs = 0
    end type fmod_procedure_t

    ! A named generic interface exported for separate compilation: its generic
    ! name and the space-joined list of specific procedure names it resolves to
    ! (e.g. "int8_fnv_1 int16_fnv_1"). Each specific is also carried in the
    ! procedures list, so a using unit imports the specifics and resolves a
    ! use-associated generic call to the matching one by argument kind.
    type :: fmod_generic_t
        character(len=:), allocatable :: name
        character(len=:), allocatable :: specifics
    end type fmod_generic_t

    type :: module_info_t
        character(len=:), allocatable :: name
        type(fmod_parameter_t), allocatable :: parameters(:)
        type(fmod_derived_type_t), allocatable :: derived_types(:)
        type(fmod_variable_t), allocatable :: variables(:)
        type(fmod_procedure_t), allocatable :: procedures(:)
        type(fmod_generic_t), allocatable :: generics(:)
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

        if (allocated(info%variables)) then
            do i = 1, size(info%variables)
                write (unit, '(A)') ''
                write (unit, '(A)') '[[variable]]'
                write (unit, '(A)') 'name = "'//field(info%variables(i)%name)//'"'
                write (unit, '(A)') 'kind = "'//field(info%variables(i)%kind)//'"'
            end do
        end if

        if (allocated(info%procedures)) then
            do i = 1, size(info%procedures)
                write (unit, '(A)') ''
                write (unit, '(A)') '[[procedure]]'
                write (unit, '(A)') 'name = "'//field(info%procedures(i)%name)//'"'
                write (unit, '(A)') 'kind = "'//field(info%procedures(i)%kind)//'"'
                write (unit, '(A,I0)') 'nargs = ', info%procedures(i)%nargs
                write (unit, '(A)') 'arg_kinds = "'// &
                    field(info%procedures(i)%arg_kinds)//'"'
            end do
        end if

        if (allocated(info%generics)) then
            do i = 1, size(info%generics)
                write (unit, '(A)') ''
                write (unit, '(A)') '[[generic]]'
                write (unit, '(A)') 'name = "'//field(info%generics(i)%name)//'"'
                write (unit, '(A)') 'specifics = "'// &
                    field(info%generics(i)%specifics)//'"'
            end do
        end if

        close (unit, iostat=io_stat)
        if (io_stat /= 0) then
            error_msg = 'could not close .fmod artefact: '//trim(path)
            return
        end if
    end subroutine write_fmod

    subroutine read_fmod(path, info, error_msg)
        ! Parse a .fmod written by write_fmod back into a module_info_t. Only
        ! the documented schema is accepted; unknown lines are ignored.
        character(len=*), intent(in) :: path
        type(module_info_t), intent(out) :: info
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: unit, io_stat
        character(len=1024) :: raw
        character(len=:), allocatable :: line, key, val
        character(len=:), allocatable :: section
        type(fmod_parameter_t), allocatable :: params(:)
        type(fmod_derived_type_t), allocatable :: dtypes(:)
        type(fmod_component_t), allocatable :: comps(:)
        type(fmod_variable_t), allocatable :: vars(:)
        type(fmod_procedure_t), allocatable :: procs(:)
        type(fmod_generic_t), allocatable :: gens(:)
        integer :: nparam, ndtype, ncomp, nvar, nproc, ngen, io_read
        character(len=:), allocatable :: cname, ckind

        allocate (character(len=0) :: error_msg)
        info%name = ''
        allocate (params(0))
        allocate (dtypes(0))
        allocate (comps(0))
        allocate (vars(0))
        allocate (procs(0))
        allocate (gens(0))
        nparam = 0
        ndtype = 0
        ncomp = 0
        nvar = 0
        nproc = 0
        ngen = 0
        section = ''

        open (newunit=unit, file=path, status='old', action='read', iostat=io_stat)
        if (io_stat /= 0) then
            error_msg = 'could not open .fmod artefact: '//trim(path)
            return
        end if

        do
            read (unit, '(A)', iostat=io_stat) raw
            if (io_stat /= 0) exit
            line = adjustl(trim(raw))
            if (len_trim(line) == 0) cycle
            if (line == '[module]') then
                section = 'module'
                cycle
            else if (line == '[[parameter]]') then
                call flush_component(comps, ncomp, dtypes, ndtype)
                section = 'parameter'
                nparam = nparam + 1
                call grow_params(params, nparam)
                params(nparam)%name = ''
                params(nparam)%kind = ''
                params(nparam)%value = ''
                cycle
            else if (line == '[[variable]]') then
                call flush_component(comps, ncomp, dtypes, ndtype)
                section = 'variable'
                nvar = nvar + 1
                call grow_vars(vars, nvar)
                vars(nvar)%name = ''
                vars(nvar)%kind = ''
                cycle
            else if (line == '[[procedure]]') then
                call flush_component(comps, ncomp, dtypes, ndtype)
                section = 'procedure'
                nproc = nproc + 1
                call grow_procs(procs, nproc)
                procs(nproc)%name = ''
                procs(nproc)%kind = ''
                procs(nproc)%arg_kinds = ''
                procs(nproc)%nargs = 0
                cycle
            else if (line == '[[generic]]') then
                call flush_component(comps, ncomp, dtypes, ndtype)
                section = 'generic'
                ngen = ngen + 1
                call grow_gens(gens, ngen)
                gens(ngen)%name = ''
                gens(ngen)%specifics = ''
                cycle
            else if (line == '[[derived_type]]') then
                call flush_component(comps, ncomp, dtypes, ndtype)
                section = 'derived_type'
                ndtype = ndtype + 1
                call grow_dtypes(dtypes, ndtype)
                dtypes(ndtype)%name = ''
                allocate (dtypes(ndtype)%components(0))
                deallocate (comps); allocate (comps(0)); ncomp = 0
                cycle
            end if

            if (index(line, '{') == 1) then
                ! A derived-type component row.
                call parse_component_line(line, cname, ckind)
                ncomp = ncomp + 1
                call grow_comps(comps, ncomp)
                comps(ncomp)%name = cname
                comps(ncomp)%kind = ckind
                cycle
            end if

            call split_key_value(line, key, val)
            if (len_trim(key) == 0) cycle
            select case (section)
            case ('module')
                if (key == 'name') info%name = unquote(val)
            case ('parameter')
                if (key == 'name') params(nparam)%name = unquote(val)
                if (key == 'kind') params(nparam)%kind = unquote(val)
                if (key == 'value') params(nparam)%value = unquote(val)
            case ('derived_type')
                if (key == 'name') dtypes(ndtype)%name = unquote(val)
            case ('variable')
                if (key == 'name') vars(nvar)%name = unquote(val)
                if (key == 'kind') vars(nvar)%kind = unquote(val)
            case ('procedure')
                if (key == 'name') procs(nproc)%name = unquote(val)
                if (key == 'kind') procs(nproc)%kind = unquote(val)
                if (key == 'arg_kinds') procs(nproc)%arg_kinds = unquote(val)
                if (key == 'nargs') then
                    read (val, *, iostat=io_read) procs(nproc)%nargs
                    if (io_read /= 0) procs(nproc)%nargs = 0
                end if
            case ('generic')
                if (key == 'name') gens(ngen)%name = unquote(val)
                if (key == 'specifics') gens(ngen)%specifics = unquote(val)
            end select
        end do
        close (unit)
        call flush_component(comps, ncomp, dtypes, ndtype)

        info%parameters = params
        info%derived_types = dtypes
        info%variables = vars(1:nvar)
        info%procedures = procs(1:nproc)
        info%generics = gens(1:ngen)
        if (len_trim(info%name) == 0) error_msg = 'malformed .fmod (no module name): '// &
            trim(path)
    end subroutine read_fmod

    subroutine flush_component(comps, ncomp, dtypes, ndtype)
        ! Attach accumulated component rows to the current derived type.
        type(fmod_component_t), allocatable, intent(inout) :: comps(:)
        integer, intent(inout) :: ncomp
        type(fmod_derived_type_t), allocatable, intent(inout) :: dtypes(:)
        integer, intent(in) :: ndtype

        if (ndtype >= 1 .and. ncomp > 0) then
            dtypes(ndtype)%components = comps(1:ncomp)
        end if
        if (allocated(comps)) deallocate (comps)
        allocate (comps(0))
        ncomp = 0
    end subroutine flush_component

    subroutine parse_component_line(line, name, kind)
        character(len=*), intent(in) :: line
        character(len=:), allocatable, intent(out) :: name
        character(len=:), allocatable, intent(out) :: kind
        integer :: p

        name = ''
        kind = ''
        p = index(line, 'name = "')
        if (p > 0) name = take_quoted(line(p + len('name = "'):))
        p = index(line, 'kind = "')
        if (p > 0) kind = take_quoted(line(p + len('kind = "'):))
    end subroutine parse_component_line

    function take_quoted(text) result(out)
        character(len=*), intent(in) :: text
        character(len=:), allocatable :: out
        integer :: q

        q = index(text, '"')
        if (q > 0) then
            out = text(1:q - 1)
        else
            out = ''
        end if
    end function take_quoted

    subroutine split_key_value(line, key, val)
        character(len=*), intent(in) :: line
        character(len=:), allocatable, intent(out) :: key
        character(len=:), allocatable, intent(out) :: val
        integer :: eq

        key = ''
        val = ''
        eq = index(line, '=')
        if (eq <= 0) return
        key = trim(adjustl(line(1:eq - 1)))
        val = trim(adjustl(line(eq + 1:)))
    end subroutine split_key_value

    function unquote(text) result(out)
        character(len=*), intent(in) :: text
        character(len=:), allocatable :: out
        character(len=:), allocatable :: t

        t = trim(adjustl(text))
        if (len(t) >= 2) then
            if (t(1:1) == '"' .and. t(len(t):len(t)) == '"') then
                out = t(2:len(t) - 1)
                return
            end if
        end if
        out = t
    end function unquote

    subroutine grow_params(arr, n)
        type(fmod_parameter_t), allocatable, intent(inout) :: arr(:)
        integer, intent(in) :: n
        type(fmod_parameter_t), allocatable :: tmp(:)

        if (n <= size(arr)) return
        allocate (tmp(n))
        tmp(1:size(arr)) = arr
        call move_alloc(tmp, arr)
    end subroutine grow_params

    subroutine grow_dtypes(arr, n)
        type(fmod_derived_type_t), allocatable, intent(inout) :: arr(:)
        integer, intent(in) :: n
        type(fmod_derived_type_t), allocatable :: tmp(:)

        if (n <= size(arr)) return
        allocate (tmp(n))
        tmp(1:size(arr)) = arr
        call move_alloc(tmp, arr)
    end subroutine grow_dtypes

    subroutine grow_comps(arr, n)
        type(fmod_component_t), allocatable, intent(inout) :: arr(:)
        integer, intent(in) :: n
        type(fmod_component_t), allocatable :: tmp(:)

        if (n <= size(arr)) return
        allocate (tmp(n))
        tmp(1:size(arr)) = arr
        call move_alloc(tmp, arr)
    end subroutine grow_comps

    subroutine grow_vars(arr, n)
        type(fmod_variable_t), allocatable, intent(inout) :: arr(:)
        integer, intent(in) :: n
        type(fmod_variable_t), allocatable :: tmp(:)

        if (n <= size(arr)) return
        allocate (tmp(n))
        tmp(1:size(arr)) = arr
        call move_alloc(tmp, arr)
    end subroutine grow_vars

    subroutine grow_procs(arr, n)
        type(fmod_procedure_t), allocatable, intent(inout) :: arr(:)
        integer, intent(in) :: n
        type(fmod_procedure_t), allocatable :: tmp(:)

        if (n <= size(arr)) return
        allocate (tmp(n))
        tmp(1:size(arr)) = arr
        call move_alloc(tmp, arr)
    end subroutine grow_procs

    subroutine grow_gens(arr, n)
        type(fmod_generic_t), allocatable, intent(inout) :: arr(:)
        integer, intent(in) :: n
        type(fmod_generic_t), allocatable :: tmp(:)

        if (n <= size(arr)) return
        allocate (tmp(n))
        tmp(1:size(arr)) = arr
        call move_alloc(tmp, arr)
    end subroutine grow_gens

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
