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
    public :: fmod_binding_t
    public :: fmod_derived_type_t
    public :: fmod_variable_t
    public :: fmod_procedure_t
    public :: fmod_generic_t
    public :: module_info_t
    public :: write_fmod
    public :: read_fmod

    character(len=*), parameter, public :: FFC_FMOD_VERSION = '0.1.0'

    ! The .fmod schema this ffc writes and is able to read. Every artefact
    ! carries it as `fmod_schema` in its [module] header. A reader rejects any
    ! other value, and an artefact without the field, so a stale or
    ! newer-than-supported artefact is diagnosed instead of silently misread
    ! (#397).
    integer, parameter, public :: FMOD_SCHEMA_VERSION = 11

    type :: fmod_parameter_t
        character(len=:), allocatable :: name
        character(len=:), allocatable :: kind
        character(len=:), allocatable :: value
    end type fmod_parameter_t

    ! One component of an exported derived type, carrying the layout the
    ! defining unit compiled rather than a description a reader would have to
    ! re-derive: the canonical scalar-kind token, the nested type's name for a
    ! derived component, the declared character length, the fixed array shape,
    ! and the component's own slot span and starting slot. slot_offset is
    ! redundant with the preceding slot_counts by construction, which is what
    ! makes it checkable: a reader rejects an artefact whose offsets do not
    ! follow from its own slot counts (#414).
    type :: fmod_component_t
        character(len=:), allocatable :: name
        character(len=:), allocatable :: kind
        ! Name of the nested derived type of a 'derived' component; empty for
        ! every other kind.
        character(len=:), allocatable :: type_name
        ! Elements the component declares (1 for a scalar), the i32 slots one
        ! element occupies, and the total and starting slots of the component.
        integer :: elem_count = 1
        integer :: slot_width = 1
        integer :: slot_count = 1
        integer :: slot_offset = 0
        ! Declared length of a character component; 0 for every other kind.
        integer :: char_length = 0
        ! First-dimension extent of a rank-2 fixed-size component; 0 otherwise.
        integer :: dim1 = 0
        logical :: is_allocatable = .false.
        logical :: is_pointer = .false.
        logical :: is_alloc_array = .false.
    end type fmod_component_t

    ! One static type-bound binding. The defining unit exports the resolved
    ! method/target pair so a separately compiled user can dispatch the same
    ! direct alias without reparsing the type declaration.
    type :: fmod_binding_t
        character(len=:), allocatable :: method_name
        character(len=:), allocatable :: target_name
        ! Space-joined specific targets for a type-bound generic. The first
        ! target_name is retained for readers of schema 10; newer readers use
        ! this list to resolve the actual argument signature.
        character(len=:), allocatable :: specific_names
        character(len=:), allocatable :: pass_name
        logical :: pass_arg = .true.
    end type fmod_binding_t

    type :: fmod_derived_type_t
        character(len=:), allocatable :: name
        ! Name of the type this one extends; empty when it extends nothing.
        character(len=:), allocatable :: parent_name
        type(fmod_component_t), allocatable :: components(:)
        type(fmod_binding_t), allocatable :: bindings(:)
    end type fmod_derived_type_t

    type :: fmod_variable_t
        character(len=:), allocatable :: name
        character(len=:), allocatable :: kind
        character(len=:), allocatable :: c_name
    end type fmod_variable_t

    ! A module procedure exported for separate compilation: its Fortran name,
    ! result kind ('integer' for an integer function, 'subroutine' for a
    ! subroutine), argument count, and the space-joined per-argument scalar-kind
    ! tokens (e.g. "integer real"). Unsupported procedures are also recorded
    ! with callable=false so a using unit can validate USE ONLY without
    ! pretending that ffc can lower a call to them (#584).
    type :: fmod_procedure_t
        character(len=:), allocatable :: name
        ! External linker symbol for a plain BIND(C) interface procedure;
        ! empty for module-mangled procedures and when it matches name.
        character(len=:), allocatable :: external_name
        character(len=:), allocatable :: kind
        character(len=:), allocatable :: arg_kinds
        ! Space-joined dummy-argument names (e.g. "hi lo"), so a using unit
        ! can associate keyword actuals with this signature (#408).
        character(len=:), allocatable :: arg_names
        ! Space-joined per-dummy INTENT tokens, one per dummy: 'in', 'out',
        ! 'inout', or 'none' when no INTENT was declared (#397).
        character(len=:), allocatable :: arg_intents
        ! Space-joined per-dummy attribute flags, one '0'/'1' per dummy, for the
        ! OPTIONAL and VALUE attributes respectively (#397).
        character(len=:), allocatable :: arg_optionals
        character(len=:), allocatable :: arg_values
        ! A function's result-variable name and its scalar kind token (e.g.
        ! 'integer'); both empty for a subroutine (#397).
        character(len=:), allocatable :: result_name
        character(len=:), allocatable :: result_kind
        ! Space-joined per-dummy rank (0 for a scalar) and, for an
        ! explicit-shape array dummy, its total element count. A generic
        ! resolves an imported specific by these ranks exactly as a same-unit
        ! call resolves one by the declared ranks (#415).
        character(len=:), allocatable :: arg_ranks
        character(len=:), allocatable :: arg_extents
        ! True when the exporter preserved a public procedure whose dummy
        ! contracts are outside the scalar lowering ABI. Such arguments must
        ! be lowered only through explicit opaque-argument paths (#584).
        logical :: opaque = .false.
        ! False means the export is known to exist but its call ABI is outside
        ! the direct-session backend. It remains visible to USE validation.
        logical :: callable = .true.
        ! True for a public plain INTERFACE procedure whose implementation is
        ! an external symbol rather than a module-mangled procedure.
        logical :: external_binding = .false.
        ! True when this module's interface declares the procedure and a
        ! submodule supplies its body. The symbol and call contract are the
        ! same either way; a separately compiled submodule reads the interface
        ! it has to implement from these records (#297).
        logical :: deferred_body = .false.
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

    type :: fmod_use_t
        character(len=:), allocatable :: name
    end type fmod_use_t

    type :: module_info_t
        character(len=:), allocatable :: name
        type(fmod_use_t), allocatable :: uses(:)
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
        write (unit, '(A,I0)') 'fmod_schema = ', FMOD_SCHEMA_VERSION

        if (allocated(info%uses)) then
            do i = 1, size(info%uses)
                write (unit, '(A)') ''
                write (unit, '(A)') '[[use]]'
                write (unit, '(A)') 'name = "'//field(info%uses(i)%name)//'"'
            end do
        end if

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
                write (unit, '(A)') 'parent_name = "'// &
                    field(info%derived_types(i)%parent_name)//'"'
                write (unit, '(A)') 'components = ['
                if (allocated(info%derived_types(i)%components)) then
                    do j = 1, size(info%derived_types(i)%components)
                        write (unit, '(A)') component_line( &
                            info%derived_types(i)%components(j))
                    end do
                end if
                write (unit, '(A)') ']'
                write (unit, '(A)') 'bindings = "'// &
                    binding_list(info%derived_types(i)%bindings)// '"'
            end do
        end if

        if (allocated(info%variables)) then
            do i = 1, size(info%variables)
                write (unit, '(A)') ''
                write (unit, '(A)') '[[variable]]'
                write (unit, '(A)') 'name = "'//field(info%variables(i)%name)//'"'
                write (unit, '(A)') 'kind = "'//field(info%variables(i)%kind)//'"'
                if (allocated(info%variables(i)%c_name)) then
                    if (len_trim(info%variables(i)%c_name) > 0) then
                        write (unit, '(A)') 'c_name = "'// &
                            field(info%variables(i)%c_name)//'"'
                    end if
                end if
            end do
        end if

        if (allocated(info%procedures)) then
            do i = 1, size(info%procedures)
                write (unit, '(A)') ''
                write (unit, '(A)') '[[procedure]]'
                write (unit, '(A)') 'name = "'//field(info%procedures(i)%name)//'"'
                write (unit, '(A)') 'external_name = "'// &
                    field(info%procedures(i)%external_name)//'"'
                write (unit, '(A)') 'kind = "'//field(info%procedures(i)%kind)//'"'
                write (unit, '(A,I0)') 'nargs = ', info%procedures(i)%nargs
                write (unit, '(A)') 'arg_kinds = "'// &
                    field(info%procedures(i)%arg_kinds)//'"'
                write (unit, '(A)') 'arg_names = "'// &
                    field(info%procedures(i)%arg_names)//'"'
                write (unit, '(A)') 'arg_intents = "'// &
                    field(info%procedures(i)%arg_intents)//'"'
                write (unit, '(A)') 'arg_optionals = "'// &
                    field(info%procedures(i)%arg_optionals)//'"'
                write (unit, '(A)') 'arg_values = "'// &
                    field(info%procedures(i)%arg_values)//'"'
                write (unit, '(A)') 'result_name = "'// &
                    field(info%procedures(i)%result_name)//'"'
                write (unit, '(A)') 'result_kind = "'// &
                    field(info%procedures(i)%result_kind)//'"'
                write (unit, '(A)') 'arg_ranks = "'// &
                    field(info%procedures(i)%arg_ranks)//'"'
                write (unit, '(A)') 'arg_extents = "'// &
                    field(info%procedures(i)%arg_extents)//'"'
                write (unit, '(A)') 'opaque = '// &
                    bool_text(info%procedures(i)%opaque)
                write (unit, '(A)') 'callable = '// &
                    bool_text(info%procedures(i)%callable)
                write (unit, '(A)') 'external_binding = '// &
                    bool_text(info%procedures(i)%external_binding)
                write (unit, '(A)') 'deferred_body = '// &
                    bool_text(info%procedures(i)%deferred_body)
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
        type(fmod_use_t), allocatable :: uses(:)
        type(fmod_parameter_t), allocatable :: params(:)
        type(fmod_derived_type_t), allocatable :: dtypes(:)
        type(fmod_component_t), allocatable :: comps(:)
        type(fmod_variable_t), allocatable :: vars(:)
        type(fmod_procedure_t), allocatable :: procs(:)
        type(fmod_generic_t), allocatable :: gens(:)
        integer :: nuse, nparam, ndtype, ncomp, nvar, nproc, ngen, io_read
        integer :: schema

        allocate (character(len=0) :: error_msg)
        info%name = ''
        allocate (uses(0))
        allocate (params(0))
        allocate (dtypes(0))
        allocate (comps(0))
        allocate (vars(0))
        allocate (procs(0))
        allocate (gens(0))
        nuse = 0
        nparam = 0
        ndtype = 0
        ncomp = 0
        nvar = 0
        nproc = 0
        ngen = 0
        schema = 0
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
            else if (line == '[[use]]') then
                call flush_component(comps, ncomp, dtypes, ndtype)
                section = 'use'
                nuse = nuse + 1
                call grow_uses(uses, nuse)
                uses(nuse)%name = ''
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
                procs(nproc)%external_name = ''
                procs(nproc)%kind = ''
                procs(nproc)%arg_kinds = ''
                procs(nproc)%arg_names = ''
                procs(nproc)%arg_intents = ''
                procs(nproc)%arg_optionals = ''
                procs(nproc)%arg_values = ''
                procs(nproc)%result_name = ''
                procs(nproc)%result_kind = ''
                procs(nproc)%arg_ranks = ''
                procs(nproc)%arg_extents = ''
                procs(nproc)%callable = .true.
                procs(nproc)%external_binding = .false.
                procs(nproc)%deferred_body = .false.
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
                allocate (dtypes(ndtype)%bindings(0))
                deallocate (comps); allocate (comps(0)); ncomp = 0
                cycle
            end if

            if (index(line, '{') == 1) then
                ! A derived-type component row.
                ncomp = ncomp + 1
                call grow_comps(comps, ncomp)
                call parse_component_line(line, comps(ncomp))
                cycle
            end if

            call split_key_value(line, key, val)
            if (len_trim(key) == 0) cycle
            select case (section)
            case ('module')
                if (key == 'name') info%name = unquote(val)
                if (key == 'fmod_schema') then
                    read (val, *, iostat=io_read) schema
                    if (io_read /= 0) schema = 0
                end if
            case ('use')
                if (key == 'name') uses(nuse)%name = unquote(val)
            case ('parameter')
                if (key == 'name') params(nparam)%name = unquote(val)
                if (key == 'kind') params(nparam)%kind = unquote(val)
                if (key == 'value') params(nparam)%value = unquote(val)
            case ('derived_type')
                if (key == 'name') dtypes(ndtype)%name = unquote(val)
                if (key == 'parent_name') &
                    dtypes(ndtype)%parent_name = unquote(val)
                if (key == 'bindings') &
                    call parse_binding_list(unquote(val), dtypes(ndtype)%bindings)
            case ('variable')
                if (key == 'name') vars(nvar)%name = unquote(val)
                if (key == 'kind') vars(nvar)%kind = unquote(val)
                if (key == 'c_name') vars(nvar)%c_name = unquote(val)
            case ('procedure')
                if (key == 'name') procs(nproc)%name = unquote(val)
                if (key == 'external_name') &
                    procs(nproc)%external_name = unquote(val)
                if (key == 'kind') procs(nproc)%kind = unquote(val)
                if (key == 'arg_kinds') procs(nproc)%arg_kinds = unquote(val)
                if (key == 'arg_names') procs(nproc)%arg_names = unquote(val)
                if (key == 'arg_intents') &
                    procs(nproc)%arg_intents = unquote(val)
                if (key == 'arg_optionals') &
                    procs(nproc)%arg_optionals = unquote(val)
                if (key == 'arg_values') procs(nproc)%arg_values = unquote(val)
                if (key == 'result_name') &
                    procs(nproc)%result_name = unquote(val)
                if (key == 'result_kind') &
                    procs(nproc)%result_kind = unquote(val)
                if (key == 'arg_ranks') procs(nproc)%arg_ranks = unquote(val)
                if (key == 'arg_extents') &
                    procs(nproc)%arg_extents = unquote(val)
                if (key == 'opaque') &
                    procs(nproc)%opaque = unquote(val) == '1'
                if (key == 'callable') &
                    procs(nproc)%callable = unquote(val) == '1'
                if (key == 'external_binding') &
                    procs(nproc)%external_binding = unquote(val) == '1'
                if (key == 'deferred_body') &
                    procs(nproc)%deferred_body = unquote(val) == '1'
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

        info%uses = uses(1:nuse)
        info%parameters = params
        info%derived_types = dtypes
        info%variables = vars(1:nvar)
        info%procedures = procs(1:nproc)
        info%generics = gens(1:ngen)
        if (len_trim(info%name) == 0) then
            error_msg = 'malformed .fmod (no module name): '//trim(path)
            return
        end if
        ! Reject an artefact this ffc cannot read rather than misinterpret it:
        ! a missing field means it predates the versioned schema, any other
        ! value means it was written by a different ffc (#397).
        if (schema /= FMOD_SCHEMA_VERSION) then
            error_msg = 'unsupported .fmod schema version'//schema_text(schema)// &
                ' in '//trim(path)//' (this ffc reads schema version '// &
                int_text(FMOD_SCHEMA_VERSION)//'): recompile the module'
        end if
    end subroutine read_fmod

    subroutine grow_uses(uses, needed)
        type(fmod_use_t), allocatable, intent(inout) :: uses(:)
        integer, intent(in) :: needed
        type(fmod_use_t), allocatable :: old(:)
        integer :: old_size, new_size

        if (size(uses) >= needed) return
        old_size = size(uses)
        new_size = max(needed, max(1, 2 * old_size))
        call move_alloc(uses, old)
        allocate (uses(new_size))
        if (old_size > 0) uses(1:old_size) = old
    end subroutine grow_uses

    function schema_text(schema) result(text)
        ! The schema version as it appears in a diagnostic; an artefact with no
        ! version field reports as unversioned.
        integer, intent(in) :: schema
        character(len=:), allocatable :: text

        if (schema <= 0) then
            text = ' (unversioned)'
        else
            text = ' '//int_text(schema)
        end if
    end function schema_text

    function int_text(value) result(text)
        integer, intent(in) :: value
        character(len=:), allocatable :: text
        character(len=32) :: buffer

        write (buffer, '(I0)') value
        text = trim(buffer)
    end function int_text

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

    function component_line(comp) result(line)
        ! One component row of a [[derived_type]] table.
        type(fmod_component_t), intent(in) :: comp
        character(len=:), allocatable :: line

        line = '    { name = "'//field(comp%name)//'", kind = "'// &
            field(comp%kind)//'", type_name = "'//field(comp%type_name)// &
            '", elem_count = '//int_text(comp%elem_count)// &
            ', slot_width = '//int_text(comp%slot_width)// &
            ', slot_count = '//int_text(comp%slot_count)// &
            ', slot_offset = '//int_text(comp%slot_offset)// &
            ', char_length = '//int_text(comp%char_length)// &
            ', dim1 = '//int_text(comp%dim1)// &
            ', allocatable = '//bool_text(comp%is_allocatable)// &
            ', pointer = '//bool_text(comp%is_pointer)// &
            ', alloc_array = '//bool_text(comp%is_alloc_array)//' },'
    end function component_line

    function binding_list(bindings) result(text)
        type(fmod_binding_t), allocatable, intent(in) :: bindings(:)
        character(len=:), allocatable :: text
        character(len=:), allocatable :: item
        integer :: i

        text = ''
        if (.not. allocated(bindings)) return
        do i = 1, size(bindings)
            item = field(bindings(i)%method_name)//'=>'// &
                   field(bindings(i)%target_name)//'|'// &
                   field(bindings(i)%pass_name)//'|'// &
                   bool_text(bindings(i)%pass_arg)//'|'// &
                   field(bindings(i)%specific_names)
            if (len_trim(text) > 0) text = text//';'
            text = text//item
        end do
    end function binding_list

    subroutine parse_binding_list(text, bindings)
        character(len=*), intent(in) :: text
        type(fmod_binding_t), allocatable, intent(out) :: bindings(:)
        character(len=:), allocatable :: rest, token, target_part, pass_part, &
            pass_arg_part
        integer :: sep, arrow, bar, count

        allocate (bindings(0))
        rest = trim(text)
        count = 0
        do while (len_trim(rest) > 0)
            sep = index(rest, ';')
            if (sep == 0) then
                token = rest
                rest = ''
            else
                token = rest(1:sep - 1)
                rest = adjustl(rest(sep + 1:))
            end if
            arrow = index(token, '=>')
            if (arrow <= 1) cycle
            count = count + 1
            call grow_bindings(bindings, count)
            bindings(count)%method_name = trim(token(1:arrow - 1))
            target_part = token(arrow + 2:)
            bar = index(target_part, '|')
            if (bar == 0) then
                bindings(count)%target_name = trim(target_part)
                bindings(count)%specific_names = trim(target_part)
                bindings(count)%pass_name = ''
                bindings(count)%pass_arg = .true.
                cycle
            end if
            bindings(count)%target_name = trim(target_part(1:bar - 1))
            pass_part = target_part(bar + 1:)
            bar = index(pass_part, '|')
            if (bar == 0) then
                bindings(count)%pass_name = trim(pass_part)
                bindings(count)%pass_arg = .true.
                bindings(count)%specific_names = bindings(count)%target_name
            else
                bindings(count)%pass_name = trim(pass_part(1:bar - 1))
                pass_arg_part = pass_part(bar + 1:)
                bar = index(pass_arg_part, '|')
                if (bar == 0) then
                    bindings(count)%pass_arg = trim(pass_arg_part) /= '0'
                    bindings(count)%specific_names = bindings(count)%target_name
                else
                    bindings(count)%pass_arg = trim(pass_arg_part(1:bar - 1)) /= '0'
                    bindings(count)%specific_names = trim(pass_arg_part(bar + 1:))
                end if
            end if
            if (.not. allocated(bindings(count)%specific_names)) &
                bindings(count)%specific_names = bindings(count)%target_name
        end do
    end subroutine parse_binding_list

    subroutine parse_component_line(line, comp)
        ! Parse one component row back into its record.
        character(len=*), intent(in) :: line
        type(fmod_component_t), intent(out) :: comp

        comp%name = quoted_field(line, 'name')
        comp%kind = quoted_field(line, 'kind')
        comp%type_name = quoted_field(line, 'type_name')
        comp%elem_count = integer_field(line, 'elem_count', 1)
        comp%slot_width = integer_field(line, 'slot_width', 1)
        comp%slot_count = integer_field(line, 'slot_count', 1)
        comp%slot_offset = integer_field(line, 'slot_offset', 0)
        comp%char_length = integer_field(line, 'char_length', 0)
        comp%dim1 = integer_field(line, 'dim1', 0)
        comp%is_allocatable = integer_field(line, 'allocatable', 0) /= 0
        comp%is_pointer = integer_field(line, 'pointer', 0) /= 0
        comp%is_alloc_array = integer_field(line, 'alloc_array', 0) /= 0
    end subroutine parse_component_line

    function quoted_field(line, key) result(out)
        ! The quoted value of `key = "..."` in a component row, or empty.
        character(len=*), intent(in) :: line
        character(len=*), intent(in) :: key
        character(len=:), allocatable :: out
        integer :: p

        out = ''
        p = index(line, key//' = "')
        if (p <= 0) return
        out = take_quoted(line(p + len(key) + 4:))
    end function quoted_field

    integer function integer_field(line, key, default_value) result(value)
        ! The integer value of `key = N` in a component row, or default_value
        ! when the key is absent or unreadable.
        character(len=*), intent(in) :: line
        character(len=*), intent(in) :: key
        integer, intent(in) :: default_value
        integer :: p, io_stat

        value = default_value
        p = index(line, key//' = ')
        if (p <= 0) return
        read (line(p + len(key) + 3:), *, iostat=io_stat) value
        if (io_stat /= 0) value = default_value
    end function integer_field

    function bool_text(flag) result(text)
        logical, intent(in) :: flag
        character(len=:), allocatable :: text

        if (flag) then
            text = '1'
        else
            text = '0'
        end if
    end function bool_text


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

    subroutine grow_bindings(arr, n)
        type(fmod_binding_t), allocatable, intent(inout) :: arr(:)
        integer, intent(in) :: n
        type(fmod_binding_t), allocatable :: tmp(:)

        if (n <= size(arr)) return
        allocate (tmp(n))
        tmp(1:size(arr)) = arr
        call move_alloc(tmp, arr)
    end subroutine grow_bindings

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
