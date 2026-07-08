    ! COMMON blocks and BLOCK DATA (#1578, #1900, #2809). FortFront parses a
    ! program COMMON statement into a common_block_node (block_names,
    ! member_names, member_block) and a BLOCK DATA unit into a block_data_node
    ! whose statement_indices point at its body nodes (a common_block_node for
    ! the layout, data_statement_nodes for the initialisers). This pass reads
    ! both from those structured nodes.
    !
    ! Each named or blank COMMON block gets one LIRIC global per element
    ! offset, keyed by block name and position-within-block (not variable
    ! spelling): different program units may list different names at the
    ! same block, and F2018 8.10.3 associates them by position, so a program
    ! and a subroutine (or BLOCK DATA unit) sharing a block name alias the
    ! same storage even when their member names differ. Each unit's own
    ! COMMON statement is re-resolved against this canonical layout when the
    ! body walk reaches it, so binding never depends on which other unit's
    ! node happens to share a member name. The BLOCK DATA unit's DATA values
    ! are folded into the matching offset's static initialiser, the same
    ! once-at-load model SAVE locals and module variables use.
    !
    ! Scope: scalar integer(4), real(4), real(8), logical, complex(4), and
    ! complex(8) slots whose initialisers are plain literals, plus 1D arrays
    ! of the numeric/logical kinds. Character members and EQUIVALENCE-linked
    ! members leave the symbol a plain local and surface the usual
    ! diagnostics, keeping the unit honest rather than miscompiling.

    subroutine emit_common_block_globals(arena, context, error_msg)
        ! Pre-pass: build the COMMON layout from program declarations and the
        ! program COMMON comment, fold BLOCK DATA initialisers, and emit one
        ! initialised global per slot. Runs before any body is lowered so the
        ! globals carry data-section initialisers (a global created mid-body is
        ! address-patched and skips its initialiser).
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(common_slot_t) :: slots(COMMON_MAX_SLOTS)
        type(common_slot_t) :: canon(COMMON_MAX_SLOTS)
        integer :: slot_count, canon_count, i

        call set_empty(error_msg)
        slot_count = 0
        call collect_common_slots(arena, context%root_index, slots, slot_count, &
                                  error_msg)
        if (len_trim(error_msg) > 0) return
        if (slot_count == 0) return
        call apply_block_data_initializers(arena, slots, slot_count, error_msg)
        if (len_trim(error_msg) > 0) return
        call canonicalize_common_slots(slots, slot_count, canon, canon_count)
        do i = 1, canon_count
            call emit_one_common_global(canon(i), context, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
    end subroutine emit_common_block_globals

    subroutine canonicalize_common_slots(slots, slot_count, canon, canon_count)
        ! Collapse the raw per-unit member list down to one entry per
        ! distinct (block_name, pos_in_block): every unit sharing a block
        ! name and offset shares one global, so only the first occurrence's
        ! kind/array shape is emitted, with any DATA initialiser found on a
        ! later duplicate merged in (a BLOCK DATA unit may spell a member
        ! differently than the first unit collected).
        type(common_slot_t), intent(in) :: slots(:)
        integer, intent(in) :: slot_count
        type(common_slot_t), intent(out) :: canon(:)
        integer, intent(out) :: canon_count
        integer :: i, j
        logical :: found

        canon_count = 0
        do i = 1, slot_count
            found = .false.
            do j = 1, canon_count
                if (canon(j)%block_name == slots(i)%block_name .and. &
                    canon(j)%pos_in_block == slots(i)%pos_in_block) then
                    call merge_common_slot_init(canon(j), slots(i))
                    found = .true.
                    exit
                end if
            end do
            if (.not. found) then
                canon_count = canon_count + 1
                canon(canon_count) = slots(i)
            end if
        end do
    end subroutine canonicalize_common_slots

    subroutine merge_common_slot_init(dest, src)
        ! Fold src's DATA initialiser into dest when dest has none of its own,
        ! so whichever unit's member name a BLOCK DATA statement addressed
        ! still reaches the canonical global for that block offset.
        type(common_slot_t), intent(inout) :: dest
        type(common_slot_t), intent(in) :: src
        integer :: idx

        if (src%has_init .and. .not. dest%has_init) then
            dest%has_init = .true.
            dest%init_text = src%init_text
        end if
        if (.not. allocated(src%array_init_indices)) return
        if (.not. allocated(dest%array_init_indices)) then
            allocate (dest%array_init_indices(max(dest%array_size, 1)))
            allocate (character(len=256) :: &
                      dest%array_init_values(max(dest%array_size, 1)))
            dest%array_init_indices = 0
            dest%array_init_values = ''
        end if
        do idx = 1, min(size(src%array_init_indices), &
                        size(dest%array_init_indices))
            if (src%array_init_indices(idx) /= 0 .and. &
                dest%array_init_indices(idx) == 0) then
                dest%array_init_indices(idx) = src%array_init_indices(idx)
                dest%array_init_values(idx) = src%array_init_values(idx)
            end if
        end do
    end subroutine merge_common_slot_init

    subroutine collect_common_slots(arena, root_index, slots, slot_count, &
                                    error_msg)
        ! Walk the arena for program-level common_block_nodes and resolve each
        ! member's kind from a matching program declaration. BLOCK DATA units
        ! carry their own common_block_node inside statement_indices; those
        ! repeat the same layout, so they are skipped here to keep one slot per
        ! variable.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: root_index
        type(common_slot_t), intent(inout) :: slots(:)
        integer, intent(inout) :: slot_count
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: n

        call set_empty(error_msg)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            if (is_block_data_member(arena, n)) cycle
            select type (node => arena%entries(n)%node)
            type is (common_block_node)
                call add_common_node_slots(arena, root_index, n, node, slots, &
                                           slot_count, error_msg)
                if (len_trim(error_msg) > 0) return
            end select
        end do
    end subroutine collect_common_slots

    logical function is_block_data_member(arena, idx)
        ! A common_block_node that belongs to a BLOCK DATA unit (it appears in
        ! some block_data_node's statement_indices) is its own copy of the
        ! program layout; skip it so the program node defines the slots once.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: idx
        integer :: n, s

        is_block_data_member = .false.
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (bd => arena%entries(n)%node)
            type is (block_data_node)
                if (.not. allocated(bd%statement_indices)) cycle
                do s = 1, size(bd%statement_indices)
                    if (bd%statement_indices(s) == idx) then
                        is_block_data_member = .true.
                        return
                    end if
                end do
            end select
        end do
    end function is_block_data_member

    subroutine add_common_node_slots(arena, root_index, node_index, node, &
                                     slots, slot_count, error_msg)
        ! Append one slot per member of a common_block_node. The owning block
        ! name comes from block_names(member_block(i)); a blank name uses the
        ! sentinel "__blnk__". The member kind comes from the declaration in
        ! the program unit that owns this COMMON statement when that scope
        ! can be resolved (root_index/node_index), so a name collision with
        ! an unrelated local in another unit cannot pick the wrong kind.
        ! pos_in_block is this node's own running element offset within its
        ! block (scalars consume one slot, arrays consume array_size),
        ! independent of any other unit's member names, so two units sharing
        ! a block name alias by position rather than by spelling (#1578,
        ! F2018 8.10.3).
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: root_index, node_index
        type(common_block_node), intent(in) :: node
        type(common_slot_t), intent(inout) :: slots(:)
        integer, intent(inout) :: slot_count
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: block_name, var_name
        character(len=64) :: block_names_seen(COMMON_MAX_SLOTS)
        integer :: block_offsets(COMMON_MAX_SLOTS)
        integer :: n_blocks_seen, b
        integer :: i, vk, arr_size
        logical :: arr_flag

        call set_empty(error_msg)
        if (.not. allocated(node%member_names)) return
        n_blocks_seen = 0
        do i = 1, size(node%member_names)
            if (slot_count >= size(slots)) then
                error_msg = 'too many COMMON variables for direct lowering'
                return
            end if
            var_name = trim(node%member_names(i)%s)
            block_name = common_member_block_name(node, i)
            call program_declaration_kind_for_common(arena, root_index, &
                node_index, var_name, vk, arr_flag, arr_size, error_msg)
            if (len_trim(error_msg) > 0) return
            call find_or_add_block_offset(block_names_seen, block_offsets, &
                                         n_blocks_seen, block_name, b, error_msg)
            if (len_trim(error_msg) > 0) return
            slot_count = slot_count + 1
            slots(slot_count)%block_name = block_name
            slots(slot_count)%var_name = var_name
            slots(slot_count)%value_kind = vk
            slots(slot_count)%is_array = arr_flag
            slots(slot_count)%array_size = arr_size
            slots(slot_count)%pos_in_block = block_offsets(b)
            slots(slot_count)%has_init = .false.
            if (arr_flag) then
                block_offsets(b) = block_offsets(b) + max(arr_size, 1)
            else
                block_offsets(b) = block_offsets(b) + 1
            end if
        end do
    end subroutine add_common_node_slots

    subroutine find_or_add_block_offset(block_names_seen, block_offsets, &
                                        n_blocks_seen, block_name, b, error_msg)
        ! Look up block_name's running-offset slot within a single node's own
        ! scratch tables, appending a fresh zero-offset entry on first sight.
        character(len=*), intent(inout) :: block_names_seen(:)
        integer, intent(inout) :: block_offsets(:)
        integer, intent(inout) :: n_blocks_seen
        character(len=*), intent(in) :: block_name
        integer, intent(out) :: b
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: j

        call set_empty(error_msg)
        do j = 1, n_blocks_seen
            if (trim(block_names_seen(j)) == block_name) then
                b = j
                return
            end if
        end do
        if (n_blocks_seen >= size(block_names_seen)) then
            error_msg = 'too many COMMON blocks for direct lowering'
            return
        end if
        n_blocks_seen = n_blocks_seen + 1
        b = n_blocks_seen
        block_names_seen(b) = block_name
        block_offsets(b) = 0
    end subroutine find_or_add_block_offset

    function common_member_block_name(node, member) result(block_name)
        ! Resolve member i's owning block name; blank common uses "__blnk__".
        type(common_block_node), intent(in) :: node
        integer, intent(in) :: member
        character(len=:), allocatable :: block_name
        integer :: b

        block_name = '__blnk__'
        if (.not. allocated(node%member_block)) return
        if (.not. allocated(node%block_names)) return
        b = node%member_block(member)
        if (b < 1 .or. b > size(node%block_names)) return
        if (len_trim(node%block_names(b)%s) > 0) &
            block_name = trim(node%block_names(b)%s)
    end function common_member_block_name

    subroutine program_declaration_kind_for_common(arena, root_index, &
                                                   node_index, var_name, &
                                                   value_kind, is_array, &
                                                   array_size, error_msg)
        ! Resolve a COMMON member's declared kind, scoped to the program unit
        ! that owns the COMMON statement at node_index when that scope can be
        ! determined: two units may each declare their own unrelated local
        ! under the same name (e.g. both scalar and array "dummy"), and an
        ! arena-wide search would silently pick whichever declaration happens
        ! to come first. Falls back to the arena-wide search when the owning
        ! unit cannot be resolved or does not declare the name locally (a
        ! host-associated or otherwise indirect reference).
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: root_index, node_index
        character(len=*), intent(in) :: var_name
        integer, intent(out) :: value_kind
        logical, intent(out) :: is_array
        integer, intent(out) :: array_size
        character(len=:), allocatable, intent(out) :: error_msg
        integer, allocatable :: scope_indices(:)
        logical :: scope_found

        call find_owning_unit_body_indices(arena, root_index, node_index, &
                                           scope_indices, scope_found)
        if (scope_found) then
            call program_declaration_kind_in_scope(arena, scope_indices, &
                var_name, value_kind, is_array, array_size, error_msg)
            if (len_trim(error_msg) == 0) return
        end if
        call program_declaration_kind_full(arena, var_name, value_kind, &
                                          is_array, array_size, error_msg)
    end subroutine program_declaration_kind_for_common

    subroutine program_declaration_kind_in_scope(arena, scope_indices, &
                                                 var_name, value_kind, &
                                                 is_array, array_size, &
                                                 error_msg)
        ! Same resolution as program_declaration_kind_full, restricted to a
        ! single unit's own statement list.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: scope_indices(:)
        character(len=*), intent(in) :: var_name
        integer, intent(out) :: value_kind
        logical, intent(out) :: is_array
        integer, intent(out) :: array_size
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: k
        logical :: found

        value_kind = VALUE_I32
        is_array = .false.
        array_size = 0
        found = .false.
        call set_empty(error_msg)
        do k = 1, size(scope_indices)
            if (.not. node_exists(arena, scope_indices(k))) cycle
            select type (decl => arena%entries(scope_indices(k))%node)
            type is (declaration_node)
                if (decl%is_pointer .or. decl%is_target .or. &
                    decl%is_allocatable .or. decl%is_parameter) cycle
                if (declaration_names_var(decl, var_name)) then
                    call declaration_value_kind(decl, value_kind, error_msg)
                    if (len_trim(error_msg) > 0) return
                    is_array = decl%is_array
                    if (is_array) then
                        call extract_array_size(arena, decl, array_size, error_msg)
                        if (len_trim(error_msg) > 0) return
                    end if
                    found = .true.
                    exit
                end if
            end select
        end do
        if (.not. found) then
            error_msg = 'COMMON variable not declared in scope: '//var_name
            return
        end if
        call validate_common_member_kind(var_name, value_kind, is_array, &
                                         error_msg)
    end subroutine program_declaration_kind_in_scope

    subroutine validate_common_member_kind(var_name, value_kind, is_array, &
                                           error_msg)
        character(len=*), intent(in) :: var_name
        integer, intent(in) :: value_kind
        logical, intent(in) :: is_array
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        select case (value_kind)
        case (VALUE_I32, VALUE_F32, VALUE_F64, VALUE_LOGICAL)
        case (VALUE_C4, VALUE_C8)
            if (is_array) then
                error_msg = 'COMMON array has an unsupported kind: '//var_name
            end if
        case default
            error_msg = 'COMMON variable has an unsupported kind: '//var_name
        end select
    end subroutine validate_common_member_kind

    subroutine find_owning_unit_body_indices(arena, root_index, target_index, &
                                             body_indices, found)
        ! Locate the top-level (or one-level-CONTAINS-nested) program unit
        ! whose own statement list holds target_index, and return that unit's
        ! statement indices as the declaration search scope.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: root_index, target_index
        integer, allocatable, intent(out) :: body_indices(:)
        logical, intent(out) :: found
        integer, allocatable :: top_units(:), inner_body(:)
        character(len=:), allocatable :: node_type, inner_type, ff_error
        integer :: i, j

        found = .false.
        call get_unit_body_indices(arena, root_index, top_units, ff_error)
        if (.not. allocated(top_units)) return
        do i = 1, size(top_units)
            node_type = get_node_type_at(arena, top_units(i))
            call unit_body_indices_for(arena, top_units(i), node_type, &
                                       body_indices, ff_error)
            if (.not. allocated(body_indices)) cycle
            if (index_in_int_list(body_indices, target_index)) then
                found = .true.
                return
            end if
            do j = 1, size(body_indices)
                inner_type = get_node_type_at(arena, body_indices(j))
                if (inner_type == 'subroutine_def_node' .or. &
                    inner_type == 'subroutine_def' .or. &
                    inner_type == 'function_def_node' .or. &
                    inner_type == 'function_def') then
                    call unit_body_indices_for(arena, body_indices(j), &
                        inner_type, inner_body, ff_error)
                    if (allocated(inner_body)) then
                        if (index_in_int_list(inner_body, target_index)) then
                            body_indices = inner_body
                            found = .true.
                            return
                        end if
                    end if
                end if
            end do
        end do
    end subroutine find_owning_unit_body_indices

    subroutine unit_body_indices_for(arena, unit_index, node_type, &
                                     body_indices, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: unit_index
        character(len=*), intent(in) :: node_type
        integer, allocatable, intent(out) :: body_indices(:)
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name, result_name
        integer, allocatable :: param_indices(:)

        if (allocated(body_indices)) deallocate (body_indices)
        call set_empty(error_msg)
        if (node_type == 'program_node' .or. node_type == 'program') then
            call get_program_body_info(arena, unit_index, name, body_indices, &
                                       error_msg)
        else if (node_type == 'subroutine_def_node' .or. &
                 node_type == 'subroutine_def') then
            call get_subroutine_body_info(arena, unit_index, name, &
                                          param_indices, body_indices, error_msg)
        else if (node_type == 'function_def_node' .or. &
                 node_type == 'function_def') then
            call get_function_body_info(arena, unit_index, name, &
                                        param_indices, body_indices, &
                                        result_name, error_msg)
        else
            allocate (body_indices(0))
        end if
    end subroutine unit_body_indices_for

    logical function index_in_int_list(list, target) result(found)
        integer, intent(in) :: list(:)
        integer, intent(in) :: target
        integer :: k

        found = .false.
        do k = 1, size(list)
            if (list(k) == target) then
                found = .true.
                return
            end if
        end do
    end function index_in_int_list

    subroutine program_declaration_kind(arena, var_name, value_kind, error_msg)
        ! Legacy wrapper: resolve a scalar COMMON variable's kind from a program
        ! declaration. Used by equivalence code.
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: var_name
        integer, intent(out) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg
        logical :: dummy_is_array
        integer :: dummy_array_size

        call program_declaration_kind_full(arena, var_name, value_kind, &
                                          dummy_is_array, dummy_array_size, &
                                          error_msg)
    end subroutine program_declaration_kind

    subroutine program_declaration_kind_full(arena, var_name, value_kind, &
                                           is_array, array_size, error_msg)
        ! Resolve a COMMON variable's kind from a program declaration. Supports
        ! both scalars and 1D arrays.
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: var_name
        integer, intent(out) :: value_kind
        logical, intent(out) :: is_array
        integer, intent(out) :: array_size
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: n
        logical :: found

        value_kind = VALUE_I32
        is_array = .false.
        array_size = 0
        found = .false.
        call set_empty(error_msg)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (decl => arena%entries(n)%node)
            type is (declaration_node)
                if (decl%is_pointer .or. decl%is_target .or. &
                    decl%is_allocatable .or. decl%is_parameter) cycle
                if (declaration_names_var(decl, var_name)) then
                    call declaration_value_kind(decl, value_kind, error_msg)
                    if (len_trim(error_msg) > 0) return
                    is_array = decl%is_array
                    if (is_array) then
                        call extract_array_size(arena, decl, array_size, error_msg)
                        if (len_trim(error_msg) > 0) return
                    end if
                    found = .true.
                    exit
                end if
            end select
        end do
        if (.not. found) then
            error_msg = 'COMMON variable not declared: '//var_name
            return
        end if
        call validate_common_member_kind(var_name, value_kind, is_array, &
                                         error_msg)
    end subroutine program_declaration_kind_full

    logical function declaration_names_var(decl, var_name)
        type(declaration_node), intent(in) :: decl
        character(len=*), intent(in) :: var_name
        integer :: i
        declaration_names_var = .false.
        if (decl%is_multi_declaration .and. allocated(decl%var_names)) then
            do i = 1, size(decl%var_names)
                if (trim(decl%var_names(i)) == var_name) then
                    declaration_names_var = .true.
                    return
                end if
            end do
        else if (allocated(decl%var_name)) then
            declaration_names_var = trim(decl%var_name) == var_name
        end if
    end function declaration_names_var

    subroutine extract_array_size(arena, decl, array_size, error_msg)
        ! Extract the total size of a 1D array from its dimension declaration.
        type(ast_arena_t), intent(in) :: arena
        type(declaration_node), intent(in) :: decl
        integer, intent(out) :: array_size
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: dim_idx, dim_size

        array_size = 1
        call set_empty(error_msg)
        if (.not. allocated(decl%dimension_indices)) return
        ! A multidimensional or runtime-extent COMMON array member is laid out
        ! as a single scalar slot - ffc does not model its multi-element storage
        ! - matching the behaviour from when fortfront dropped the array-spec.
        ! Decline to a scalar rather than failing the whole unit (#2870 made
        ! these shapes visible again).
        if (size(decl%dimension_indices) /= 1) return
        dim_idx = decl%dimension_indices(1)
        call extract_dimension_size(arena, dim_idx, dim_size, error_msg)
        if (len_trim(error_msg) > 0) then
            array_size = 1
            call set_empty(error_msg)
            return
        end if
        array_size = dim_size
    end subroutine extract_array_size

    subroutine extract_dimension_size(arena, dim_idx, dim_size, error_msg)
        ! Extract the size of a dimension from the arena. Simple constant
        ! dimensions only; expressions are unsupported.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: dim_idx
        integer, intent(out) :: dim_size
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: io_stat

        dim_size = 0
        call set_empty(error_msg)
        if (.not. node_exists(arena, dim_idx)) then
            error_msg = 'dimension index not found in arena'
            return
        end if
        select type (dim => arena%entries(dim_idx)%node)
        type is (literal_node)
            read (dim%value, *, iostat=io_stat) dim_size
            if (io_stat /= 0) then
                error_msg = 'malformed dimension literal: '//trim(dim%value)
                return
            end if
            if (dim_size < 0) dim_size = 0
        class default
            error_msg = 'unsupported dimension expression (must be constant)'
        end select
    end subroutine extract_dimension_size

    subroutine apply_block_data_initializers(arena, slots, slot_count, error_msg)
        ! Fold each BLOCK DATA unit's DATA statements into the matching slots.
        type(ast_arena_t), intent(in) :: arena
        type(common_slot_t), intent(inout) :: slots(:)
        integer, intent(in) :: slot_count
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: n, s

        call set_empty(error_msg)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (bd => arena%entries(n)%node)
            type is (block_data_node)
                if (.not. allocated(bd%statement_indices)) cycle
                do s = 1, size(bd%statement_indices)
                    call apply_block_data_statement(arena, &
                                                    bd%statement_indices(s), &
                                                    slots, slot_count, error_msg)
                    if (len_trim(error_msg) > 0) return
                end do
            end select
        end do
    end subroutine apply_block_data_initializers

    subroutine apply_block_data_statement(arena, idx, slots, slot_count, error_msg)
        ! A BLOCK DATA body statement is a structured node. Only data_statement
        ! nodes carry initialisers; map each object onto one or more values by
        ! name. A bare array name (no subscript) consumes one value per
        ! element in declaration order (DATA x /v1, v2, .../), so the object
        ! and value lists need not be the same length; anything else pairs
        ! one object with one value.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: idx
        type(common_slot_t), intent(inout) :: slots(:)
        integer, intent(in) :: slot_count
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name, value_text
        integer :: i, s, k, value_cursor

        call set_empty(error_msg)
        if (.not. node_exists(arena, idx)) return
        select type (data => arena%entries(idx)%node)
        type is (data_statement_node)
            if (.not. allocated(data%object_indices) .or. &
                .not. allocated(data%value_indices)) return
            value_cursor = 0
            do i = 1, size(data%object_indices)
                call data_object_name_and_index(arena, data%object_indices(i), &
                                                name, error_msg)
                if (len_trim(error_msg) > 0) return
                if (len(name) == 0) then
                    value_cursor = value_cursor + 1
                    cycle
                end if
                s = 0
                if (index(name, '(') == 0) then
                    s = find_common_slot(slots, slot_count, name)
                end if
                if (s > 0 .and. slots(s)%is_array) then
                    do k = 1, slots(s)%array_size
                        value_cursor = value_cursor + 1
                        if (value_cursor > size(data%value_indices)) then
                            error_msg = 'BLOCK DATA value count does not '// &
                                        'match variable count'
                            return
                        end if
                        call data_value_literal_text(arena, &
                                data%value_indices(value_cursor), value_text, error_msg)
                        if (len_trim(error_msg) > 0) return
                        call add_array_element_init(slots(s), k, value_text, &
                                                    error_msg)
                        if (len_trim(error_msg) > 0) return
                    end do
                else
                    value_cursor = value_cursor + 1
                    if (value_cursor > size(data%value_indices)) then
                        error_msg = 'BLOCK DATA value count does not '// &
                                    'match variable count'
                        return
                    end if
                    call data_value_literal_text(arena, &
                                data%value_indices(value_cursor), value_text, error_msg)
                    if (len_trim(error_msg) > 0) return
                    call store_data_initialization(slots, slot_count, name, &
                                                   value_text, error_msg)
                    if (len_trim(error_msg) > 0) return
                end if
            end do
            if (value_cursor /= size(data%value_indices)) then
                error_msg = 'BLOCK DATA value count does not match variable count'
            end if
        end select
    end subroutine apply_block_data_statement

    subroutine data_object_name_and_index(arena, idx, name, error_msg)
        ! Extract the name and index from a DATA object. Handles both scalar
        ! identifiers (name='x', index=0) and array element references
        ! (name='x(2)', index=2). Implied-do constructs yield empty name.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: idx
        character(len=:), allocatable, intent(out) :: name
        character(len=:), allocatable, intent(out) :: error_msg

        name = ''
        call set_empty(error_msg)
        if (.not. node_exists(arena, idx)) return
        select type (obj => arena%entries(idx)%node)
        type is (identifier_node)
            name = trim(obj%name)
        type is (call_or_subscript_node)
            if (obj%is_array_access .and. allocated(obj%name) .and. &
                allocated(obj%arg_indices)) then
                if (size(obj%arg_indices) == 1) then
                    call extract_array_element_name(arena, obj, name, error_msg)
                else
                    name = ''
                end if
            end if
        end select
    end subroutine data_object_name_and_index

    subroutine extract_array_element_name(arena, arr_node, name, error_msg)
        ! Build the name string for an array element like "x(2)".
        use, intrinsic :: iso_fortran_env, only: int64
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: arr_node
        character(len=:), allocatable, intent(out) :: name
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: index_str_alloc

        name = ''
        call set_empty(error_msg)
        if (.not. allocated(arr_node%name)) return
        if (.not. allocated(arr_node%arg_indices)) return
        if (size(arr_node%arg_indices) /= 1) return
        call extract_literal_integer(arena, arr_node%arg_indices(1), &
                                    index_str_alloc, error_msg)
        if (len_trim(error_msg) > 0) return
        name = trim(arr_node%name)//'('//trim(index_str_alloc)//')'
    end subroutine extract_array_element_name

    subroutine extract_literal_integer(arena, idx, int_str, error_msg)
        ! Extract a literal integer as a string.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: idx
        character(len=:), allocatable, intent(out) :: int_str
        character(len=:), allocatable, intent(out) :: error_msg

        int_str = ''
        call set_empty(error_msg)
        if (.not. node_exists(arena, idx)) then
            error_msg = 'literal integer not found in arena'
            return
        end if
        select type (lit => arena%entries(idx)%node)
        type is (literal_node)
            int_str = trim(lit%value)
        class default
            error_msg = 'array subscript must be a constant literal'
        end select
    end subroutine extract_literal_integer

    subroutine store_data_initialization(slots, slot_count, name, value_text, &
                                        error_msg)
        ! Store DATA value in the appropriate slot. Handles both scalar
        ! variables and array element references (name contains index like "x(2)").
        type(common_slot_t), intent(inout) :: slots(:)
        integer, intent(in) :: slot_count
        character(len=*), intent(in) :: name
        character(len=*), intent(in) :: value_text
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: s, paren_pos, elem_idx
        character(len=:), allocatable :: var_name, index_str
        integer :: io_stat

        call set_empty(error_msg)
        paren_pos = index(name, '(')
        if (paren_pos > 0) then
            var_name = name(1:paren_pos - 1)
            s = find_common_slot(slots, slot_count, var_name)
            if (s <= 0) return
            if (.not. slots(s)%is_array) then
                error_msg = 'subscript applied to non-array: '//var_name
                return
            end if
            index_str = name(paren_pos + 1:index(name, ')') - 1)
            read (index_str, *, iostat=io_stat) elem_idx
            if (io_stat /= 0) then
                error_msg = 'malformed array subscript: '//trim(index_str)
                return
            end if
            call add_array_element_init(slots(s), elem_idx, value_text, error_msg)
        else
            var_name = name
            s = find_common_slot(slots, slot_count, var_name)
            if (s <= 0) return
            slots(s)%has_init = .true.
            slots(s)%init_text = value_text
        end if
    end subroutine store_data_initialization

    subroutine add_array_element_init(slot, elem_idx, value_text, error_msg)
        ! Add an array element initialization to a slot's init arrays.
        type(common_slot_t), intent(inout) :: slot
        integer, intent(in) :: elem_idx
        character(len=*), intent(in) :: value_text
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        if (elem_idx < 1 .or. elem_idx > slot%array_size) then
            error_msg = 'array subscript out of bounds'
            return
        end if
        if (.not. allocated(slot%array_init_indices)) then
            allocate (slot%array_init_indices(slot%array_size))
            allocate (character(len=256) :: slot%array_init_values(slot%array_size))
            slot%array_init_indices = 0
            slot%array_init_values = ''
        end if
        slot%array_init_indices(elem_idx) = elem_idx
        slot%array_init_values(elem_idx) = trim(value_text)
    end subroutine add_array_element_init

    subroutine data_value_literal_text(arena, idx, value_text, error_msg)
        ! Source text of a scalar DATA value literal, used as the slot's static
        ! initialiser. fold_common_initializer parses the text (it accepts a
        ! leading sign embedded in the literal). Non-literal value expressions
        ! are outside the scalar-COMMON scope and are reported unsupported.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: idx
        character(len=:), allocatable, intent(out) :: value_text
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        value_text = ''
        if (.not. node_exists(arena, idx)) then
            error_msg = 'BLOCK DATA value is not a literal'
            return
        end if
        select type (val => arena%entries(idx)%node)
        type is (literal_node)
            value_text = trim(val%value)
        type is (complex_literal_node)
            call complex_literal_text(arena, val, value_text, error_msg)
        class default
            error_msg = 'unsupported BLOCK DATA value expression'
        end select
    end subroutine data_value_literal_text

    subroutine complex_literal_text(arena, node, value_text, error_msg)
        ! Encode a (re, im) DATA literal as "re,im" text; the comma cannot
        ! appear inside either component (both are plain numeric literals),
        ! so fold_complex_common_initializer can split on the first one.
        type(ast_arena_t), intent(in) :: arena
        type(complex_literal_node), intent(in) :: node
        character(len=:), allocatable, intent(out) :: value_text
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: re_text, im_text

        call set_empty(error_msg)
        value_text = ''
        call literal_text_at(arena, node%real_index, re_text, error_msg)
        if (len_trim(error_msg) > 0) return
        call literal_text_at(arena, node%imag_index, im_text, error_msg)
        if (len_trim(error_msg) > 0) return
        value_text = re_text//','//im_text
    end subroutine complex_literal_text

    subroutine literal_text_at(arena, idx, text, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: idx
        character(len=:), allocatable, intent(out) :: text
        character(len=:), allocatable, intent(out) :: error_msg

        text = ''
        call set_empty(error_msg)
        if (.not. node_exists(arena, idx)) then
            error_msg = 'complex literal component not found in arena'
            return
        end if
        select type (n => arena%entries(idx)%node)
        type is (literal_node)
            text = trim(n%value)
        class default
            error_msg = 'unsupported complex literal component'
        end select
    end subroutine literal_text_at

    integer function find_common_slot(slots, slot_count, var_name) result(s)
        type(common_slot_t), intent(in) :: slots(:)
        integer, intent(in) :: slot_count
        character(len=*), intent(in) :: var_name
        integer :: i
        s = 0
        do i = 1, slot_count
            if (slots(i)%var_name == var_name) then
                s = i
                return
            end if
        end do
    end function find_common_slot

    subroutine emit_one_common_global(slot, context, error_msg)
        use, intrinsic :: iso_c_binding, only: c_bool, c_char, c_int32_t, &
                                               c_loc, c_size_t, c_ptr, &
                                               c_double, c_float, c_int64_t
        use liric_session_bindings, only: lr_type_i32_s, lr_type_f32_s, &
                                          lr_type_f64_s
        type(common_slot_t), intent(in) :: slot
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: gname
        character(kind=c_char), allocatable :: c_name(:)
        integer(c_int32_t), target :: init_i32
        real(c_float), target :: init_f32
        real(c_double), target :: init_f64
        type(c_ptr) :: gtype, init_ptr
        integer(c_size_t) :: init_size
        integer(c_int32_t) :: global_id
        logical(c_bool) :: c_false_local

        if (slot%value_kind == VALUE_C4 .or. slot%value_kind == VALUE_C8) then
            call emit_complex_common_global(slot, context, error_msg)
            return
        end if

        c_false_local = .false.
        gname = common_global_name(slot%block_name, slot%pos_in_block)
        call to_c_chars(gname, c_name)

        if (slot%is_array) then
            call emit_array_common_global(slot, context, c_name, error_msg)
            return
        end if

        call fold_common_initializer(slot, init_i32, init_f32, init_f64, &
                                     error_msg)
        if (len_trim(error_msg) > 0) return
        select case (slot%value_kind)
        case (VALUE_I32, VALUE_LOGICAL)
            gtype = lr_type_i32_s(context%session%handle)
            init_ptr = c_loc(init_i32)
            init_size = 4_c_size_t
        case (VALUE_F32)
            gtype = lr_type_f32_s(context%session%handle)
            init_ptr = c_loc(init_f32)
            init_size = 4_c_size_t
        case (VALUE_F64)
            gtype = lr_type_f64_s(context%session%handle)
            init_ptr = c_loc(init_f64)
            init_size = 8_c_size_t
        case default
            error_msg = 'unsupported COMMON variable kind for global emission'
            return
        end select

        global_id = lr_session_global(context%session%handle, c_name, gtype, &
                                      c_false_local, init_ptr, init_size)
        if (global_id < 0_c_int32_t) then
            error_msg = 'LIRIC could not create COMMON global: '//gname
            return
        end if
        call set_empty(error_msg)
    end subroutine emit_one_common_global

    subroutine emit_complex_common_global(slot, context, error_msg)
        ! A COMMON complex scalar stores its real and imaginary parts as two
        ! separate globals (matching the local-variable complex storage model
        ! in session_program_lowering_complex.inc: address=re, element_address
        ! =im), named by appending .re/.im to the block-offset global name.
        use, intrinsic :: iso_c_binding, only: c_bool, c_char, c_loc, &
                                               c_size_t, c_ptr, c_double, &
                                               c_float, c_int32_t
        use liric_session_bindings, only: lr_type_f32_s, lr_type_f64_s
        type(common_slot_t), intent(in) :: slot
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: base_name, gname_re, gname_im
        character(kind=c_char), allocatable :: c_name_re(:), c_name_im(:)
        real(c_float), target :: init_f32_re, init_f32_im
        real(c_double), target :: init_f64_re, init_f64_im
        type(c_ptr) :: gtype, init_ptr_re, init_ptr_im
        integer(c_size_t) :: init_size
        integer(c_int32_t) :: global_id
        logical(c_bool) :: c_false_local

        c_false_local = .false.
        call fold_complex_common_initializer(slot, init_f32_re, init_f32_im, &
                                            init_f64_re, init_f64_im, error_msg)
        if (len_trim(error_msg) > 0) return

        base_name = common_global_name(slot%block_name, slot%pos_in_block)
        gname_re = base_name//'.re'
        gname_im = base_name//'.im'
        call to_c_chars(gname_re, c_name_re)
        call to_c_chars(gname_im, c_name_im)

        if (slot%value_kind == VALUE_C8) then
            gtype = lr_type_f64_s(context%session%handle)
            init_ptr_re = c_loc(init_f64_re)
            init_ptr_im = c_loc(init_f64_im)
            init_size = 8_c_size_t
        else
            gtype = lr_type_f32_s(context%session%handle)
            init_ptr_re = c_loc(init_f32_re)
            init_ptr_im = c_loc(init_f32_im)
            init_size = 4_c_size_t
        end if

        global_id = lr_session_global(context%session%handle, c_name_re, &
                                      gtype, c_false_local, init_ptr_re, &
                                      init_size)
        if (global_id < 0_c_int32_t) then
            error_msg = 'LIRIC could not create COMMON global: '//gname_re
            return
        end if
        global_id = lr_session_global(context%session%handle, c_name_im, &
                                      gtype, c_false_local, init_ptr_im, &
                                      init_size)
        if (global_id < 0_c_int32_t) then
            error_msg = 'LIRIC could not create COMMON global: '//gname_im
            return
        end if
        call set_empty(error_msg)
    end subroutine emit_complex_common_global

    subroutine fold_complex_common_initializer(slot, init_f32_re, init_f32_im, &
                                               init_f64_re, init_f64_im, &
                                               error_msg)
        use, intrinsic :: iso_c_binding, only: c_double, c_float
        type(common_slot_t), intent(in) :: slot
        real(c_float), intent(out) :: init_f32_re, init_f32_im
        real(c_double), intent(out) :: init_f64_re, init_f64_im
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: comma_pos, io_stat
        real(c_double) :: re_val, im_val

        init_f32_re = 0.0_c_float
        init_f32_im = 0.0_c_float
        init_f64_re = 0.0_c_double
        init_f64_im = 0.0_c_double
        call set_empty(error_msg)
        if (.not. slot%has_init) return
        comma_pos = index(slot%init_text, ',')
        if (comma_pos <= 0) then
            error_msg = 'malformed COMMON complex initializer: '//slot%init_text
            return
        end if
        read (slot%init_text(1:comma_pos - 1), *, iostat=io_stat) re_val
        if (io_stat /= 0) then
            error_msg = 'malformed COMMON complex initializer: '//slot%init_text
            return
        end if
        read (slot%init_text(comma_pos + 1:), *, iostat=io_stat) im_val
        if (io_stat /= 0) then
            error_msg = 'malformed COMMON complex initializer: '//slot%init_text
            return
        end if
        init_f32_re = real(re_val, c_float)
        init_f32_im = real(im_val, c_float)
        init_f64_re = re_val
        init_f64_im = im_val
    end subroutine fold_complex_common_initializer

    subroutine emit_array_common_global(slot, context, c_name, error_msg)
        ! Builds and interns the array global in one call: the init buffer
        ! is a local allocatable, so lr_session_global (which copies the
        ! bytes immediately) must run before this subroutine returns and the
        ! buffer goes out of scope (a caller-held c_ptr into it would dangle).
        use, intrinsic :: iso_c_binding, only: c_bool, c_int32_t, c_int64_t, &
                                               c_float, c_double, c_loc, &
                                               c_size_t, c_ptr
        use liric_session_bindings, only: lr_type_i32_s, lr_type_f32_s, &
                                          lr_type_f64_s, lr_type_array_s
        type(common_slot_t), intent(in) :: slot
        type(lowering_context_t), intent(in) :: context
        character(kind=c_char), intent(in) :: c_name(:)
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int32_t), target, allocatable :: array_i32(:)
        real(c_float), target, allocatable :: array_f32(:)
        real(c_double), target, allocatable :: array_f64(:)
        type(c_ptr) :: elem_type, gtype, init_ptr
        integer(c_size_t) :: init_size
        integer(c_int32_t) :: global_id
        logical(c_bool) :: c_false_local
        integer :: i, io_stat

        call set_empty(error_msg)
        c_false_local = .false.
        select case (slot%value_kind)
        case (VALUE_I32, VALUE_LOGICAL)
            allocate (array_i32(slot%array_size))
            array_i32 = 0_c_int32_t
            if (allocated(slot%array_init_values)) then
                do i = 1, size(slot%array_init_values)
                    if (len_trim(slot%array_init_values(i)) > 0) then
                        call fold_array_element_i32(slot%array_init_values(i), &
                                                   array_i32(i), error_msg)
                        if (len_trim(error_msg) > 0) return
                    end if
                end do
            end if
            elem_type = lr_type_i32_s(context%session%handle)
            gtype = lr_type_array_s(context%session%handle, elem_type, &
                                   int(slot%array_size, c_int64_t))
            init_ptr = c_loc(array_i32)
            init_size = int(slot%array_size, c_size_t) * 4_c_size_t
        case (VALUE_F32)
            allocate (array_f32(slot%array_size))
            array_f32 = 0.0_c_float
            if (allocated(slot%array_init_values)) then
                do i = 1, size(slot%array_init_values)
                    if (len_trim(slot%array_init_values(i)) > 0) then
                        read (slot%array_init_values(i), *, iostat=io_stat) &
                            array_f32(i)
                        if (io_stat /= 0) then
                            error_msg = 'malformed real initializer: '// &
                                       slot%array_init_values(i)
                            return
                        end if
                    end if
                end do
            end if
            elem_type = lr_type_f32_s(context%session%handle)
            gtype = lr_type_array_s(context%session%handle, elem_type, &
                                   int(slot%array_size, c_int64_t))
            init_ptr = c_loc(array_f32)
            init_size = int(slot%array_size, c_size_t) * 4_c_size_t
        case (VALUE_F64)
            allocate (array_f64(slot%array_size))
            array_f64 = 0.0_c_double
            if (allocated(slot%array_init_values)) then
                do i = 1, size(slot%array_init_values)
                    if (len_trim(slot%array_init_values(i)) > 0) then
                        read (slot%array_init_values(i), *, iostat=io_stat) &
                            array_f64(i)
                        if (io_stat /= 0) then
                            error_msg = 'malformed double initializer: '// &
                                       slot%array_init_values(i)
                            return
                        end if
                    end if
                end do
            end if
            elem_type = lr_type_f64_s(context%session%handle)
            gtype = lr_type_array_s(context%session%handle, elem_type, &
                                   int(slot%array_size, c_int64_t))
            init_ptr = c_loc(array_f64)
            init_size = int(slot%array_size, c_size_t) * 8_c_size_t
        case default
            error_msg = 'unsupported array COMMON variable kind'
        end select
        if (len_trim(error_msg) > 0) return
        global_id = lr_session_global(context%session%handle, c_name, gtype, &
                                      c_false_local, init_ptr, init_size)
        if (global_id < 0_c_int32_t) then
            error_msg = 'LIRIC could not create COMMON global'
        end if
    end subroutine emit_array_common_global

    subroutine fold_array_element_i32(init_text, init_value, error_msg)
        use, intrinsic :: iso_c_binding, only: c_int32_t, c_int64_t
        character(len=*), intent(in) :: init_text
        integer(c_int32_t), intent(out) :: init_value
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int64_t) :: iconst
        integer :: io_stat

        init_value = 0_c_int32_t
        call set_empty(error_msg)
        call parse_i32_literal(init_text, iconst, error_msg)
        if (len_trim(error_msg) > 0) return
        init_value = int(iconst, c_int32_t)
    end subroutine fold_array_element_i32

    subroutine fold_common_initializer(slot, init_i32, init_f32, init_f64, &
                                       error_msg)
        use, intrinsic :: iso_c_binding, only: c_int32_t, c_int64_t, c_float, &
                                               c_double
        type(common_slot_t), intent(in) :: slot
        integer(c_int32_t), intent(out) :: init_i32
        real(c_float), intent(out) :: init_f32
        real(c_double), intent(out) :: init_f64
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int64_t) :: iconst
        integer :: io_stat

        init_i32 = 0_c_int32_t
        init_f32 = 0.0_c_float
        init_f64 = 0.0_c_double
        call set_empty(error_msg)
        if (.not. slot%has_init) return
        select case (slot%value_kind)
        case (VALUE_I32)
            call parse_i32_literal(slot%init_text, iconst, error_msg)
            if (len_trim(error_msg) > 0) return
            init_i32 = int(iconst, c_int32_t)
        case (VALUE_LOGICAL)
            init_i32 = logical_literal_to_i32(slot%init_text)
        case (VALUE_F32)
            read (slot%init_text, *, iostat=io_stat) init_f32
            if (io_stat /= 0) error_msg = 'malformed COMMON real initializer: '// &
                                          slot%init_text
        case (VALUE_F64)
            read (slot%init_text, *, iostat=io_stat) init_f64
            if (io_stat /= 0) error_msg = 'malformed COMMON real initializer: '// &
                                          slot%init_text
        end select
    end subroutine fold_common_initializer

    subroutine bind_common_block_symbols(arena, node, node_index, context, &
                                         error_msg)
        ! Body-side binding: rebind this occurrence's own COMMON member names
        ! as addressed references into their canonical (block_name,
        ! pos_in_block) slot globals. Resolving only this node's own layout
        ! (rather than a whole-arena name search) keeps binding unambiguous
        ! when another unit reuses the same block name with different member
        ! spellings (#1578, F2018 8.10.3); binding is idempotent, so reaching
        ! any COMMON statement is enough. A member's own type declaration may
        ! come before or after the COMMON statement in source order; when it
        ! has not been reached yet, a bare symbol slot is grown here so the
        ! later declaration finds an existing same-kind (scalar) or
        ! is_common_bound (array) entry and treats itself as a benign
        ! redeclare instead of allocating fresh storage over the binding.
        type(ast_arena_t), intent(in) :: arena
        type(common_block_node), intent(in) :: node
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(common_slot_t) :: slots(COMMON_MAX_SLOTS)
        integer :: slot_count, i, sym

        call set_empty(error_msg)
        slot_count = 0
        call add_common_node_slots(arena, context%root_index, node_index, &
                                   node, slots, slot_count, error_msg)
        if (len_trim(error_msg) > 0) return
        do i = 1, slot_count
            sym = find_symbol(context, slots(i)%var_name)
            if (sym <= 0) then
                call declare_pending_common_symbol(context, slots(i)%var_name, &
                                                   sym)
            end if
            call rebind_common_symbol(slots(i), sym, context, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
    end subroutine bind_common_block_symbols

    subroutine declare_pending_common_symbol(context, name, sym)
        ! Grow a bare symbol slot for a COMMON member whose type declaration
        ! has not been body-walked yet; rebind_common_symbol fills in every
        ! field that matters (kind, address, shape) right after this returns.
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        integer, intent(out) :: sym

        call grow_symbols(context)
        sym = context%symbol_count + 1
        context%symbols(sym)%name = trim(name)
        context%symbol_count = sym
    end subroutine declare_pending_common_symbol

    subroutine rebind_common_symbol(slot, sym, context, error_msg)
        use, intrinsic :: iso_c_binding, only: c_char, c_int32_t, c_int64_t
        type(common_slot_t), intent(in) :: slot
        integer, intent(in) :: sym
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: gname
        character(kind=c_char), allocatable :: c_name(:)
        integer(c_int32_t) :: global_id

        if (slot%value_kind == VALUE_C4 .or. slot%value_kind == VALUE_C8) then
            call rebind_complex_common_symbol(slot, sym, context, error_msg)
            return
        end if

        gname = common_global_name(slot%block_name, slot%pos_in_block)
        call to_c_chars(gname, c_name)
        global_id = lr_session_intern(context%session%handle, c_name)
        if (global_id < 0_c_int32_t) then
            error_msg = 'COMMON global not found: '//gname
            return
        end if
        context%symbols(sym)%value_kind = slot%value_kind
        context%symbols(sym)%has_address = .true.
        context%symbols(sym)%is_reference = .true.
        context%symbols(sym)%is_common_bound = .true.
        context%symbols(sym)%address%kind = LR_OP_KIND_GLOBAL
        context%symbols(sym)%address%payload = int(global_id, c_int64_t)
        context%symbols(sym)%address%typ = lr_type_ptr_s(context%session%handle)
        context%symbols(sym)%address%global_offset = 0_c_int64_t
        if (slot%is_array) then
            context%symbols(sym)%is_array = .true.
            context%symbols(sym)%array_rank = 1
            context%symbols(sym)%array_size = slot%array_size
            context%symbols(sym)%array_dim_sizes(1) = slot%array_size
            context%symbols(sym)%array_dim_lowers(1) = 1
            ! Element GEP addresses off element_address (session_program_
            ! lowering_array_elements.inc), the same field a local array's
            ! alloca fills; without this a COMMON array keeps reading/writing
            ! whatever storage its own declaration allocated instead of the
            ! shared global (#1578).
            context%symbols(sym)%element_address = context%symbols(sym)%address
        end if
        call set_empty(error_msg)
    end subroutine rebind_common_symbol

    subroutine rebind_complex_common_symbol(slot, sym, context, error_msg)
        ! Bind a COMMON complex scalar's re/im parts to the .re/.im globals,
        ! matching the local-variable complex storage model (address=re,
        ! element_address=im) so the existing complex load/store paths work
        ! unchanged (session_program_lowering_complex.inc).
        use, intrinsic :: iso_c_binding, only: c_char, c_int32_t, c_int64_t
        type(common_slot_t), intent(in) :: slot
        integer, intent(in) :: sym
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: base_name, gname_re, gname_im
        character(kind=c_char), allocatable :: c_name_re(:), c_name_im(:)
        integer(c_int32_t) :: global_id_re, global_id_im

        base_name = common_global_name(slot%block_name, slot%pos_in_block)
        gname_re = base_name//'.re'
        gname_im = base_name//'.im'
        call to_c_chars(gname_re, c_name_re)
        call to_c_chars(gname_im, c_name_im)
        global_id_re = lr_session_intern(context%session%handle, c_name_re)
        if (global_id_re < 0_c_int32_t) then
            error_msg = 'COMMON global not found: '//gname_re
            return
        end if
        global_id_im = lr_session_intern(context%session%handle, c_name_im)
        if (global_id_im < 0_c_int32_t) then
            error_msg = 'COMMON global not found: '//gname_im
            return
        end if
        context%symbols(sym)%value_kind = slot%value_kind
        context%symbols(sym)%has_address = .true.
        context%symbols(sym)%is_reference = .true.
        context%symbols(sym)%is_common_bound = .true.
        context%symbols(sym)%address%kind = LR_OP_KIND_GLOBAL
        context%symbols(sym)%address%payload = int(global_id_re, c_int64_t)
        context%symbols(sym)%address%typ = lr_type_ptr_s(context%session%handle)
        context%symbols(sym)%address%global_offset = 0_c_int64_t
        context%symbols(sym)%element_address%kind = LR_OP_KIND_GLOBAL
        context%symbols(sym)%element_address%payload = int(global_id_im, c_int64_t)
        context%symbols(sym)%element_address%typ = &
            lr_type_ptr_s(context%session%handle)
        context%symbols(sym)%element_address%global_offset = 0_c_int64_t
        call set_empty(error_msg)
    end subroutine rebind_complex_common_symbol

    function common_global_name(block_name, pos_in_block) result(gname)
        character(len=*), intent(in) :: block_name
        integer, intent(in) :: pos_in_block
        character(len=:), allocatable :: gname
        character(len=32) :: idx_text
        write (idx_text, '(I0)') pos_in_block
        gname = '.ffc.common.'//block_name//'.'//trim(idx_text)
    end function common_global_name

    ! --- text parsing helpers (shared with the EQUIVALENCE scraper) ---

    subroutine split_csv(text, items, count)
        ! Split a comma-separated list, trimming blanks around each item.
        character(len=*), intent(in) :: text
        character(len=:), allocatable, intent(out) :: items(:)
        integer, intent(out) :: count
        integer :: i, start, max_len, n
        character(len=:), allocatable :: piece

        count = 0
        n = count_commas(text) + 1
        max_len = max(len(text), 1)
        allocate (character(len=max_len) :: items(n))
        items = ''
        start = 1
        do i = 1, len(text) + 1
            if (i > len(text)) then
                piece = trim(adjustl(text(start:)))
                if (len(piece) > 0) then
                    count = count + 1
                    items(count) = piece
                end if
            else if (text(i:i) == ',') then
                if (i > start) then
                    piece = trim(adjustl(text(start:i - 1)))
                else
                    piece = ''
                end if
                if (len(piece) > 0) then
                    count = count + 1
                    items(count) = piece
                end if
                start = i + 1
            end if
        end do
    end subroutine split_csv

    integer function count_commas(text) result(c)
        character(len=*), intent(in) :: text
        integer :: i
        c = 0
        do i = 1, len(text)
            if (text(i:i) == ',') c = c + 1
        end do
    end function count_commas
