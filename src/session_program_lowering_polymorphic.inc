    ! Monomorphic-subset polymorphism (#141). A class(t) scalar dummy or local
    ! whose dynamic type is only ever its declared type t lowers exactly like
    ! type(t): the derived-type declaration path already routes both spellings
    ! (extracted_derived_type_name treats type(t) and class(t) alike). This file
    ! adds SELECT TYPE dispatch for such a declared-type selector: the dynamic
    ! type equals the declared type, so exactly one arm is ever live and is
    ! chosen statically. Full dynamic dispatch on an extensible allocatable /
    ! pointer class object (a runtime vtable / type-id) stays out of scope and
    ! declines gracefully.

    ! Runtime type identity for class scalars (#417). A class(t) scalar dummy is
    ! passed as the canonical scalar class descriptor of docs/RUNTIME_ABI.md
    ! (data, declared_type, dynamic_type, ownership), built by the caller and
    ! borrowed by the callee. The declared type is fixed by the dummy's
    ! declaration; the dynamic type travels with the value, so re-passing a
    ! class dummy onwards preserves the identity of the original actual instead
    ! of collapsing it to the declared type.

    recursive logical function derived_type_extends(context, type_index, &
                                                    ancestor_index) result(yes)
        ! True when type_index is ancestor_index or extends it, transitively.
        ! This is the type-compatibility test of Fortran 2018 7.3.2.3: an actual
        ! is valid for a class(t) dummy exactly when its declared type is t or
        ! an extension of t.
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: type_index
        integer, intent(in) :: ancestor_index

        yes = .false.
        if (type_index <= 0) return
        if (ancestor_index <= 0) return
        if (type_index == ancestor_index) then
            yes = .true.
            return
        end if
        if (type_index > context%derived_type_count) return
        if (len_trim(context%derived_types(type_index)%parent_name) == 0) return
        yes = derived_type_extends(context, &
            find_derived_type(context, &
                trim(context%derived_types(type_index)%parent_name)), &
            ancestor_index)
    end function derived_type_extends

    subroutine class_dummy_declared_type_index(context, arena, callee_name, &
                                               param_pos, type_index)
        ! Registered index of the declared type of the callee's param_pos-th
        ! dummy when that dummy is class(t); 0 when it is not.
        type(lowering_context_t), intent(in) :: context
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: callee_name
        integer, intent(in) :: param_pos
        integer, intent(out) :: type_index
        character(len=:), allocatable :: type_name

        type_index = 0
        call callee_dummy_declared_derived_type(arena, callee_name, param_pos, &
                                                .true., type_name)
        if (len_trim(type_name) == 0) return
        type_index = find_derived_type(context, type_name)
    end subroutine class_dummy_declared_type_index

    subroutine class_actual_dynamic_type(context, symbol_index, dynamic_id, &
                                         error_msg)
        ! The dynamic type identity of a derived actual. A monomorphic actual's
        ! dynamic type is its declared type, known at compile time. A class
        ! dummy being re-passed already carries a runtime identity, which is
        ! loaded and forwarded so the original value's identity is preserved
        ! across any number of hand-offs.
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        type(lr_operand_desc_t), intent(out) :: dynamic_id
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        if (context%symbols(symbol_index)%is_polymorphic .and. &
            context%symbols(symbol_index)%has_dynamic_type_address) then
            if (.not. emit_i64_load(context%session, &
                    context%symbols(symbol_index)%dynamic_type_address, &
                    dynamic_id, error_msg)) return
            return
        end if
        dynamic_id = i64_immediate(context%session, &
            int(context%symbols(symbol_index)%derived_type_index, c_int64_t))
    end subroutine class_actual_dynamic_type

    subroutine build_class_scalar_descriptor(context, symbol_index, &
                                             declared_type_index, descriptor, &
                                             error_msg)
        ! Borrowed class descriptor for a whole derived actual named by a symbol.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        integer, intent(in) :: declared_type_index
        type(lr_operand_desc_t), intent(out) :: descriptor
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: dynamic_id

        call class_actual_dynamic_type(context, symbol_index, dynamic_id, &
                                       error_msg)
        if (len_trim(error_msg) > 0) return
        call emit_class_scalar_descriptor(context, &
            context%symbols(symbol_index)%element_address, declared_type_index, &
            dynamic_id, descriptor, error_msg)
    end subroutine build_class_scalar_descriptor

    subroutine emit_class_scalar_descriptor(context, data_address, &
                                            declared_type_index, dynamic_id, &
                                            descriptor, error_msg)
        ! Materialise the borrowed scalar class descriptor a class(t) dummy
        ! receives. The descriptor is a caller-frame temporary that points at
        ! storage the caller owns: ownership is BORROWED, so the callee never
        ! frees the data and the data outlives the call.
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: data_address
        integer, intent(in) :: declared_type_index
        type(lr_operand_desc_t), intent(in) :: dynamic_id
        type(lr_operand_desc_t), intent(out) :: descriptor
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: field_addr

        if (.not. emit_alloca_bytes(context%session, &
                i64_immediate(context%session, &
                    int(POLYMORPHIC_DESCRIPTOR_SIZE, c_int64_t)), descriptor, &
                error_msg)) return
        if (.not. emit_ptr_offset(context%session, descriptor, &
                int(POLYMORPHIC_DESCRIPTOR_DATA_OFFSET, c_int64_t), field_addr, &
                error_msg)) return
        if (.not. emit_ptr_store(context%session, data_address, field_addr, &
                error_msg)) return
        if (.not. emit_ptr_offset(context%session, descriptor, &
                int(POLYMORPHIC_DESCRIPTOR_DECLARED_TYPE_OFFSET, c_int64_t), &
                field_addr, error_msg)) return
        if (.not. emit_i64_store(context%session, &
                i64_immediate(context%session, &
                    int(declared_type_index, c_int64_t)), field_addr, &
                error_msg)) return
        if (.not. emit_ptr_offset(context%session, descriptor, &
                int(POLYMORPHIC_DESCRIPTOR_DYNAMIC_TYPE_OFFSET, c_int64_t), &
                field_addr, error_msg)) return
        if (.not. emit_i64_store(context%session, dynamic_id, field_addr, &
                                 error_msg)) return
        if (.not. emit_ptr_offset(context%session, descriptor, &
                int(POLYMORPHIC_DESCRIPTOR_OWNERSHIP_OFFSET, c_int64_t), &
                field_addr, error_msg)) return
        if (.not. emit_i32_store(context%session, &
                i32_immediate(context%session, &
                    int(POLYMORPHIC_OWNERSHIP_BORROWED, c_int64_t)), field_addr, &
                error_msg)) return
        call set_empty(error_msg)
    end subroutine emit_class_scalar_descriptor

    subroutine bind_class_dummy_descriptor(context, symbol_index, declared_index, &
                                           error_msg)
        ! Bind a class(t) dummy to the descriptor the caller passed. The
        ! parameter pointer is the descriptor; the value's storage is the data
        ! field, so every component access keeps using element_address and reads
        ! the declared-type prefix of whatever dynamic type arrived.
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        integer, intent(in) :: declared_index
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: data_addr, data_value, dynamic_addr

        if (.not. emit_ptr_offset(context%session, &
                context%symbols(symbol_index)%address, &
                int(POLYMORPHIC_DESCRIPTOR_DATA_OFFSET, c_int64_t), data_addr, &
                error_msg)) return
        if (.not. emit_ptr_load(context%session, data_addr, data_value, &
                                error_msg)) return
        if (.not. emit_ptr_offset(context%session, &
                context%symbols(symbol_index)%address, &
                int(POLYMORPHIC_DESCRIPTOR_DYNAMIC_TYPE_OFFSET, c_int64_t), &
                dynamic_addr, error_msg)) return
        context%symbols(symbol_index)%class_descriptor_address = &
            context%symbols(symbol_index)%address
        context%symbols(symbol_index)%element_address = data_value
        context%symbols(symbol_index)%dynamic_type_address = dynamic_addr
        context%symbols(symbol_index)%has_dynamic_type_address = .true.
        context%symbols(symbol_index)%is_polymorphic = .true.
        context%symbols(symbol_index)%derived_type_index = declared_index
        call set_empty(error_msg)
    end subroutine bind_class_dummy_descriptor

    ! Type-bound dispatch through vtables (#420). Every derived type that has
    ! type-bound bindings gets one const vtable global, `__ffc_vtable_<type>`,
    ! holding one code address per binding slot. Slot numbering is the binding
    ! numbering of the derived-type table: a child inherits its parent's slots
    ! in order and an override replaces the target in place, so slot k names
    ! the same binding in a type and in every extension of it. One link-unit
    ! table, `__ffc_vtable_table`, maps a dynamic type id to that type's
    ! vtable: the id is the dense index the class descriptor already carries,
    ! so entry i of the table is the vtable of type id i and entry 0 is the
    ! reserved "no type" null. Keeping the mapping in a link-unit table rather
    ! than in a fifth descriptor field leaves the 32-byte scalar class
    ! descriptor of #400 unchanged, so every existing descriptor producer and
    ! consumer keeps working unmodified.

    subroutine type_vtable_symbol_name(context, type_index, name)
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: type_index
        character(len=:), allocatable, intent(out) :: name

        name = '__ffc_vtable_'//trim(context%derived_types(type_index)%name)
    end subroutine type_vtable_symbol_name

    subroutine emit_type_vtables(arena, context, error_msg)
        ! Emit every per-type vtable and the link-unit lookup table. Called
        ! once, before any procedure body is compiled, because a global created
        ! mid-compile loses its data-section initializer and relocations.
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=128), allocatable :: slots(:), table(:)
        character(len=:), allocatable :: vtable_name
        integer :: t, b, n

        call set_empty(error_msg)
        if (context%derived_type_count <= 0) return
        allocate (table(context%derived_type_count + 1))
        table = ''
        do t = 1, context%derived_type_count
            if (context%derived_types(t)%is_imported) cycle
            if (len_trim(context%derived_types(t)%duplicate_binding_name) > 0) then
                error_msg = 'type "'//trim(context%derived_types(t)%name)// &
                    '" declares the type-bound procedure "'// &
                    trim(context%derived_types(t)%duplicate_binding_name)// &
                    '" more than once'
                return
            end if
            n = context%derived_types(t)%binding_count
            if (n <= 0) cycle
            if (.not. allocated(context%derived_types(t)%binding_target_names)) cycle
            allocate (slots(n))
            slots = ''
            do b = 1, n
                ! A binding whose implementation is not defined in this
                ! compilation unit gets a null slot instead of a relocation, so
                ! emitting a vtable never introduces an undefined symbol for a
                ! binding the program never calls.
                if (arena_proc_param_count(arena, &
                        trim(context%derived_types(t)%binding_target_names(b))) &
                    < 0) cycle
                slots(b) = call_emit_name(arena, &
                    trim(context%derived_types(t)%binding_target_names(b)))
            end do
            call type_vtable_symbol_name(context, t, vtable_name)
            call create_pointer_table_global(context%session, vtable_name, &
                                             slots, error_msg)
            deallocate (slots)
            if (len_trim(error_msg) > 0) return
            table(t + 1) = vtable_name
        end do
        if (.not. context%unit_is_module_only) then
            call create_pointer_table_global(context%session, '__ffc_vtable_table', &
                                             table, error_msg)
            if (len_trim(error_msg) > 0) return
            call emit_type_size_table(context, error_msg)
        end if
    end subroutine emit_type_vtables

    subroutine emit_type_size_table(context, error_msg)
        ! Link-unit table of exact storage sizes, indexed the same way as
        ! __ffc_vtable_table: entry i is the byte size of the type whose id is
        ! i, entry 0 is the reserved "no type" zero (#421). Allocating a class
        ! value whose dynamic type is only known at run time reads its concrete
        ! size from here, so the storage is the dynamic type's layout rather
        ! than the declared type's prefix.
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int64_t), allocatable :: sizes(:)
        integer :: t

        call set_empty(error_msg)
        if (context%derived_type_count <= 0) return
        allocate (sizes(context%derived_type_count + 1))
        sizes = 0_c_int64_t
        do t = 1, context%derived_type_count
            sizes(t + 1) = int(total_component_slots(context, t), c_int64_t) * &
                4_c_int64_t
        end do
        call create_i64_table_global(context%session, '__ffc_type_size_table', &
                                     sizes, error_msg)
    end subroutine emit_type_size_table

    ! Scalar class allocatables (#421). A `class(t), allocatable` scalar owns a
    ! 32-byte class descriptor in its frame, the same canonical descriptor a
    ! class dummy receives. Unallocated is the fully null descriptor. ALLOCATE
    ! chooses the concrete dynamic type (from SOURCE=, from the type-spec, or
    ! the declared type), allocates exactly that type's layout, records the
    ! identity and OWNED ownership in the descriptor, and DEALLOCATE finalizes,
    ! frees once, and resets the descriptor to null. Because the descriptor is
    ! the same one #417 defined, SELECT TYPE (#419) and type-bound dispatch
    ! (#420) work through it with no path of their own.

    subroutine bind_class_allocatable_descriptor(context, symbol_index, &
                                                 declared_index, error_msg)
        ! Reserve and null the descriptor of a class allocatable or pointer at
        ! its declaration, so an unallocated/unassociated entity reads as the
        ! null descriptor.
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        integer, intent(in) :: declared_index
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: descriptor, field_addr

        if (.not. emit_alloca_bytes(context%session, &
                i64_immediate(context%session, &
                    int(POLYMORPHIC_DESCRIPTOR_SIZE, c_int64_t)), descriptor, &
                error_msg)) return
        if (.not. emit_ptr_store(context%session, &
                null_ptr_operand(context), descriptor, error_msg)) return
        if (.not. emit_ptr_offset(context%session, descriptor, &
                int(POLYMORPHIC_DESCRIPTOR_DECLARED_TYPE_OFFSET, c_int64_t), &
                field_addr, error_msg)) return
        if (.not. emit_i64_store(context%session, &
                i64_immediate(context%session, int(declared_index, c_int64_t)), &
                field_addr, error_msg)) return
        if (.not. emit_ptr_offset(context%session, descriptor, &
                int(POLYMORPHIC_DESCRIPTOR_DYNAMIC_TYPE_OFFSET, c_int64_t), &
                field_addr, error_msg)) return
        if (.not. emit_i64_store(context%session, &
                i64_immediate(context%session, &
                    int(POLYMORPHIC_TYPE_ID_NONE, c_int64_t)), field_addr, &
                error_msg)) return
        if (.not. emit_ptr_offset(context%session, descriptor, &
                int(POLYMORPHIC_DESCRIPTOR_OWNERSHIP_OFFSET, c_int64_t), &
                field_addr, error_msg)) return
        if (.not. emit_i32_store(context%session, &
                i32_immediate(context%session, &
                    int(POLYMORPHIC_OWNERSHIP_NONE, c_int64_t)), field_addr, &
                error_msg)) return

        context%symbols(symbol_index)%class_descriptor_address = descriptor
        if (.not. emit_ptr_offset(context%session, descriptor, &
                int(POLYMORPHIC_DESCRIPTOR_DYNAMIC_TYPE_OFFSET, c_int64_t), &
                field_addr, error_msg)) return
        context%symbols(symbol_index)%dynamic_type_address = field_addr
        context%symbols(symbol_index)%has_dynamic_type_address = .true.
        context%symbols(symbol_index)%is_polymorphic = .true.
        call set_empty(error_msg)
    end subroutine bind_class_allocatable_descriptor

    subroutine record_class_allocation(context, symbol_index, data_ptr, &
                                       dynamic_id, error_msg)
        ! Publish an allocation into the entity's class descriptor: the data
        ! address and the dynamic type just allocated. Allocatable descriptors
        ! record OWNED storage; class-pointer descriptors deliberately record
        ! no ownership because pointer deallocation/finalization is outside
        ! this slice.
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        type(lr_operand_desc_t), intent(in) :: data_ptr
        type(lr_operand_desc_t), intent(in) :: dynamic_id
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: field_addr

        call set_empty(error_msg)
        if (.not. context%symbols(symbol_index)%has_dynamic_type_address) return
        if (.not. emit_ptr_store(context%session, data_ptr, &
                context%symbols(symbol_index)%class_descriptor_address, &
                error_msg)) return
        if (.not. emit_i64_store(context%session, dynamic_id, &
                context%symbols(symbol_index)%dynamic_type_address, &
                error_msg)) return
        if (.not. emit_ptr_offset(context%session, &
                context%symbols(symbol_index)%class_descriptor_address, &
                int(POLYMORPHIC_DESCRIPTOR_OWNERSHIP_OFFSET, c_int64_t), &
                field_addr, error_msg)) return
        if (context%symbols(symbol_index)%is_pointer) then
            if (.not. emit_i32_store(context%session, &
                    i32_immediate(context%session, &
                        int(POLYMORPHIC_OWNERSHIP_NONE, c_int64_t)), field_addr, &
                    error_msg)) return
        else
            if (.not. emit_i32_store(context%session, &
                    i32_immediate(context%session, &
                        int(POLYMORPHIC_OWNERSHIP_OWNED, c_int64_t)), field_addr, &
                    error_msg)) return
        end if
    end subroutine record_class_allocation

    subroutine release_class_allocation(context, symbol_index, error_msg)
        ! Reset the descriptor after the storage is freed, so the entity reads
        ! as unallocated and the released address can never be handed out or
        ! freed a second time.
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: field_addr

        call set_empty(error_msg)
        if (.not. context%symbols(symbol_index)%has_dynamic_type_address) return
        if (.not. emit_ptr_store(context%session, null_ptr_operand(context), &
                context%symbols(symbol_index)%class_descriptor_address, &
                error_msg)) return
        if (.not. emit_i64_store(context%session, &
                i64_immediate(context%session, &
                    int(POLYMORPHIC_TYPE_ID_NONE, c_int64_t)), &
                context%symbols(symbol_index)%dynamic_type_address, &
                error_msg)) return
        if (.not. emit_ptr_offset(context%session, &
                context%symbols(symbol_index)%class_descriptor_address, &
                int(POLYMORPHIC_DESCRIPTOR_OWNERSHIP_OFFSET, c_int64_t), &
                field_addr, error_msg)) return
        if (.not. emit_i32_store(context%session, &
                i32_immediate(context%session, &
                    int(POLYMORPHIC_OWNERSHIP_NONE, c_int64_t)), field_addr, &
                error_msg)) return
    end subroutine release_class_allocation

    subroutine dynamic_type_size_bytes(context, symbol_index, dynamic_id, &
                                       static_type_index, size_bytes, error_msg)
        ! Exact storage size of the type being allocated. A statically known
        ! dynamic type folds to an immediate; a source whose dynamic type is
        ! only known at run time reads its size from __ffc_type_size_table.
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        type(lr_operand_desc_t), intent(in) :: dynamic_id
        integer, intent(in) :: static_type_index
        type(lr_operand_desc_t), intent(out) :: size_bytes
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: byte_offset, table_base, entry_addr
        character(kind=c_char), allocatable :: c_name(:)
        integer(c_int32_t) :: table_symbol
        integer :: slots

        call set_empty(error_msg)
        if (symbol_index <= 0) then
            slots = total_component_slots(context, static_type_index)
            if (slots <= 0) slots = 1
            size_bytes = i64_immediate(context%session, &
                int(slots, c_int64_t) * 4_c_int64_t)
            return
        end if

        if (.not. emit_i64_binary(context%session, LR_OP_MUL, dynamic_id, &
                i64_immediate(context%session, 8_c_int64_t), byte_offset, &
                error_msg)) return
        call to_c_chars('__ffc_type_size_table', c_name)
        table_symbol = lr_session_intern(context%session%handle, c_name)
        if (table_symbol < 0_c_int32_t) then
            error_msg = 'could not intern the type size table symbol'
            return
        end if
        table_base%kind = LR_OP_KIND_GLOBAL
        table_base%payload = int(table_symbol, c_int64_t)
        table_base%typ = lr_type_ptr_s(context%session%handle)
        table_base%global_offset = 0_c_int64_t
        if (.not. emit_ptr_offset_dyn(context%session, table_base, byte_offset, &
                entry_addr, error_msg)) return
        if (.not. emit_i64_load(context%session, entry_addr, size_bytes, &
                error_msg)) return
    end subroutine dynamic_type_size_bytes

    subroutine lower_derived_allocate_source(arena, node, target_sym, context, &
                                             error_msg)
        ! allocate(x, source=y) / allocate(x, mold=y) where x is a scalar
        ! allocatable of derived or class type. The dynamic type is y's: for a
        ! monomorphic y it is y's declared type, for a polymorphic y it is the
        ! identity y carries. The allocation is that type's exact layout and
        ! SOURCE copies the value whole; MOLD leaves the new value undefined.
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(ast_arena_t), intent(in) :: arena
        type(allocate_statement_node), intent(in) :: node
        integer, intent(in) :: target_sym
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: source_name
        integer :: source_sym, declared_index, source_type_index, dyn_symbol
        integer :: source_expr
        logical :: copy_values
        type(lr_operand_desc_t) :: dynamic_id, size_bytes, data_ptr

        call set_empty(error_msg)
        copy_values = node%source_expr_index > 0
        if (copy_values) then
            source_expr = node%source_expr_index
        else if (node%mold_expr_index > 0) then
            source_expr = node%mold_expr_index
        else
            call unsupported_feature_error('allocate statement', node%line, &
                node%column, 'derived or class allocatable requires SOURCE= '// &
                'or MOLD=', error_msg)
            return
        end if
        if (.not. is_identifier(arena, source_expr)) then
            if (copy_values) then
                call unsupported_feature_error('allocate statement', node%line, &
                    node%column, 'SOURCE= for a scalar derived or class '// &
                    'allocatable must name a variable', error_msg)
            else
                call unsupported_feature_error('allocate statement', node%line, &
                    node%column, 'MOLD= for a scalar derived or class '// &
                    'allocatable must name a variable', error_msg)
            end if
            return
        end if
        call get_identifier_name(arena, source_expr, source_name, &
                                 error_msg)
        if (len_trim(error_msg) > 0) return
        source_sym = find_symbol_compat(context, source_name)
        if (source_sym <= 0) then
            error_msg = 'allocate SOURCE= references an undeclared variable: '// &
                trim(source_name)
            return
        end if
        if (.not. context%symbols(source_sym)%is_derived) then
            error_msg = 'allocate SOURCE= for "'// &
                trim(context%symbols(target_sym)%name)// &
                '" must name a derived value, not '//trim(source_name)
            return
        end if

        declared_index = context%symbols(target_sym)%derived_type_index
        source_type_index = context%symbols(source_sym)%derived_type_index
        ! F2018 C946: SOURCE= must be type compatible with the allocatable.
        if (.not. derived_type_extends(context, source_type_index, &
                                       declared_index)) then
            error_msg = 'allocate SOURCE= type "'// &
                trim(context%derived_types(source_type_index)%name)// &
                '" is not type compatible with "'// &
                trim(context%symbols(target_sym)%name)//'" of declared type "'// &
                trim(context%derived_types(declared_index)%name)//'"'
            return
        end if

        call class_actual_dynamic_type(context, source_sym, dynamic_id, error_msg)
        if (len_trim(error_msg) > 0) return
        dyn_symbol = 0
        if (context%symbols(source_sym)%is_polymorphic .and. &
            context%symbols(source_sym)%has_dynamic_type_address) dyn_symbol = &
            source_sym
        call dynamic_type_size_bytes(context, dyn_symbol, dynamic_id, &
                                     source_type_index, size_bytes, error_msg)
        if (len_trim(error_msg) > 0) return

        if (.not. emit_malloc(context%session, size_bytes, data_ptr, &
                              error_msg)) return
        if (copy_values) then
            if (.not. emit_memcpy(context%session, data_ptr, &
                    context%symbols(source_sym)%element_address, size_bytes, &
                    error_msg)) return
        end if

        context%symbols(target_sym)%element_address = data_ptr
        context%symbols(target_sym)%array_size = &
            max(total_component_slots(context, source_type_index), 1)
        context%symbols(target_sym)%has_address = .true.
        call record_class_allocation(context, target_sym, data_ptr, dynamic_id, &
                                     error_msg)
    end subroutine lower_derived_allocate_source

    integer function type_binding_slot(context, type_index, method_name) &
            result(slot)
        ! Binding slot of method_name in type_index, or 0 when it has none.
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: type_index
        character(len=*), intent(in) :: method_name
        integer :: b

        slot = 0
        if (type_index <= 0) return
        if (type_index > context%derived_type_count) return
        if (.not. allocated(context%derived_types(type_index)%binding_method_names)) &
            return
        do b = 1, context%derived_types(type_index)%binding_count
            if (same_name(context%derived_types(type_index)% &
                          binding_method_names(b), method_name)) then
                slot = b
                return
            end if
        end do
    end function type_binding_slot

    logical function class_receiver_dispatches_dynamically(context, symbol_index)
        ! True when the receiver carries a runtime type identity, i.e. it is a
        ! class entity whose dynamic type is not fixed at compile time. A
        ! type(t) receiver keeps direct dispatch: its dynamic type is its
        ! declared type by definition.
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: symbol_index

        class_receiver_dispatches_dynamically = .false.
        if (symbol_index <= 0) return
        if (.not. context%symbols(symbol_index)%is_polymorphic) return
        class_receiver_dispatches_dynamically = &
            context%symbols(symbol_index)%has_dynamic_type_address
    end function class_receiver_dispatches_dynamically

    subroutine class_dispatch_callee(context, symbol_index, slot, callee_ptr, &
                                     error_msg)
        ! Load the implementation address for binding slot `slot` from the
        ! receiver's dynamic type:
        !   vtable = __ffc_vtable_table[dynamic_type];
        !   callee = vtable[slot - 1];
        ! Both loads read const data the linker filled in, so the sequence
        ! neither writes memory nor aliases the receiver's storage.
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        integer, intent(in) :: slot
        type(lr_operand_desc_t), intent(out) :: callee_ptr
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: dynamic_id, byte_offset, table_base
        type(lr_operand_desc_t) :: entry_addr, vtable, slot_addr
        character(kind=c_char), allocatable :: c_name(:)
        integer(c_int32_t) :: symbol_id

        call set_empty(error_msg)
        if (.not. emit_i64_load(context%session, &
                context%symbols(symbol_index)%dynamic_type_address, dynamic_id, &
                error_msg)) return
        if (.not. emit_i64_binary(context%session, LR_OP_MUL, dynamic_id, &
                i64_immediate(context%session, 8_c_int64_t), byte_offset, &
                error_msg)) return

        call to_c_chars('__ffc_vtable_table', c_name)
        symbol_id = lr_session_intern(context%session%handle, c_name)
        if (symbol_id < 0_c_int32_t) then
            error_msg = 'could not intern the vtable lookup table symbol'
            return
        end if
        table_base%kind = LR_OP_KIND_GLOBAL
        table_base%payload = int(symbol_id, c_int64_t)
        table_base%typ = lr_type_ptr_s(context%session%handle)
        table_base%global_offset = 0_c_int64_t

        if (.not. emit_ptr_offset_dyn(context%session, table_base, byte_offset, &
                entry_addr, error_msg)) return
        if (.not. emit_ptr_load(context%session, entry_addr, vtable, &
                error_msg)) return
        if (.not. emit_ptr_offset(context%session, vtable, &
                int(slot - 1, c_int64_t) * 8_c_int64_t, slot_addr, &
                error_msg)) return
        if (.not. emit_ptr_load(context%session, slot_addr, callee_ptr, &
                error_msg)) return
    end subroutine class_dispatch_callee

    subroutine reject_unsupported_runtime_binding(context, type_index, slot, &
                                                  error_msg)
        ! A runtime vtable slot has one fixed code address. Generic bindings
        ! need call-site overload resolution and deferred bindings have no
        ! address, so neither can be used by this scalar dispatch slice.
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: type_index
        integer, intent(in) :: slot
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: method_name, specifics

        call set_empty(error_msg)
        if (type_index <= 0 .or. type_index > context%derived_type_count) return
        if (slot <= 0 .or. slot > context%derived_types(type_index)%binding_count) return
        method_name = trim(context%derived_types(type_index)%binding_method_names(slot))
        if (allocated(context%derived_types(type_index)%binding_target_names)) then
            if (len_trim(context%derived_types(type_index)% &
                    binding_target_names(slot)) == 0) then
                error_msg = 'deferred type-bound procedure "'//trim(method_name)// &
                    '" is not supported by runtime dispatch'
                return
            end if
        end if
        if (allocated(context%derived_types(type_index)%binding_specific_names)) then
            specifics = trim(context%derived_types(type_index)% &
                binding_specific_names(slot))
            if (index(specifics, ' ') > 0) then
                error_msg = 'generic type-bound procedure "'//trim(method_name)// &
                    '" is not supported by runtime dispatch'
            end if
        end if
    end subroutine reject_unsupported_runtime_binding

    subroutine class_receiver_argument(context, arena, impl_name, position, &
                                       symbol_index, argument, error_msg)
        ! Argument the receiver of a type-bound call occupies. The passed-object
        ! dummy is conventionally class(t), so it takes the same scalar class
        ! descriptor an explicit class actual takes; a type(t) passed-object
        ! dummy keeps the bare address.
        type(lowering_context_t), intent(inout) :: context
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: impl_name
        integer, intent(in) :: position
        integer, intent(in) :: symbol_index
        type(lr_operand_desc_t), intent(out) :: argument
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: declared_index

        call class_dummy_declared_type_index(context, arena, impl_name, position, &
                                             declared_index)
        if (declared_index <= 0) then
            argument = context%symbols(symbol_index)%element_address
            call set_empty(error_msg)
            return
        end if
        call build_class_scalar_descriptor(context, symbol_index, declared_index, &
                                           argument, error_msg)
    end subroutine class_receiver_argument

    ! Runtime SELECT TYPE on a scalar class selector (#419). The selector's
    ! dynamic type identity is loaded once from its class descriptor and each
    ! guard becomes an equality test against the compile-time-known set of type
    ! ids it admits: exactly one id for TYPE IS, the named type and every
    ! registered extension of it for CLASS IS. Guards are emitted in the order
    ! F2018 11.1.11.2 requires (TYPE IS first, then the most specific CLASS IS),
    ! not in source order, so the selected arm does not depend on how the
    ! programmer sorted the guards.

    logical function select_type_selector_is_runtime_polymorphic(context, &
                                                                 sel_index)
        ! True when the selector carries a runtime dynamic type identity, i.e.
        ! it is a class(t) scalar bound to a class descriptor (#417).
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: sel_index

        select_type_selector_is_runtime_polymorphic = .false.
        if (sel_index <= 0) return
        if (.not. context%symbols(sel_index)%is_polymorphic) return
        if (.not. context%symbols(sel_index)%has_dynamic_type_address) return
        if (context%symbols(sel_index)%derived_type_index <= 0) return
        select_type_selector_is_runtime_polymorphic = .true.
    end function select_type_selector_is_runtime_polymorphic

    integer function derived_type_depth(context, type_index) result(depth)
        ! Number of ancestors of type_index; 0 for a root type. Bounded by the
        ! registered type count, so a malformed cycle cannot spin.
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: type_index
        integer :: current

        depth = 0
        current = type_index
        do while (current > 0 .and. depth <= context%derived_type_count)
            if (len_trim(context%derived_types(current)%parent_name) == 0) return
            current = find_derived_type(context, &
                trim(context%derived_types(current)%parent_name))
            if (current <= 0) return
            depth = depth + 1
        end do
    end function derived_type_depth

    subroutine runtime_guard_type_index(arena, guard_index, context, guard_kind, &
                                        type_index, error_msg)
        ! Registered type index named by a guard, with its kind. type_index is 0
        ! for CLASS DEFAULT and for a guard naming an intrinsic type, which can
        ! never match a derived selector.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: guard_index
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable, intent(out) :: guard_kind
        integer, intent(out) :: type_index
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: type_name

        type_index = 0
        call select_type_guard_type_name(arena, guard_index, guard_kind, &
                                         type_name, error_msg)
        if (len_trim(error_msg) > 0) return
        if (trim(guard_kind) == 'class_default') return
        if (len_trim(type_name) == 0) return
        type_index = find_derived_type(context, type_name)
    end subroutine runtime_guard_type_index

    subroutine validate_runtime_guards(arena, guard_indices, context, &
                                       declared_index, error_msg)
        ! F2018 C1162/C1163: no two guards may name the same type, and a guard
        ! naming a type outside the selector's declared hierarchy can never be
        ! selected. Both stay hard errors rather than becoming dead code.
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: guard_indices(:)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: declared_index
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: kind_a, kind_b
        integer :: g, h, index_a, index_b

        call set_empty(error_msg)
        if (.not. allocated(guard_indices)) return
        do g = 1, size(guard_indices)
            call runtime_guard_type_index(arena, guard_indices(g), context, &
                                          kind_a, index_a, error_msg)
            if (len_trim(error_msg) > 0) return
            if (index_a <= 0) cycle
            if (.not. derived_type_extends(context, index_a, declared_index)) then
                error_msg = 'select type guard '//trim(kind_a)//' ('// &
                    trim(context%derived_types(index_a)%name)// &
                    ') cannot match a selector declared class('// &
                    trim(context%derived_types(declared_index)%name)//')'
                return
            end if
            do h = g + 1, size(guard_indices)
                call runtime_guard_type_index(arena, guard_indices(h), context, &
                                              kind_b, index_b, error_msg)
                if (len_trim(error_msg) > 0) return
                if (index_b /= index_a) cycle
                if (trim(kind_b) /= trim(kind_a)) cycle
                error_msg = 'duplicate select type guard '//trim(kind_a)//' ('// &
                    trim(context%derived_types(index_a)%name)//')'
                return
            end do
        end do
    end subroutine validate_runtime_guards

    subroutine ordered_runtime_guards(arena, guard_indices, context, ordered, &
                                      error_msg)
        ! Guard indices in the order F2018 11.1.11.2 selects them: every TYPE IS
        ! guard first (an exact identity match is always preferred), then the
        ! CLASS IS guards from most to least specific. Guards that can never
        ! match a derived selector are dropped.
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: guard_indices(:)
        type(lowering_context_t), intent(inout) :: context
        integer, allocatable, intent(out) :: ordered(:)
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: guard_kind
        integer, allocatable :: kept(:), depth(:)
        integer :: g, n, type_index, best, k, i

        call set_empty(error_msg)
        allocate (ordered(0))
        if (.not. allocated(guard_indices)) return
        allocate (kept(size(guard_indices)))
        allocate (depth(size(guard_indices)))
        n = 0
        do g = 1, size(guard_indices)
            call runtime_guard_type_index(arena, guard_indices(g), context, &
                                          guard_kind, type_index, error_msg)
            if (len_trim(error_msg) > 0) return
            if (type_index <= 0) cycle
            if (trim(guard_kind) /= 'type_is') cycle
            n = n + 1
            kept(n) = guard_indices(g)
            depth(n) = 0
        end do
        k = n
        do g = 1, size(guard_indices)
            call runtime_guard_type_index(arena, guard_indices(g), context, &
                                          guard_kind, type_index, error_msg)
            if (len_trim(error_msg) > 0) return
            if (type_index <= 0) cycle
            if (trim(guard_kind) /= 'class_is') cycle
            k = k + 1
            kept(k) = guard_indices(g)
            depth(k) = derived_type_depth(context, type_index)
        end do
        ! Selection sort the CLASS IS tail by decreasing inheritance depth: a
        ! deeper type is the more specific match.
        do i = n + 1, k - 1
            best = i
            do g = i + 1, k
                if (depth(g) > depth(best)) best = g
            end do
            if (best /= i) then
                call swap_ints(kept(i), kept(best))
                call swap_ints(depth(i), depth(best))
            end if
        end do
        deallocate (ordered)
        allocate (ordered(k))
        ordered = kept(1:k)
    end subroutine ordered_runtime_guards

    subroutine swap_ints(a, b)
        integer, intent(inout) :: a, b
        integer :: tmp

        tmp = a
        a = b
        b = tmp
    end subroutine swap_ints

    subroutine runtime_guard_match_ids(arena, guard_index, context, ids, &
                                       error_msg)
        ! Type ids a guard admits: exactly the named type for TYPE IS; the named
        ! type and every registered extension of it for CLASS IS. The whole set
        ! is known at compile time, so walking ancestry becomes a fixed sequence
        ! of identity comparisons with no runtime table.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: guard_index
        type(lowering_context_t), intent(inout) :: context
        integer, allocatable, intent(out) :: ids(:)
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: guard_kind
        integer, allocatable :: found(:)
        integer :: type_index, t, n

        allocate (ids(0))
        call runtime_guard_type_index(arena, guard_index, context, guard_kind, &
                                      type_index, error_msg)
        if (len_trim(error_msg) > 0) return
        if (type_index <= 0) return
        if (trim(guard_kind) == 'type_is') then
            deallocate (ids)
            allocate (ids(1))
            ids(1) = type_index
            return
        end if
        allocate (found(context%derived_type_count))
        n = 0
        do t = 1, context%derived_type_count
            if (.not. derived_type_extends(context, t, type_index)) cycle
            n = n + 1
            found(n) = t
        end do
        deallocate (ids)
        allocate (ids(n))
        ids = found(1:n)
    end subroutine runtime_guard_match_ids

    subroutine bind_runtime_guard_symbol(context, sel_index, assoc_name, &
                                         guard_type_index, exact, &
                                         selector_element_address)
        ! Bind the associate name inside a guard's arm. It aliases the
        ! selector's storage (same element_address), so writes in the arm reach
        ! the selector, but its declared identity is narrowed to the guard's
        ! type, which is what component access inside the arm resolves against.
        ! After TYPE IS the dynamic type is known exactly, so the associate name
        ! is no longer polymorphic; after CLASS IS it still is.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: sel_index
        character(len=*), intent(in) :: assoc_name
        integer, intent(in) :: guard_type_index
        logical, intent(in) :: exact
        type(lr_operand_desc_t), intent(in), optional :: selector_element_address
        integer :: idx

        idx = find_symbol_compat(context, assoc_name)
        if (idx <= 0) then
            call grow_symbols(context)
            idx = context%symbol_count + 1
            context%symbol_count = idx
        end if
        context%symbols(idx) = context%symbols(sel_index)
        context%symbols(idx)%name = trim(assoc_name)
        context%symbols(idx)%derived_type_index = guard_type_index
        context%symbols(idx)%array_size = &
            total_component_slots(context, guard_type_index)
        if (present(selector_element_address)) then
            context%symbols(idx)%is_array = .false.
            context%symbols(idx)%array_rank = 0
            context%symbols(idx)%is_allocatable = .false.
            context%symbols(idx)%allocatable_static_size = 0
            context%symbols(idx)%element_address = selector_element_address
            context%symbols(idx)%has_address = .true.
            context%symbols(idx)%is_reference = .true.
        end if
        if (exact) then
            context%symbols(idx)%is_polymorphic = .false.
            context%symbols(idx)%has_dynamic_type_address = .false.
        end if
    end subroutine bind_runtime_guard_symbol

    logical function select_type_selector_is_monomorphic(context, sel_index)
        ! True when the selector names a plain declared-type derived object (a
        ! class(t) dummy or local lowered as type(t)): non-allocatable,
        ! non-pointer, with a known derived type. Such an object's dynamic type
        ! is its declared type, so SELECT TYPE resolves statically. An
        ! allocatable / pointer class object may hold an extension at runtime and
        ! needs a real vtable, so it is excluded.
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: sel_index

        select_type_selector_is_monomorphic = .false.
        if (sel_index <= 0) return
        if (context%symbols(sel_index)%value_kind /= VALUE_DERIVED) return
        if (context%symbols(sel_index)%is_allocatable) return
        if (context%symbols(sel_index)%is_pointer) return
        if (context%symbols(sel_index)%derived_type_index <= 0) return
        select_type_selector_is_monomorphic = .true.
    end function select_type_selector_is_monomorphic

    subroutine select_type_guard_type_name(arena, guard_index, guard_kind, &
                                           type_name, error_msg)
        ! Resolve a SELECT TYPE guard's kind ('type_is', 'class_is',
        ! 'class_default') and, for the first two, the derived type name it
        ! names. type_name is empty for class default.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: guard_index
        character(len=:), allocatable, intent(out) :: guard_kind
        character(len=:), allocatable, intent(out) :: type_name
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: tname_index
        integer, allocatable :: body_tmp(:)
        character(len=:), allocatable :: terr, lit_type

        call set_empty(error_msg)
        guard_kind = ''
        type_name = ''
        call get_type_guard_info(arena, guard_index, guard_kind, tname_index, &
                                 body_tmp)
        if (trim(guard_kind) == 'class_default') return
        if (is_identifier(arena, tname_index)) then
            call get_identifier_name(arena, tname_index, type_name, terr)
            if (len_trim(terr) > 0) error_msg = terr
        else if (is_literal(arena, tname_index)) then
            call get_literal_info(arena, tname_index, type_name, lit_type, terr)
            if (len_trim(terr) > 0) error_msg = terr
        else
            error_msg = 'select type guard type name is not resolvable'
        end if
    end subroutine select_type_guard_type_name

    logical function select_type_discriminates_subtypes(arena, guard_indices, &
                                                         context, declared_type, &
                                                         error_msg)
        ! True when a guard names a derived type different from the selector's
        ! declared type. That is the signature of a SELECT TYPE written to
        ! discriminate runtime subtypes (genuine polymorphism), which the
        ! monomorphic static-dispatch path must not silently miscompile - a
        ! non-allocatable class(t) dummy can still receive a child actual by
        ! reference. Such a construct declines to the vtable-needed diagnostic.
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: guard_indices(:)
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: declared_type
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: guard_kind, type_name
        integer :: g

        select_type_discriminates_subtypes = .false.
        call set_empty(error_msg)
        if (.not. allocated(guard_indices)) return
        do g = 1, size(guard_indices)
            call select_type_guard_type_name(arena, guard_indices(g), &
                                             guard_kind, type_name, error_msg)
            if (len_trim(error_msg) > 0) return
            if (trim(guard_kind) == 'class_default') cycle
            if (same_name(type_name, declared_type)) cycle
            ! A guard naming another declared type (not the selector's declared
            ! type and not an intrinsic kind) means subtype discrimination.
            if (find_derived_type(context, type_name) > 0) then
                select_type_discriminates_subtypes = .true.
                return
            end if
        end do
    end function select_type_discriminates_subtypes

    subroutine choose_monomorphic_guard(arena, guard_indices, default_idx, &
                                        declared_type, chosen_index, error_msg)
        ! Pick the single live arm for a declared-type selector. A `type is (t)`
        ! guard naming the declared type wins; otherwise a `class is (t)` guard
        ! naming it; otherwise the class default arm (default_idx, may be 0). An
        ! intrinsic-type guard (type is (integer) etc.) never matches a derived
        ! selector and is skipped.
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: guard_indices(:)
        integer, intent(in) :: default_idx
        character(len=*), intent(in) :: declared_type
        integer, intent(out) :: chosen_index
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: guard_kind, type_name
        integer :: g

        chosen_index = 0
        call set_empty(error_msg)
        if (allocated(guard_indices)) then
            do g = 1, size(guard_indices)
                call select_type_guard_type_name(arena, guard_indices(g), &
                                                 guard_kind, type_name, error_msg)
                if (len_trim(error_msg) > 0) return
                if (trim(guard_kind) == 'type_is' .and. &
                    same_name(type_name, declared_type)) then
                    chosen_index = guard_indices(g)
                    return
                end if
            end do
            do g = 1, size(guard_indices)
                call select_type_guard_type_name(arena, guard_indices(g), &
                                                 guard_kind, type_name, error_msg)
                if (len_trim(error_msg) > 0) return
                if (trim(guard_kind) == 'class_is' .and. &
                    same_name(type_name, declared_type)) then
                    chosen_index = guard_indices(g)
                    return
                end if
            end do
        end if
        chosen_index = default_idx
    end subroutine choose_monomorphic_guard

    subroutine lower_select_type_runtime(arena, node_index, sel_index, &
                                         assoc_name, context, error_msg)
        ! SELECT TYPE driven by the selector's runtime dynamic type identity.
        ! The identity is loaded once, before any arm, and every guard is an
        ! equality test against it; the first test that succeeds enters its arm
        ! and the rest are skipped, so exactly one arm runs.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        integer, intent(in) :: sel_index
        character(len=*), intent(in) :: assoc_name
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: selector_idx, default_idx, g, i, declared_index
        integer :: selector_expr_index, selector_type_index
        integer, allocatable :: guard_indices(:), ordered(:), ids(:)
        integer, allocatable :: body(:)
        character(len=:), allocatable :: guard_kind
        integer :: guard_type_index
        type(lr_operand_desc_t) :: sel_type_id, cond, value
        integer(c_int32_t) :: arm_block, next_block, merge_block
        type(symbol_t), allocatable :: baseline(:)
        integer :: baseline_count
        logical :: terminated, captured
        type(branch_result_t), allocatable :: results(:)
        integer :: nresults
        logical :: selector_is_element
        type(lr_operand_desc_t) :: selector_element_address

        call set_empty(error_msg)
        call get_select_type_info(arena, node_index, selector_idx, &
                                  guard_indices, default_idx)
        selector_is_element = .false.
        selector_expr_index = 0
        if (node_exists(arena, selector_idx)) then
            select type (selector => arena%entries(selector_idx)%node)
            type is (assignment_node)
                selector_expr_index = selector%value_index
            type is (pointer_assignment_node)
                selector_expr_index = selector%target_index
            class default
                selector_expr_index = selector_idx
            end select
        end if
        if (node_exists(arena, selector_expr_index)) then
            select type (selector_expr => arena%entries(selector_expr_index)%node)
            type is (call_or_subscript_node)
                if (selector_expr%base_expr_index <= 0 .and. &
                    allocated(selector_expr%name) .and. &
                    allocated(selector_expr%arg_indices)) then
                    call derived_instance_addr(arena, selector_expr_index, context, &
                        selector_element_address, selector_type_index, error_msg)
                    if (len_trim(error_msg) > 0) return
                    selector_is_element = selector_type_index > 0
                end if
            class default
            end select
        end if
        declared_index = context%symbols(sel_index)%derived_type_index
        call validate_runtime_guards(arena, guard_indices, context, &
                                     declared_index, error_msg)
        if (len_trim(error_msg) > 0) return
        call ordered_runtime_guards(arena, guard_indices, context, ordered, &
                                    error_msg)
        if (len_trim(error_msg) > 0) return

        ! One load of the dynamic identity for the whole construct: every guard
        ! tests the same value, so an arm cannot observe a different type than
        ! the one that selected it.
        if (.not. emit_i32_load(context%session, &
                context%symbols(sel_index)%dynamic_type_address, sel_type_id, &
                error_msg)) return

        merge_block = create_liric_block(context%session)
        call snapshot_symbols(context, baseline, baseline_count)
        nresults = 0
        allocate (results(size(ordered) + 1))

        do g = 1, size(ordered)
            call runtime_guard_type_index(arena, ordered(g), context, guard_kind, &
                                          guard_type_index, error_msg)
            if (len_trim(error_msg) > 0) return
            call runtime_guard_match_ids(arena, ordered(g), context, ids, &
                                         error_msg)
            if (len_trim(error_msg) > 0) return
            arm_block = create_liric_block(context%session)
            next_block = create_liric_block(context%session)
            do i = 1, size(ids)
                if (.not. emit_liric_i32_icmp(context%session, LR_CMP_EQ, &
                        sel_type_id, i32_immediate(context%session, &
                            int(ids(i), c_int64_t)), cond, error_msg)) return
                if (i < size(ids)) then
                    block
                        integer(c_int32_t) :: try_block

                        try_block = create_liric_block(context%session)
                        if (.not. emit_liric_condbr(context%session, cond, &
                                arm_block, try_block, error_msg)) return
                        if (.not. set_liric_block(context%session, try_block, &
                                                  error_msg)) return
                        context%current_block_id = try_block
                        context%current_block_terminated = .false.
                    end block
                else
                    if (.not. emit_liric_condbr(context%session, cond, arm_block, &
                                                next_block, error_msg)) return
                end if
            end do
            if (size(ids) == 0) then
                if (.not. emit_liric_br(context%session, next_block, error_msg)) &
                    return
            end if

            if (.not. set_liric_block(context%session, arm_block, error_msg)) return
            context%current_block_id = arm_block
            context%current_block_terminated = .false.
            call restore_symbols(context, baseline, baseline_count)
            if (selector_is_element) then
                call bind_runtime_guard_symbol(context, sel_index, assoc_name, &
                    guard_type_index, trim(guard_kind) == 'type_is', &
                    selector_element_address)
            else
                call bind_runtime_guard_symbol(context, sel_index, assoc_name, &
                    guard_type_index, trim(guard_kind) == 'type_is')
            end if
            call select_type_guard_body(arena, ordered(g), body)
            terminated = .false.
            if (allocated(body)) then
                call lower_statement_list(arena, body, context, value, &
                                          terminated, error_msg)
                if (len_trim(error_msg) > 0) return
            end if
            call select_type_capture_arm(context, baseline, baseline_count, &
                                         terminated, results(nresults + 1), &
                                         captured)
            if (captured) nresults = nresults + 1
            if (.not. terminated) then
                if (.not. emit_liric_br(context%session, merge_block, error_msg)) &
                    return
            end if
            if (.not. set_liric_block(context%session, next_block, error_msg)) &
                return
            context%current_block_id = next_block
            context%current_block_terminated = .false.
        end do

        call restore_symbols(context, baseline, baseline_count)
        if (default_idx > 0) then
            call lower_select_type_default(arena, default_idx, context, &
                                           terminated, error_msg)
            if (len_trim(error_msg) > 0) return
            call select_type_capture_arm(context, baseline, baseline_count, &
                                         terminated, results(nresults + 1), &
                                         captured)
            if (captured) nresults = nresults + 1
            if (.not. terminated) then
                if (.not. emit_liric_br(context%session, merge_block, error_msg)) &
                    return
            end if
        else
            ! No CLASS DEFAULT: an unmatched dynamic type falls through the
            ! construct without executing any arm.
            call capture_select_result(context, .false., results(nresults + 1))
            nresults = nresults + 1
            if (.not. emit_liric_br(context%session, merge_block, error_msg)) return
        end if

        call restore_symbols(context, baseline, baseline_count)
        call finish_select_merge(context, merge_block, results, nresults, &
                                 error_msg)
    end subroutine lower_select_type_runtime

    subroutine lower_select_type_derived(arena, node_index, sel_index, &
                                         sel_name, assoc_name, context, error_msg)
        ! SELECT TYPE on a monomorphic declared-type selector. The dynamic type
        ! equals the declared type, so the live arm is resolved at compile time
        ! and lowered inline. The associate name aliases the same derived
        ! storage (identical element_address / derived_type_index), so component
        ! access inside the arm reads and writes the selector's slots.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        integer, intent(in) :: sel_index
        character(len=*), intent(in) :: sel_name, assoc_name
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: selector_idx, default_idx, chosen_index, idx
        integer, allocatable :: guard_indices(:)
        integer, allocatable :: body(:)
        character(len=:), allocatable :: declared_type
        type(lr_operand_desc_t) :: value
        logical :: terminated

        call set_empty(error_msg)
        call get_select_type_info(arena, node_index, selector_idx, &
                                  guard_indices, default_idx)
        declared_type = trim(context%derived_types( &
            context%symbols(sel_index)%derived_type_index)%name)

        ! A construct that discriminates other subtypes is genuine polymorphism
        ! and needs a runtime type id; decline rather than statically mispick.
        if (select_type_discriminates_subtypes(arena, guard_indices, context, &
                                               declared_type, error_msg)) then
            if (len_trim(error_msg) > 0) return
            call unsupported_feature_error('select type statement', 0, 0, &
                'direct LIRIC session supports monomorphic select type '// &
                '(guards naming only the selector''s declared type); '// &
                'discriminating runtime subtypes needs a vtable', error_msg)
            return
        end if
        if (len_trim(error_msg) > 0) return

        call choose_monomorphic_guard(arena, guard_indices, default_idx, &
                                      declared_type, chosen_index, error_msg)
        if (len_trim(error_msg) > 0) return
        if (chosen_index <= 0) return

        ! Bind the associate name to the selector's derived storage. For
        ! `select type (v)` the names coincide and the symbol is already bound.
        if (trim(assoc_name) /= trim(sel_name)) then
            idx = find_symbol_compat(context, assoc_name)
            if (idx <= 0) then
                call grow_symbols(context)
                idx = context%symbol_count + 1
                context%symbol_count = idx
            end if
            context%symbols(idx) = context%symbols(sel_index)
            context%symbols(idx)%name = trim(assoc_name)
        end if

        call select_type_guard_body(arena, chosen_index, body)
        if (.not. allocated(body)) return
        terminated = .false.
        call lower_statement_list(arena, body, context, value, terminated, &
                                  error_msg)
    end subroutine lower_select_type_derived
