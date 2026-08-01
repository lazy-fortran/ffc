    ! Canonical descriptor access for allocatable arrays (#336).
    !
    ! An `allocatable :: a(:)` / `a(:,:)` entity is described by one
    ! `array_descriptor_t` (`docs/ARRAY_DESCRIPTOR_ABI.md`), replacing the
    ! bespoke 40-byte `{data, lower1, upper1, lower2, upper2}` record. Every
    ! read and write of an allocatable's shape goes through the helpers below,
    ! so the byte offsets appear in exactly one place.
    !
    ! The base pointer stays at offset 0, so "unallocated" remains "base is
    ! null" and no allocation-state test had to change. What changed is that
    ! each dimension now records `(lower_bound, extent, stride_bytes)` rather
    ! than `(lower, upper)`, which is what makes the layout shared with
    ! assumed-shape dummies and automatic arrays.

    integer(c_int64_t) function alloc_desc_dim_offset(dim, field) result(offset)
        integer, intent(in) :: dim
        integer(c_int64_t), intent(in) :: field

        offset = int(ARRAY_DESCRIPTOR_DIM_OFFSET, c_int64_t) &
                 + int(ARRAY_DIMENSION_BYTES, c_int64_t)*int(dim - 1, c_int64_t) &
                 + field
    end function alloc_desc_dim_offset

    subroutine emit_alloc_desc_header(context, descriptor, value_kind, rank, &
                                      error_msg)
        ! Write the rank-invariant header of an allocatable's descriptor. Called
        ! where the entity is declared, so the element kind and rank are known
        ! before any ALLOCATE runs and an unallocated descriptor still describes
        ! its own type.
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: descriptor
        integer, intent(in) :: value_kind
        integer, intent(in) :: rank
        character(len=:), allocatable, intent(out) :: error_msg

        if (.not. emit_i64_store_at(context%session, &
                i64_immediate(context%session, &
                    allocatable_elem_size(value_kind)), descriptor, &
                int(ARRAY_DESCRIPTOR_ELEMENT_SIZE_OFFSET, c_int64_t), &
                error_msg)) return
        call store_descriptor_i32(context, assumed_shape_element_type(value_kind), &
            descriptor, int(ARRAY_DESCRIPTOR_ELEMENT_TYPE_OFFSET, c_int64_t), &
            error_msg)
        if (len_trim(error_msg) > 0) return
        call store_descriptor_i32(context, int(rank, c_int32_t), descriptor, &
            int(ARRAY_DESCRIPTOR_RANK_OFFSET, c_int64_t), error_msg)
        if (len_trim(error_msg) > 0) return
        call store_descriptor_i32(context, 0_c_int32_t, descriptor, &
            int(ARRAY_DESCRIPTOR_RESERVED_OFFSET, c_int64_t), error_msg)
    end subroutine emit_alloc_desc_header

    subroutine emit_alloc_desc_flags(context, descriptor, allocated_state, &
                                     error_msg)
        ! Allocation state. An allocated allocatable array is contiguous and
        ! owns its heap block, which is precisely what makes DEALLOCATE legal
        ! on it; an unallocated one carries no flags at all.
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: descriptor
        logical, intent(in) :: allocated_state
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int32_t) :: flags

        flags = 0_c_int32_t
        if (allocated_state) flags = ARRAY_FLAG_ALLOCATED + ARRAY_FLAG_ASSOCIATED &
                                     + ARRAY_FLAG_OWNS_DATA + ARRAY_FLAG_CONTIGUOUS
        call store_descriptor_i32(context, flags, descriptor, &
            int(ARRAY_DESCRIPTOR_FLAGS_OFFSET, c_int64_t), error_msg)
    end subroutine emit_alloc_desc_flags

    subroutine emit_alloc_desc_set_dim(context, descriptor, dim, lower_i64, &
                                       extent_i64, stride_i64, error_msg)
        ! Write one dimension's lower bound, extent, and byte stride.
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: descriptor
        integer, intent(in) :: dim
        type(lr_operand_desc_t), intent(in) :: lower_i64
        type(lr_operand_desc_t), intent(in) :: extent_i64
        type(lr_operand_desc_t), intent(in) :: stride_i64
        character(len=:), allocatable, intent(out) :: error_msg

        if (.not. emit_i64_store_at(context%session, lower_i64, descriptor, &
                alloc_desc_dim_offset(dim, &
                    int(ARRAY_DIMENSION_LOWER_OFFSET, c_int64_t)), &
                error_msg)) return
        if (.not. emit_i64_store_at(context%session, extent_i64, descriptor, &
                alloc_desc_dim_offset(dim, &
                    int(ARRAY_DIMENSION_EXTENT_OFFSET, c_int64_t)), &
                error_msg)) return
        if (.not. emit_i64_store_at(context%session, stride_i64, descriptor, &
                alloc_desc_dim_offset(dim, &
                    int(ARRAY_DIMENSION_STRIDE_OFFSET, c_int64_t)), &
                error_msg)) return
        call set_empty(error_msg)
    end subroutine emit_alloc_desc_set_dim

    subroutine emit_alloc_desc_load_lower(context, descriptor, dim, lower_i64, &
                                          error_msg)
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: descriptor
        integer, intent(in) :: dim
        type(lr_operand_desc_t), intent(out) :: lower_i64
        character(len=:), allocatable, intent(out) :: error_msg

        if (.not. emit_i64_load_at(context%session, descriptor, &
                alloc_desc_dim_offset(dim, &
                    int(ARRAY_DIMENSION_LOWER_OFFSET, c_int64_t)), &
                lower_i64, error_msg)) return
        call set_empty(error_msg)
    end subroutine emit_alloc_desc_load_lower

    subroutine emit_alloc_desc_load_extent(context, descriptor, dim, extent_i64, &
                                           error_msg)
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: descriptor
        integer, intent(in) :: dim
        type(lr_operand_desc_t), intent(out) :: extent_i64
        character(len=:), allocatable, intent(out) :: error_msg

        if (.not. emit_i64_load_at(context%session, descriptor, &
                alloc_desc_dim_offset(dim, &
                    int(ARRAY_DIMENSION_EXTENT_OFFSET, c_int64_t)), &
                extent_i64, error_msg)) return
        call set_empty(error_msg)
    end subroutine emit_alloc_desc_load_extent

    subroutine emit_alloc_desc_load_upper(context, descriptor, dim, upper_i64, &
                                          error_msg)
        ! upper = lower + extent - 1, recomputed rather than stored, so the
        ! descriptor keeps one representation of the shape.
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: descriptor
        integer, intent(in) :: dim
        type(lr_operand_desc_t), intent(out) :: upper_i64
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: lower_i64, extent_i64, sum_i64

        call emit_alloc_desc_load_lower(context, descriptor, dim, lower_i64, &
                                        error_msg)
        if (len_trim(error_msg) > 0) return
        call emit_alloc_desc_load_extent(context, descriptor, dim, extent_i64, &
                                         error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_i64_binary(context%session, LR_OP_ADD, lower_i64, &
                extent_i64, sum_i64, error_msg)) return
        if (.not. emit_i64_binary(context%session, LR_OP_SUB, sum_i64, &
                i64_immediate(context%session, 1_c_int64_t), upper_i64, &
                error_msg)) return
        call set_empty(error_msg)
    end subroutine emit_alloc_desc_load_upper

    subroutine emit_alloc_desc_allocate_shape(context, descriptor, value_kind, &
                                              rank, extents_i64, error_msg)
        ! Install the shape an ALLOCATE just produced: unit lower bounds, the
        ! requested extents, and contiguous column-major byte strides, with the
        ! allocated/owning flags set. Storage must already be stored at offset 0.
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: descriptor
        integer, intent(in) :: value_kind
        integer, intent(in) :: rank
        type(lr_operand_desc_t), intent(in) :: extents_i64(:)
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: running, next_running, one
        integer :: d

        call emit_alloc_desc_header(context, descriptor, value_kind, rank, &
                                    error_msg)
        if (len_trim(error_msg) > 0) return
        call emit_alloc_desc_flags(context, descriptor, .true., error_msg)
        if (len_trim(error_msg) > 0) return

        one = i64_immediate(context%session, 1_c_int64_t)
        running = i64_immediate(context%session, &
                                allocatable_elem_size(value_kind))
        do d = 1, rank
            call emit_alloc_desc_set_dim(context, descriptor, d, one, &
                                         extents_i64(d), running, error_msg)
            if (len_trim(error_msg) > 0) return
            if (d < rank) then
                if (.not. emit_i64_binary(context%session, LR_OP_MUL, running, &
                        extents_i64(d), next_running, error_msg)) return
                running = next_running
            end if
        end do
        call set_empty(error_msg)
    end subroutine emit_alloc_desc_allocate_shape

    subroutine emit_alloc_desc_clear(context, descriptor, error_msg)
        ! Return the descriptor to the unallocated state: null base, no flags,
        ! and zero extents. The element size, type, and rank are left in place
        ! so a deallocated entity still describes what it can hold.
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: descriptor
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: zero
        integer :: d

        zero = i64_immediate(context%session, 0_c_int64_t)
        if (.not. emit_i64_store_at(context%session, zero, descriptor, &
                0_c_int64_t, error_msg)) return
        call emit_alloc_desc_flags(context, descriptor, .false., error_msg)
        if (len_trim(error_msg) > 0) return
        do d = 1, 2
            if (.not. emit_i64_store_at(context%session, zero, descriptor, &
                    alloc_desc_dim_offset(d, &
                        int(ARRAY_DIMENSION_EXTENT_OFFSET, c_int64_t)), &
                    error_msg)) return
        end do
        call set_empty(error_msg)
    end subroutine emit_alloc_desc_clear
