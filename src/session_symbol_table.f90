module session_symbol_table
    !! Lowering symbols keyed by FortFront binding identity (#327).
    !!
    !! ffc's lowering context stores symbols in a flat array indexed by
    !! creation order and, historically, looked them up by their text name.
    !! Text cannot express Fortran's name rules: a BLOCK shadow, a
    !! host-associated name and a USE-renamed name can all share a spelling
    !! and denote different entities. FortFront resolves names and hands back
    !! a `declaration_binding_t`; the triple
    !! `(declaration_node_index, declaration_entity_index, scope_node_index)`
    !! from that binding is a stable identity for the declared entity,
    !! independent of where it is referenced.
    !!
    !! This module owns only the mapping from that identity to a slot in the
    !! lowering context's symbol array. It deliberately knows nothing about
    !! names, types, storage or Fortran scoping rules: FortFront stays the
    !! sole owner of name resolution, and ffc stays the sole owner of storage
    !! and ABI metadata. Keeping the interface this narrow is what makes it
    !! possible to give the lowering include fragments private symbol state
    !! later.
    implicit none
    private

    ! One (declaration, entity, scope) -> symbol slot association. The entity
    ! index separates the names of a multi-name declaration
    ! (integer :: a, b, c), which all share one declaration node.
    type, public :: symbol_binding_t
        integer :: declaration_node_index = 0
        integer :: declaration_entity_index = 0
        integer :: scope_node_index = 0
        integer :: symbol_index = 0
    end type symbol_binding_t

    type, public :: session_symbol_table_t
        type(symbol_binding_t), allocatable :: bindings(:)
        integer :: binding_count = 0
    contains
        procedure :: find_binding => table_find_binding
        procedure :: insert_binding => table_insert_binding
        procedure :: drop_from_symbol => table_drop_from_symbol
        procedure :: reset => table_reset
    end type session_symbol_table_t

contains

    integer function table_find_binding(self, declaration_node_index, &
            declaration_entity_index, &
            scope_node_index) result(symbol_index)
        !! Symbol slot bound to this identity, or 0 when the identity has no
        !! lowering symbol yet. Never falls back to a name comparison.
        class(session_symbol_table_t), intent(in) :: self
        integer, intent(in) :: declaration_node_index
        integer, intent(in) :: declaration_entity_index
        integer, intent(in) :: scope_node_index
        integer :: i

        symbol_index = 0
        if (declaration_node_index <= 0) return
        if (scope_node_index <= 0) return
        if (.not. allocated(self%bindings)) return
        ! Newest first: a re-entered scope (a repeated call to the same
        ! contained procedure) rebinds the same identity to a fresh slot.
        do i = self%binding_count, 1, -1
            if (self%bindings(i)%declaration_node_index /= &
                declaration_node_index) cycle
            if (self%bindings(i)%declaration_entity_index /= &
                declaration_entity_index) cycle
            if (self%bindings(i)%scope_node_index /= scope_node_index) cycle
            symbol_index = self%bindings(i)%symbol_index
            return
        end do
    end function table_find_binding

    subroutine table_insert_binding(self, declaration_node_index, &
            declaration_entity_index, &
            scope_node_index, symbol_index)
        !! Bind an identity to a symbol slot. Any association that already
        !! points at this slot or a later one is dropped first: the lowering
        !! context reuses slot numbers after a BLOCK or a procedure body pops
        !! its locals, so a stale association would otherwise resolve a name
        !! to whatever now occupies the slot.
        class(session_symbol_table_t), intent(inout) :: self
        integer, intent(in) :: declaration_node_index
        integer, intent(in) :: declaration_entity_index
        integer, intent(in) :: scope_node_index
        integer, intent(in) :: symbol_index

        if (declaration_node_index <= 0) return
        if (scope_node_index <= 0) return
        if (symbol_index <= 0) return
        call self%drop_from_symbol(symbol_index)
        call grow_bindings(self)
        self%binding_count = self%binding_count + 1
        self%bindings(self%binding_count)%declaration_node_index = &
            declaration_node_index
        self%bindings(self%binding_count)%declaration_entity_index = &
            declaration_entity_index
        self%bindings(self%binding_count)%scope_node_index = scope_node_index
        self%bindings(self%binding_count)%symbol_index = symbol_index
    end subroutine table_insert_binding

    subroutine table_drop_from_symbol(self, first_symbol_index)
        !! Forget every association whose symbol slot is `first_symbol_index`
        !! or later. Called when the lowering context truncates its symbol
        !! array back to an enclosing scope.
        class(session_symbol_table_t), intent(inout) :: self
        integer, intent(in) :: first_symbol_index
        integer :: i
        integer :: kept

        if (.not. allocated(self%bindings)) return
        kept = 0
        do i = 1, self%binding_count
            if (self%bindings(i)%symbol_index >= first_symbol_index) cycle
            kept = kept + 1
            self%bindings(kept) = self%bindings(i)
        end do
        self%binding_count = kept
    end subroutine table_drop_from_symbol

    subroutine table_reset(self)
        class(session_symbol_table_t), intent(inout) :: self

        self%binding_count = 0
    end subroutine table_reset

    subroutine grow_bindings(self)
        class(session_symbol_table_t), intent(inout) :: self
        type(symbol_binding_t), allocatable :: tmp(:)
        integer :: new_size

        if (.not. allocated(self%bindings)) then
            allocate (self%bindings(64))
            return
        end if
        if (self%binding_count < size(self%bindings)) return
        new_size = 2*size(self%bindings)
        allocate (tmp(new_size))
        tmp(1:self%binding_count) = self%bindings(1:self%binding_count)
        call move_alloc(tmp, self%bindings)
    end subroutine grow_bindings

end module session_symbol_table
