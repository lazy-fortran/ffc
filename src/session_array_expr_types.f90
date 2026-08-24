module session_array_expr_types
    !! Typed plan for one array-valued expression.
    !!
    !! A plan records the rank, shape, element type, and element-producer
    !! metadata of an array expression once, before any element is emitted.
    !! Element lowering then reads the plan instead of re-deriving the element
    !! kind and the iteration extent at every operand site.
    !!
    !! The plan describes an expression, not storage: the canonical
    !! `array_descriptor_t` of `docs/ARRAY_DESCRIPTOR_ABI.md` stays the single
    !! representation for array *objects*, and a plan never becomes a second
    !! descriptor. Extents are held in Fortran column-major dimension order,
    !! so dimension 1 varies fastest over the linear element index.
    implicit none
    private

    integer, parameter, public :: ARRAY_EXPR_MAX_RANK = 7

    type, public :: array_expr_plan_t
        !! Rank, shape, element type, and element-producer metadata of one
        !! array-valued expression.
        integer :: rank = 0
        !! Number of dimensions, 1 to ARRAY_EXPR_MAX_RANK.
        integer :: extents(ARRAY_EXPR_MAX_RANK) = 0
        !! Extent per dimension, dimension 1 first. Entries beyond `rank` are
        !! not part of the value.
        integer :: element_kind = 0
        !! Element type of every value the producer yields, as a lowering
        !! VALUE_* code.
        integer :: target_symbol = 0
        !! Symbol the producer resolves operand kinds and broadcasts against.
        integer :: expr_index = 0
        !! Arena index of the expression the producer walks.
    contains
        procedure :: element_count => array_expr_plan_element_count
    end type array_expr_plan_t

    public :: array_expr_plans_conform

contains

    pure integer function array_expr_plan_element_count(self) result(n)
        !! Number of elements the plan iterates: the product of its extents.
        !! A rank-0 or negative-extent plan yields zero elements.
        class(array_expr_plan_t), intent(in) :: self
        integer :: d

        n = 0
        if (self%rank < 1 .or. self%rank > ARRAY_EXPR_MAX_RANK) return
        n = 1
        do d = 1, self%rank
            if (self%extents(d) < 0) then
                n = 0
                return
            end if
            if (self%extents(d) == 0) then
                n = 0
                return
            end if
            if (self%extents(d) > huge(n)/n) then
                n = 0
                return
            end if
            n = n*self%extents(d)
        end do
    end function array_expr_plan_element_count

    pure logical function array_expr_plans_conform(left, right) result(conform)
        !! Two plans conform when they have equal rank and equal extents in
        !! every dimension. Conformance is a property of the shape alone; the
        !! element kinds may differ and are reconciled by assignment.
        type(array_expr_plan_t), intent(in) :: left
        type(array_expr_plan_t), intent(in) :: right
        integer :: d

        conform = .false.
        if (left%rank /= right%rank) return
        if (left%rank < 1 .or. left%rank > ARRAY_EXPR_MAX_RANK) return
        do d = 1, left%rank
            if (left%extents(d) /= right%extents(d)) return
        end do
        conform = .true.
    end function array_expr_plans_conform

end module session_array_expr_types
