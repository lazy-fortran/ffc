module ffc_fortfront_queries
    ! Thin wrapper around the FortFront compiler-facing queries that `ffc`
    ! uses for lowering.  The lowerer should reach FortFront through this
    ! module rather than via arena method calls; centralising the surface
    ! makes upstream API churn local to one file.
    use fortfront, only: ast_arena_t, &
                         node_exists, &
                         get_node_line, get_node_column, &
                         get_node_type_at
    implicit none
    private

    public :: node_exists
    public :: get_node_line
    public :: get_node_column
    public :: get_node_type_at

end module ffc_fortfront_queries
