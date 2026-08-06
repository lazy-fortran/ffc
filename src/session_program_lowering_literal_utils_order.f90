! fo indexes ordinary MODULE units but does not infer a SUBMODULE's parent
! dependency. This private ordering shim gives the literal submodule an
! explicit dependency edge and has no runtime API or behavior.
module session_program_lowering_literal_utils_order
    use session_program_lowering_impl
    implicit none
    private
end module session_program_lowering_literal_utils_order
