module empty_program_lowering
    use fortfront, only: ast_arena_t, get_node_type_at
    implicit none
    private

    public :: lower_empty_program_to_llvm

contains

    subroutine lower_empty_program_to_llvm(arena, root_index, llvm_ir, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: root_index
        character(len=:), allocatable, intent(out) :: llvm_ir
        character(len=:), allocatable, intent(out) :: error_msg

        if (root_index <= 0) then
            error_msg = 'FortFront did not return a root program index'
            call set_empty(llvm_ir)
            return
        end if

        if (get_node_type_at(arena, root_index) /= 'program') then
            error_msg = 'ffc MVP only supports a top-level program unit'
            call set_empty(llvm_ir)
            return
        end if

        llvm_ir = 'define i32 @main() {'//new_line('a')// &
                  'entry:'//new_line('a')// &
                  '  ret i32 0'//new_line('a')// &
                  '}'//new_line('a')
        call set_empty(error_msg)
    end subroutine lower_empty_program_to_llvm

    subroutine set_empty(value)
        character(len=:), allocatable, intent(out) :: value

        allocate (character(len=0) :: value)
    end subroutine set_empty

end module empty_program_lowering
