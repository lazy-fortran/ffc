! HLFIR Helper Functions Module
! This module contains helper functions for generating HLFIR operations
module mlir_hlfir_helpers
    implicit none

    private
    public :: generate_hlfir_constant_code, generate_hlfir_load_code, generate_hlfir_string_literal_code

contains

    ! Helper function for integer to string conversion
    function int_to_str(i) result(str)
        integer, intent(in) :: i
        character(len=20) :: str
        write(str, '(I0)') i
    end function int_to_str

    ! Helper function to generate HLFIR constant expressions
    function generate_hlfir_constant_code(ssa_val, value, mlir_type, indent_str) result(mlir)
        character(len=*), intent(in) :: ssa_val, value, mlir_type, indent_str
        character(len=:), allocatable :: mlir

        mlir = indent_str//ssa_val//" = hlfir.expr { %c = fir.constant "//value// &
               " : "//mlir_type//"; fir.result %c : "//mlir_type//" } : !hlfir.expr<"//mlir_type//">"//new_line('a')
    end function generate_hlfir_constant_code

    ! Helper function to generate HLFIR designate and load operations
    function generate_hlfir_load_code(designate_ssa, load_ssa, memref_ssa, indices, element_type, indent_str) result(mlir)
        character(len=*), intent(in) :: designate_ssa, load_ssa, memref_ssa, indices, element_type, indent_str
        character(len=:), allocatable :: mlir

        ! Generate hlfir.designate
        mlir = indent_str//designate_ssa//" = hlfir.designate "//memref_ssa//"["//indices//"] : " // &
               "(!fir.ref<!fir.array<?x"//element_type//">>, index) -> !fir.ref<"//element_type//">"//new_line('a')
        
        ! Create HLFIR expression from the load
        mlir = mlir//indent_str//load_ssa//" = hlfir.expr { %val = fir.load "//designate_ssa// &
               " : !fir.ref<"//element_type//">; fir.result %val : "//element_type//" } : " // &
               "!hlfir.expr<"//element_type//">"//new_line('a')
    end function generate_hlfir_load_code

    ! Helper function to generate HLFIR string literal expressions
    function generate_hlfir_string_literal_code(ssa_val, string_value, indent_str) result(mlir)
        character(len=*), intent(in) :: ssa_val, string_value, indent_str
        character(len=:), allocatable :: mlir
        integer :: str_len

        str_len = len(string_value)
        mlir = indent_str//ssa_val//" = hlfir.expr { %s = fir.string_lit """//string_value// &
               """ : !fir.char<1,"//trim(adjustl(int_to_str(str_len)))//">; " // &
               "fir.result %s : !fir.char<1,"//trim(adjustl(int_to_str(str_len)))//"> } : " // &
               "!hlfir.expr<!fir.char<1,"//trim(adjustl(int_to_str(str_len)))//">>"//new_line('a')
    end function generate_hlfir_string_literal_code

end module mlir_hlfir_helpers