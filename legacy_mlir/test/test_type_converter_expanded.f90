program test_type_converter_expanded
    use iso_c_binding
    use mlir_c_core
    use mlir_c_types
    use fortfc_type_converter
    implicit none

    logical :: all_tests_passed

    print *, "=== Expanded Type Converter Tests ==="
    print *

    all_tests_passed = .true.

    ! Run expanded tests
    if (.not. test_character_types()) all_tests_passed = .false.
    if (.not. test_complex_types()) all_tests_passed = .false.
    if (.not. test_array_types()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All expanded type converter tests passed!"
        stop 0
    else
        print *, "Some expanded type converter tests failed!"
        stop 1
    end if

contains

    function test_character_types() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_type_converter_t) :: converter
        type(mlir_type_t) :: char_type
        character(len=:), allocatable :: type_str
        
        passed = .true.
        
        ! Create context and converter
        context = create_mlir_context()
        converter = create_type_converter(context)
        
        ! Test CHARACTER(20) type
        char_type = converter%create_character_type(20)
        passed = passed .and. char_type%is_valid()
        type_str = converter%get_character_type_string(20)
        passed = passed .and. (type_str == "!fir.char<1,20>")
        
        ! Test CHARACTER(1) type
        char_type = converter%create_character_type(1)
        passed = passed .and. char_type%is_valid()
        type_str = converter%get_character_type_string(1)
        passed = passed .and. (type_str == "!fir.char<1,1>")
        
        if (passed) then
            print *, "PASS: test_character_types"
        else
            print *, "FAIL: test_character_types"
        end if
        
        ! Cleanup
        call destroy_type_converter(converter)
        call destroy_mlir_context(context)
    end function test_character_types

    function test_complex_types() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_type_converter_t) :: converter
        type(mlir_type_t) :: complex_type
        character(len=:), allocatable :: type_str
        
        passed = .true.
        
        ! Create context and converter
        context = create_mlir_context()
        converter = create_type_converter(context)
        
        ! Test COMPLEX*8 (32-bit elements) -> !fir.complex<4>
        complex_type = converter%create_complex_type(32)
        if (.not. complex_type%is_valid()) then
            print *, "FAIL: Complex type creation failed"
            passed = .false.
        end if
        type_str = converter%get_complex_type_string(32)
        if (trim(type_str) /= "!fir.complex<4>") then
            print *, "FAIL: Expected !fir.complex<4>, got", trim(type_str)
            passed = .false.
        end if
        
        ! Test COMPLEX*16 (64-bit elements) -> !fir.complex<8>
        complex_type = converter%create_complex_type(64)
        if (.not. complex_type%is_valid()) then
            print *, "FAIL: Complex type creation failed for 64-bit"
            passed = .false.
        end if
        type_str = converter%get_complex_type_string(64)
        if (trim(type_str) /= "!fir.complex<8>") then
            print *, "FAIL: Expected !fir.complex<8>, got", trim(type_str)
            passed = .false.
        end if
        
        if (passed) then
            print *, "PASS: test_complex_types"
        else
            print *, "FAIL: test_complex_types"
        end if
        
        ! Cleanup
        call destroy_type_converter(converter)
        call destroy_mlir_context(context)
    end function test_complex_types

    function test_array_types() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_type_converter_t) :: converter
        type(mlir_type_t) :: element_type, array_type
        character(len=:), allocatable :: type_str
        integer, dimension(1) :: dims_1d
        integer, dimension(2) :: dims_2d
        
        passed = .true.
        
        ! Create context and converter
        context = create_mlir_context()
        converter = create_type_converter(context)
        
        ! Test 1D array: INTEGER, DIMENSION(10) -> !fir.array<10xi32>
        element_type = converter%create_integer_type(32)
        dims_1d = [10]
        array_type = converter%create_array_type(element_type, dims_1d)
        passed = passed .and. array_type%is_valid()
        type_str = converter%get_array_type_string("i32", dims_1d)
        passed = passed .and. (type_str == "!fir.array<10xi32>")
        
        ! Test 2D array: REAL, DIMENSION(5,10) -> !fir.array<5x10xf32>
        element_type = converter%create_float_type(32)
        dims_2d = [5, 10]
        array_type = converter%create_array_type(element_type, dims_2d)
        passed = passed .and. array_type%is_valid()
        type_str = converter%get_array_type_string("f32", dims_2d)
        passed = passed .and. (type_str == "!fir.array<5x10xf32>")
        
        ! Test assumed-shape array: REAL, DIMENSION(:,:) -> !fir.array<?x?xf32>
        dims_2d = [-1, -1]  ! -1 indicates assumed-shape
        type_str = converter%get_array_type_string("f32", dims_2d)
        passed = passed .and. (type_str == "!fir.array<?x?xf32>")
        
        if (passed) then
            print *, "PASS: test_array_types"
        else
            print *, "FAIL: test_array_types"
        end if
        
        ! Cleanup
        call destroy_type_converter(converter)
        call destroy_mlir_context(context)
    end function test_array_types

end program test_type_converter_expanded