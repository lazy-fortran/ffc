program test_type_helpers
    use iso_c_binding
    use mlir_c_core
    use mlir_c_types
    use fortfc_type_converter
    use type_conversion_helpers
    implicit none

    logical :: all_tests_passed

    print *, "=== Type Conversion Helpers Tests ==="
    print *

    all_tests_passed = .true.

    ! Run all tests - these will fail initially (RED phase)
    if (.not. test_type_builder_helpers()) all_tests_passed = .false.
    if (.not. test_array_shape_extraction()) all_tests_passed = .false.
    if (.not. test_reference_type_wrapping()) all_tests_passed = .false.
    if (.not. test_type_equivalence_checking()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All type conversion helpers tests passed!"
        stop 0
    else
        print *, "Some type conversion helpers tests failed!"
        stop 1
    end if

contains

    function test_type_builder_helpers() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_type_converter_t) :: converter
        type(type_helper_t) :: helpers
        character(len=:), allocatable :: descriptor
        integer, dimension(3) :: dims
        
        passed = .true.
        
        ! Create context and helpers
        context = create_mlir_context()
        converter = create_type_converter(context)
        helpers = create_type_helpers(converter)
        passed = passed .and. helpers%is_valid()
        
        ! Test array descriptor generation
        dims = [10, 20, 30]
        descriptor = helpers%get_array_descriptor("i32", dims)
        passed = passed .and. (descriptor == "10x20x30xi32")
        
        ! Test assumed-shape array descriptor
        dims = [-1, -1, 5]  ! -1 indicates assumed-shape
        descriptor = helpers%get_array_descriptor("f64", dims)
        passed = passed .and. (descriptor == "?x?x5xf64")
        
        ! Test derived type name mangling
        descriptor = helpers%mangle_derived_type_name("person")
        passed = passed .and. (descriptor == "_QTperson")
        
        descriptor = helpers%mangle_derived_type_name("my_type")
        passed = passed .and. (descriptor == "_QTmy_type")
        
        if (passed) then
            print *, "PASS: test_type_builder_helpers"
        else
            print *, "FAIL: test_type_builder_helpers"
        end if
        
        ! Cleanup
        call destroy_type_helpers(helpers)
        call destroy_type_converter(converter)
        call destroy_mlir_context(context)
    end function test_type_builder_helpers

    function test_array_shape_extraction() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_type_converter_t) :: converter
        type(type_helper_t) :: helpers
        type(array_info_t) :: array_info
        integer, dimension(:), allocatable :: extracted_dims
        
        passed = .true.
        
        ! Create context and helpers
        context = create_mlir_context()
        converter = create_type_converter(context)
        helpers = create_type_helpers(converter)
        
        ! Test fixed-size array shape extraction
        array_info = create_test_array_info([5, 10], .false., .false.)
        extracted_dims = helpers%extract_array_shape(array_info)
        passed = passed .and. allocated(extracted_dims)
        passed = passed .and. (size(extracted_dims) == 2)
        passed = passed .and. (extracted_dims(1) == 5)
        passed = passed .and. (extracted_dims(2) == 10)
        
        ! Test assumed-shape array
        array_info = create_test_array_info([-1, -1], .false., .false.)
        extracted_dims = helpers%extract_array_shape(array_info)
        passed = passed .and. (size(extracted_dims) == 2)
        passed = passed .and. (extracted_dims(1) == -1)
        passed = passed .and. (extracted_dims(2) == -1)
        
        ! Test array type classification
        array_info = create_test_array_info([10], .false., .false.)
        passed = passed .and. (.not. helpers%is_assumed_shape(array_info))
        passed = passed .and. (.not. helpers%is_allocatable(array_info))
        passed = passed .and. (.not. helpers%is_pointer(array_info))
        
        array_info = create_test_array_info([-1], .false., .false.)
        passed = passed .and. helpers%is_assumed_shape(array_info)
        
        array_info = create_test_array_info([10], .true., .false.)
        passed = passed .and. helpers%is_allocatable(array_info)
        
        array_info = create_test_array_info([10], .false., .true.)
        passed = passed .and. helpers%is_pointer(array_info)
        
        if (passed) then
            print *, "PASS: test_array_shape_extraction"
        else
            print *, "FAIL: test_array_shape_extraction"
        end if
        
        ! Cleanup
        if (allocated(extracted_dims)) deallocate(extracted_dims)
        call destroy_type_helpers(helpers)
        call destroy_type_converter(converter)
        call destroy_mlir_context(context)
    end function test_array_shape_extraction

    function test_reference_type_wrapping() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_type_converter_t) :: converter
        type(type_helper_t) :: helpers
        type(mlir_type_t) :: base_type, wrapped_type
        character(len=:), allocatable :: type_str
        
        passed = .true.
        
        ! Create context and helpers
        context = create_mlir_context()
        converter = create_type_converter(context)
        helpers = create_type_helpers(converter)
        
        ! Test simple reference wrapping: i32 -> !fir.ref<i32>
        base_type = converter%create_integer_type(32)
        wrapped_type = helpers%wrap_with_reference_type(base_type)
        passed = passed .and. wrapped_type%is_valid()
        type_str = helpers%get_reference_type_string("i32")
        passed = passed .and. (type_str == "!fir.ref<i32>")
        
        ! Test box wrapping: !fir.array<?xi32> -> !fir.box<!fir.array<?xi32>>
        type_str = helpers%wrap_with_box_type("!fir.array<?xi32>")
        passed = passed .and. (type_str == "!fir.box<!fir.array<?xi32>>")
        
        ! Test heap wrapping: !fir.array<?xi32> -> !fir.heap<!fir.array<?xi32>>
        type_str = helpers%wrap_with_heap_type("!fir.array<?xi32>")
        passed = passed .and. (type_str == "!fir.heap<!fir.array<?xi32>>")
        
        ! Test pointer wrapping: i32 -> !fir.ptr<i32>
        type_str = helpers%wrap_with_pointer_type("i32")
        passed = passed .and. (type_str == "!fir.ptr<i32>")
        
        ! Test complex wrapping: allocatable array
        ! REAL, ALLOCATABLE, DIMENSION(:) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
        type_str = helpers%create_allocatable_array_type_string("f32", 1)
        passed = passed .and. (type_str == "!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>")
        
        if (passed) then
            print *, "PASS: test_reference_type_wrapping"
        else
            print *, "FAIL: test_reference_type_wrapping"
        end if
        
        ! Cleanup
        call destroy_type_helpers(helpers)
        call destroy_type_converter(converter)
        call destroy_mlir_context(context)
    end function test_reference_type_wrapping

    function test_type_equivalence_checking() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_type_converter_t) :: converter
        type(type_helper_t) :: helpers
        type(mlir_type_t) :: type1, type2, type3
        character(len=:), allocatable :: element_type
        
        passed = .true.
        
        ! Create context and helpers
        context = create_mlir_context()
        converter = create_type_converter(context)
        helpers = create_type_helpers(converter)
        
        ! Test basic type equivalence
        type1 = converter%create_integer_type(32)
        type2 = converter%create_integer_type(32)
        type3 = converter%create_integer_type(64)
        
        
        if (.not. helpers%types_equivalent(type1, type2)) then
            print *, "FAIL: types_equivalent(type1, type2) should be true"
            passed = .false.
        end if
        if (helpers%types_equivalent(type1, type3)) then
            print *, "FAIL: types_equivalent(type1, type3) should be false"
            passed = .false.
        end if
        
        ! Test type compatibility checking
        if (.not. helpers%types_compatible(type1, type2)) then
            print *, "FAIL: types_compatible(type1, type2) should be true"
            passed = .false.
        end if
        if (helpers%types_compatible(type1, type3)) then
            print *, "FAIL: types_compatible(type1, type3) should be false"
            passed = .false.
        end if
        
        ! Test element type extraction
        element_type = helpers%get_element_type("!fir.array<10xi32>")
        if (element_type /= "i32") then
            print *, "FAIL: get_element_type array test, got:", element_type
            passed = .false.
        end if
        
        element_type = helpers%get_element_type("!fir.box<!fir.array<?xf64>>")
        if (element_type /= "f64") then
            print *, "FAIL: get_element_type box test, got:", element_type
            passed = .false.
        end if
        
        element_type = helpers%get_element_type("!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>")
        if (element_type /= "f32") then
            print *, "FAIL: get_element_type ref test, got:", element_type
            passed = .false.
        end if
        
        ! Test type classification
        if (.not. helpers%is_integer_type_string("i32")) then
            print *, "FAIL: is_integer_type_string(i32) should be true"
            passed = .false.
        end if
        if (.not. helpers%is_integer_type_string("i64")) then
            print *, "FAIL: is_integer_type_string(i64) should be true"
            passed = .false.
        end if
        if (helpers%is_integer_type_string("f32")) then
            print *, "FAIL: is_integer_type_string(f32) should be false"
            passed = .false.
        end if
        
        if (.not. helpers%is_float_type_string("f32")) then
            print *, "FAIL: is_float_type_string(f32) should be true"
            passed = .false.
        end if
        if (.not. helpers%is_float_type_string("f64")) then
            print *, "FAIL: is_float_type_string(f64) should be true"
            passed = .false.
        end if
        if (helpers%is_float_type_string("i32")) then
            print *, "FAIL: is_float_type_string(i32) should be false"
            passed = .false.
        end if
        
        if (.not. helpers%is_array_type_string("!fir.array<10xi32>")) then
            print *, "FAIL: is_array_type_string array test should be true"
            passed = .false.
        end if
        if (helpers%is_array_type_string("i32")) then
            print *, "FAIL: is_array_type_string scalar test should be false"
            passed = .false.
        end if
        
        if (passed) then
            print *, "PASS: test_type_equivalence_checking"
        else
            print *, "FAIL: test_type_equivalence_checking"
        end if
        
        ! Cleanup
        call destroy_type_helpers(helpers)
        call destroy_type_converter(converter)
        call destroy_mlir_context(context)
    end function test_type_equivalence_checking

    ! Helper function to create test array info
    function create_test_array_info(dimensions, is_allocatable, is_pointer) result(info)
        integer, dimension(:), intent(in) :: dimensions
        logical, intent(in) :: is_allocatable, is_pointer
        type(array_info_t) :: info
        
        info%dimensions = dimensions
        info%is_allocatable = is_allocatable
        info%is_pointer = is_pointer
        info%rank = size(dimensions)
    end function create_test_array_info

end program test_type_helpers