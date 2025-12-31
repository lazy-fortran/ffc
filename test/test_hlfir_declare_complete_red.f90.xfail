program test_hlfir_declare_complete_red
    ! RED Test: Complete hlfir.declare with Fortran variable semantics
    ! This tests shape, type parameters, attributes as per Epic 8.1 requirements
    use iso_fortran_env, only: error_unit
    implicit none
    
    ! Test must verify:
    ! 1. All variable types are properly declared (integer, real, logical, character)
    ! 2. Array shapes are correctly encoded
    ! 3. Type parameters are preserved (e.g., character length)
    ! 4. Variable attributes are maintained (parameter, allocatable, pointer)
    
    character(len=:), allocatable :: test_program, output_file, compile_cmd
    integer :: exit_code, unit, ios
    character(len=1024) :: line
    logical :: test_passed = .true.
    
    ! Tracking what we find
    logical :: found_integer_x = .false.
    logical :: found_real_y = .false.
    logical :: found_logical_flag = .false.
    logical :: found_character_name = .false.
    logical :: found_array_with_shape = .false.
    logical :: found_parameter_const = .false.
    
    print *, "=== Test: Complete hlfir.declare with variable semantics ==="
    print *, "Testing: shape, type parameters, attributes"
    print *, "This test verifies proper HLFIR generation for all Fortran variable types"
    print *
    
    ! Create a Fortran program with various variable types and attributes
    test_program = "test_complete_declare.f90"
    open(newunit=unit, file=test_program, status='replace')
    write(unit, '(A)') 'program test_complete_declare'
    write(unit, '(A)') '    implicit none'
    write(unit, '(A)') '    ! Basic types'
    write(unit, '(A)') '    integer :: x'
    write(unit, '(A)') '    real :: y' 
    write(unit, '(A)') '    logical :: flag'
    write(unit, '(A)') '    character(len=10) :: name'
    write(unit, '(A)') '    ! Arrays with explicit shape'
    write(unit, '(A)') '    integer, dimension(5,3) :: matrix'
    write(unit, '(A)') '    ! Parameter (constant)'
    write(unit, '(A)') '    real, parameter :: pi = 3.14159'
    write(unit, '(A)') '    '
    write(unit, '(A)') '    x = 42'
    write(unit, '(A)') '    y = 3.14'
    write(unit, '(A)') '    flag = .true.'
    write(unit, '(A)') '    name = "test"'
    write(unit, '(A)') '    matrix(1,1) = 100'
    write(unit, '(A)') 'end program test_complete_declare'
    close(unit)
    
    ! Try to compile with HLFIR output using MLIR C API
    output_file = "test_complete_declare_hlfir.mlir"
    compile_cmd = "./ffc " // test_program // " --emit-hlfir -o " // output_file
    
    print *, "Compiling: ", trim(compile_cmd)
    call execute_command_line(compile_cmd, exitstat=exit_code)
    
    if (exit_code /= 0) then
        print *, "FAIL: Compilation failed with exit code:", exit_code
        print *, "Expected: Should compile and generate proper HLFIR"
        stop 1
    end if
    
    ! Check the generated HLFIR for completeness
    open(newunit=unit, file=output_file, status='old', iostat=ios)
    if (ios /= 0) then
        print *, "FAIL: Could not open output file"
        stop 1
    end if
    
    ! Scan the output for expected declarations
    do
        read(unit, '(A)', iostat=ios) line
        if (ios /= 0) exit
        
        ! Check for integer x with proper type
        if (index(line, 'hlfir.declare') > 0 .and. &
            index(line, 'name = "x"') > 0 .and. &
            index(line, '!fir.ref<i32>') > 0) then
            found_integer_x = .true.
        end if
        
        ! Check for real y with proper type
        if (index(line, 'hlfir.declare') > 0 .and. &
            index(line, 'name = "y"') > 0 .and. &
            index(line, '!fir.ref<f32>') > 0) then
            found_real_y = .true.
        end if
        
        ! Check for logical flag
        if (index(line, 'hlfir.declare') > 0 .and. &
            index(line, 'name = "flag"') > 0 .and. &
            index(line, '!fir.ref<i1>') > 0) then
            found_logical_flag = .true.
        end if
        
        ! Check for character with length parameter
        if (index(line, 'hlfir.declare') > 0 .and. &
            index(line, 'name = "name"') > 0 .and. &
            index(line, '!fir.char<1,10>') > 0) then
            found_character_name = .true.
        end if
        
        ! Check for array with proper shape (5x3)
        if (index(line, 'hlfir.declare') > 0 .and. &
            index(line, 'name = "matrix"') > 0 .and. &
            index(line, '!fir.array<5x3xi32>') > 0) then
            found_array_with_shape = .true.
        end if
        
        ! Check for parameter constant
        if (index(line, 'name = "pi"') > 0 .and. &
            (index(line, 'fir.global') > 0 .or. &
             index(line, 'arith.constant') > 0)) then
            found_parameter_const = .true.
        end if
    end do
    close(unit)
    
    ! Report results
    print *, "Checking generated HLFIR for completeness:"
    
    if (found_integer_x) then
        print *, "  PASS: Found integer x with i32 type"
    else
        print *, "  FAIL: Missing proper integer x declaration"
        test_passed = .false.
    end if
    
    if (found_real_y) then
        print *, "  PASS: Found real y with f32 type"
    else
        print *, "  FAIL: Missing proper real y declaration"
        test_passed = .false.
    end if
    
    if (found_logical_flag) then
        print *, "  PASS: Found logical flag with i1 type"
    else
        print *, "  FAIL: Missing proper logical flag declaration"
        test_passed = .false.
    end if
    
    if (found_character_name) then
        print *, "  PASS: Found character name with length 10"
    else
        print *, "  FAIL: Missing proper character declaration with length"
        test_passed = .false.
    end if
    
    if (found_array_with_shape) then
        print *, "  PASS: Found array matrix with shape 5x3"
    else
        print *, "  FAIL: Missing proper array declaration with shape"
        test_passed = .false.
    end if
    
    if (found_parameter_const) then
        print *, "  PASS: Found parameter constant pi"
    else
        print *, "  FAIL: Missing proper parameter constant handling"
        test_passed = .false.
    end if
    
    print *
    if (test_passed) then
        print *, "SUCCESS: All declarations found with correct semantics!"
        stop 0
    else
        print *, "FAILURE: Missing proper HLFIR declarations"
        print *, "Implementation needs to handle all Fortran variable types correctly"
        stop 1
    end if
    
end program test_hlfir_declare_complete_red