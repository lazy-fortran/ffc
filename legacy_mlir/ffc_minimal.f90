program ffc_minimal
    ! Minimal FFC compiler that can handle --emit-hlfir flag
    implicit none
    
    character(len=256) :: input_file, output_file, arg
    logical :: emit_hlfir = .false.
    integer :: i, num_args, ios
    
    ! Get command line arguments
    num_args = command_argument_count()
    if (num_args < 1) then
        print *, "Usage: ffc <input.f90> [options]"
        print *, "Options:"
        print *, "  -o <file>     Output file"
        print *, "  --emit-hlfir  Emit HLFIR code"
        stop 1
    end if
    
    ! Get input file
    call get_command_argument(1, input_file)
    
    ! Default output file
    output_file = "output.mlir"
    
    ! Parse options
    i = 2
    do while (i <= num_args)
        call get_command_argument(i, arg)
        
        select case (trim(arg))
        case ("--emit-hlfir")
            emit_hlfir = .true.
            
        case ("-o")
            if (i < num_args) then
                i = i + 1
                call get_command_argument(i, output_file)
            else
                print *, "Error: -o requires an argument"
                stop 1
            end if
            
        case default
            print *, "Unknown option: ", trim(arg)
            stop 1
        end select
        
        i = i + 1
    end do
    
    ! For now, we just fail - this is the RED phase
    print *, "Error: HLFIR generation not implemented yet"
    print *, "Input file: ", trim(input_file)
    print *, "Output file: ", trim(output_file)
    print *, "Emit HLFIR: ", emit_hlfir
    
    stop 1  ! Always fail in RED phase
    
end program ffc_minimal