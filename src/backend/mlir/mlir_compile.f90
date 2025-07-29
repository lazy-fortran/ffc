module mlir_compile
    use backend_interface
    ! TODO: Replace with stdlib equivalents
    ! use temp_utils
    ! use system_utils, only: sys_run_command, sys_file_exists
    implicit none
    private

    public :: compile_mlir_to_output, apply_mlir_lowering_passes

contains

    subroutine compile_mlir_to_output(mlir_code, options, success, error_msg)
        character(len=*), intent(in) :: mlir_code
        type(backend_options_t), intent(in) :: options
        logical, intent(out) :: success
        character(len=*), intent(out) :: error_msg

        ! TODO: Implement MLIR compilation using stdlib utilities
        success = .false.
        error_msg = "MLIR compilation not yet implemented with stdlib utilities"

        ! Write MLIR to temporary file
        mlir_file = temp_mgr%get_file_path('temp.mlir')
        open (newunit=unit, file=mlir_file, action='write', status='replace')
        write (unit, '(A)') mlir_code
        close (unit)

        ! Debug: Save a copy for debugging
        print *, "DEBUG: Writing MLIR to debug_mlir.txt"
        print *, "DEBUG: MLIR length:", len_trim(mlir_code)
        open (newunit=unit, file='debug_mlir.txt', action='write', status='replace')
        write (unit, '(A)') mlir_code
        close (unit)

        ! Debug: Print MLIR content
        if (options%debug_info) then
            print *, "Generated MLIR:"
            print *, trim(mlir_code)
        end if

        ! Convert HLFIR to LLVM IR using flang toolchain
        llvm_file = temp_mgr%get_file_path('temp.ll')
        command = 'tco-19 '//mlir_file//' -o '//llvm_file
        if (options%debug_info) then
            print *, "Running tco command:", trim(command)
        end if
        call sys_run_command(command, cmd_output, exit_code)

        if (exit_code /= 0) then
            error_msg = "Failed to convert HLFIR to LLVM IR using tco: "//trim(cmd_output)
            return
        end if

        ! Check if we're building an executable or just object file
        is_executable = options%generate_executable .or. &
                        (index(options%output_file, '.o') == 0 .and. &
                         index(options%output_file, '.obj') == 0)

        if (is_executable) then
            ! Compile LLVM IR to object file first
            obj_file = temp_mgr%get_file_path('temp.o')
            command = 'llc -filetype=obj '//llvm_file//' -o '//obj_file

            if (options%optimize) then
                command = 'llc -O3 -filetype=obj '//llvm_file//' -o '//obj_file
            end if

            if (options%debug_info) then
                print *, "Running llc command:", trim(command)
            end if
            call sys_run_command(command, cmd_output, exit_code)

            if (exit_code /= 0) then
              error_msg = "Failed to compile LLVM IR to object code: "//trim(cmd_output)
                return
            end if

            ! Link to executable
            if (options%link_runtime) then
                ! Link with Fortran runtime
                command = 'gfortran '//obj_file//' -o '//options%output_file
            else
                ! Basic linking with clang
                command = 'clang -no-pie '//obj_file//' -o '//options%output_file
            end if

            if (options%debug_info) then
                print *, "Running link command:", trim(command)
            end if
            call sys_run_command(command, cmd_output, exit_code)

            if (exit_code /= 0) then
                error_msg = "Failed to link executable: "//trim(cmd_output)
                return
            end if
        else
            ! Just compile to object file
            command = 'llc -filetype=obj '//llvm_file//' -o '//options%output_file

            if (options%optimize) then
              command = 'llc -O3 -filetype=obj '//llvm_file//' -o '//options%output_file
            end if

            call sys_run_command(command, cmd_output, exit_code)

            if (exit_code /= 0) then
              error_msg = "Failed to compile LLVM IR to object code: "//trim(cmd_output)
                return
            end if
        end if

        ! Verify output file exists
        if (sys_file_exists(options%output_file)) then
            success = .true.
        else
            error_msg = "Output file was not created"
        end if

    end subroutine compile_mlir_to_output

    subroutine apply_mlir_lowering_passes(mlir_code, target_level, output, success, error_msg)
        character(len=*), intent(in) :: mlir_code
        character(len=*), intent(in) :: target_level  ! "fir" or "llvm"
        character(len=:), allocatable, intent(out) :: output
        logical, intent(out) :: success
        character(len=*), intent(out) :: error_msg

        type(temp_dir_manager) :: temp_mgr
        character(len=:), allocatable :: mlir_file, output_file, llvm_mlir_file
        character(len=:), allocatable :: command
        character(len=1024) :: cmd_output
        integer :: unit, exit_code, file_size
        logical :: file_exists

        success = .false.
        error_msg = ""
        output = ""

        ! Create temporary directory
        call temp_mgr%create('mlir_lowering')

        ! Write MLIR to temporary file
        mlir_file = temp_mgr%get_file_path('input.mlir')
        open (newunit=unit, file=mlir_file, action='write', status='replace')
        write (unit, '(A)') mlir_code
        close (unit)

        if (target_level == "fir") then
            ! Apply HLFIR to FIR lowering using flang-opt
            output_file = temp_mgr%get_file_path('output_fir.mlir')
            command = 'flang-opt-19 --hlfir-lowering '//mlir_file//' -o '//output_file
        else if (target_level == "llvm") then
            ! Apply FIR to LLVM IR lowering using tco
            output_file = temp_mgr%get_file_path('output.ll')
            command = 'tco-19 '//mlir_file//' -o '//output_file
        else
            error_msg = "Unknown target level: "//target_level
            return
        end if

        ! Execute the command
        call sys_run_command(command, cmd_output, exit_code)
        if (exit_code /= 0) then
            error_msg = "Lowering command failed: "//trim(cmd_output)
            return
        end if

        ! Read the output file
        inquire (file=output_file, exist=file_exists, size=file_size)
        if (.not. file_exists .or. file_size == 0) then
            error_msg = "Output file not created or empty"
            return
        end if

        ! Read the file content
        block
            character(len=file_size) :: file_content
            open (newunit=unit, file=output_file, action='read', status='old', access='stream', form='unformatted')
            read (unit) file_content
            close (unit)
            output = trim(file_content)
        end block

        success = .true.
    end subroutine apply_mlir_lowering_passes

end module mlir_compile
