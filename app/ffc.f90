program ffc_main
    use fortfront_compiler, only: compiler_frontend_options_t, &
                                  compiler_frontend_result_t, &
                                  compile_frontend_from_file, INPUT_MODE_LAZY, &
                                  INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe, &
                                        lower_program_to_liric_object
    use ffc_cli_options, only: cli_options_t, parse_arguments, CLI_PATH_LEN
    implicit none

    type(compiler_frontend_options_t) :: frontend_options
    type(compiler_frontend_result_t) :: frontend_result
    type(cli_options_t) :: opts
    character(len=CLI_PATH_LEN), allocatable :: argv(:)
    character(len=:), allocatable :: error_msg
    character(len=:), allocatable :: diag_msg
    character(len=:), allocatable :: primary_diag
    character(len=CLI_PATH_LEN) :: output_file
    character(len=CLI_PATH_LEN), allocatable :: search_paths(:)
    integer :: nargs, i

    nargs = command_argument_count()
    if (nargs < 1) then
        call print_usage()
        stop 1
    end if

    allocate (argv(nargs))
    do i = 1, nargs
        call get_command_argument(i, argv(i))
    end do
    call parse_arguments(argv, opts)
    if (opts%error) then
        print '(A)', opts%error_message
        stop 1
    end if

    output_file = opts%output_file
    if (len_trim(output_file) == 0) output_file = default_output_name(opts%emit_object)

    ! A .fmod sits beside the .o it was emitted with, so a linked object's
    ! directory is also searched when resolving `use`.
    search_paths = opts%include_paths
    do i = 1, size(opts%link_inputs)
        call append_path(search_paths, dirname(trim(opts%link_inputs(i))))
    end do

    ! A .lf file is lazy Fortran by definition: compile it under lazy inference,
    ! which standardizes the AST so inferred declarations (including function
    ! result variables) gain explicit types before lowering. Routing by
    ! extension is robust: it does not depend on standard-mode semantics
    ! rejecting the bare lazy fragment to trigger a retry.
    !
    ! For standard sources, lower from the declared AST first. This preserves the
    ! diagnostics for genuinely unsupported constructs and compiles every
    ! declared program. Only when the failure is an undeclared-variable signal (a
    ! bare lazy fragment in a non-.lf file) retry under lazy inference. The
    ! standard diagnostic is reported on total failure, never the lazy retry's.
    if (is_lazy_source(opts%input_file)) then
        if (.not. try_compile(INPUT_MODE_LAZY, .true.)) &
            call report_failure(diag_msg)
    else if (.not. try_compile(INPUT_MODE_STANDARD, .false.)) then
        primary_diag = diag_msg
        if (needs_inference(primary_diag)) then
            if (.not. try_compile(INPUT_MODE_LAZY, .true.)) &
                call report_failure(primary_diag)
        else
            call report_failure(primary_diag)
        end if
    end if

contains

    ! Compile and lower under one dialect config. Returns .true. on success.
    ! On failure leaves diag_msg set to the frontend or lowering error.
    logical function try_compile(input_mode, standardize) result(ok)
        integer, intent(in) :: input_mode
        logical, intent(in) :: standardize

        ok = .false.
        frontend_options = compiler_frontend_options_t()
        frontend_options%run_semantics = .true.
        frontend_options%input_mode = input_mode
        frontend_options%standardize = standardize

        call compile_frontend_from_file(trim(opts%input_file), frontend_result, &
                                        frontend_options)
        if (.not. frontend_result%success()) then
            diag_msg = trim(frontend_result%diagnostic_text)
            return
        end if

        error_msg = ''
        if (opts%emit_object) then
            call lower_program_to_liric_object(frontend_result%arena, &
                                               frontend_result%root_index, &
                                               trim(output_file), error_msg, &
                                               search_paths)
        else
            call lower_program_to_liric_exe(frontend_result%arena, &
                                            frontend_result%root_index, &
                                            trim(output_file), error_msg, &
                                            search_paths)
        end if
        if (len_trim(error_msg) > 0) then
            diag_msg = trim(error_msg)
            return
        end if
        ok = .true.
    end function try_compile

    subroutine report_failure(message)
        character(len=*), intent(in) :: message
        print '(A)', trim(message)
        stop 1
    end subroutine report_failure

    ! True when a standard-mode failure is an undeclared-variable signal, i.e.
    ! the source is a lazy fragment whose variables fortfront can infer. Genuine
    ! unsupported-construct diagnostics do not match, so they are never masked by
    ! the lazy retry.
    logical function needs_inference(diag) result(infer)
        character(len=*), intent(in) :: diag

        infer = index(diag, 'not declared') > 0 .or. &
                index(diag, 'not been declared') > 0 .or. &
                index(diag, 'implicit none') > 0
    end function needs_inference

    logical function is_lazy_source(path) result(lazy)
        ! A .lf file is lazy Fortran and must be standardized via lazy inference.
        character(len=*), intent(in) :: path
        integer :: n

        n = len_trim(path)
        lazy = n >= 3
        if (lazy) lazy = path(n - 2:n) == '.lf'
    end function is_lazy_source

    subroutine print_usage()
        print '(A)', 'Usage: ffc <input.f90> [options]'
        print '(A)', 'Options:'
        print '(A)', '  -o <file>     Output file'
        print '(A)', '  -c            Emit object file'
        print '(A)', '  -I <dir>      Add module/include search directory'
        print '(A)', '  <file>.o      Link input object (its directory is '// &
            'searched for .fmod)'
    end subroutine print_usage

    function default_output_name(emit_object) result(name)
        logical, intent(in) :: emit_object
        character(len=CLI_PATH_LEN) :: name

        if (emit_object) then
            name = 'a.o'
        else
            name = 'a.out'
        end if
    end function default_output_name

    function dirname(path) result(dir)
        character(len=*), intent(in) :: path
        character(len=:), allocatable :: dir
        integer :: slash

        slash = index(path, '/', back=.true.)
        if (slash > 0) then
            dir = path(1:slash - 1)
        else
            dir = '.'
        end if
    end function dirname

    subroutine append_path(paths, dir)
        character(len=CLI_PATH_LEN), allocatable, intent(inout) :: paths(:)
        character(len=*), intent(in) :: dir
        character(len=CLI_PATH_LEN), allocatable :: tmp(:)
        integer :: m

        if (.not. allocated(paths)) allocate (paths(0))
        m = size(paths)
        allocate (tmp(m + 1))
        if (m > 0) tmp(1:m) = paths(1:m)
        tmp(m + 1) = dir
        call move_alloc(tmp, paths)
    end subroutine append_path

end program ffc_main
