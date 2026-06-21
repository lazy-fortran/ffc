    subroutine lower_program_to_liric_exe(arena, root_index, output_path, &
                                          error_msg, include_paths)

        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: root_index
        character(len=*), intent(in) :: output_path
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=*), intent(in), optional :: include_paths(:)
        call lower_program_to_liric_path(arena, root_index, output_path, &
                                         .true., error_msg, include_paths)
    end subroutine lower_program_to_liric_exe
    subroutine lower_program_to_liric_object(arena, root_index, output_path, &
                                             error_msg, include_paths)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: root_index
        character(len=*), intent(in) :: output_path
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=*), intent(in), optional :: include_paths(:)
        call lower_program_to_liric_path(arena, root_index, output_path, &
                                         .false., error_msg, include_paths)
    end subroutine lower_program_to_liric_object
    subroutine lower_program_to_liric_path(arena, root_index, output_path, &
                                           emit_executable, error_msg, &
                                           include_paths)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: root_index
        character(len=*), intent(in) :: output_path
        character(len=*), intent(in), optional :: include_paths(:)
        logical, intent(in) :: emit_executable
        character(len=:), allocatable, intent(out) :: error_msg
        type(lowering_context_t) :: context
        type(lr_operand_desc_t) :: return_value
        integer(c_int32_t) :: argc_vreg
        integer(c_int32_t) :: argv_vreg
        call validate_program(arena, root_index, error_msg)
        if (len_trim(error_msg) > 0) return
        call liric_session_create(context%session, error_msg)
        if (len_trim(error_msg) > 0) return
        allocate(context%symbols(16))
        allocate(context%derived_types(8))
        allocate(context%module_exports(4))
        allocate(context%function_names(8))
        allocate(context%function_value_kinds(8))
        context%function_value_kinds = VALUE_I32
        context%arena = arena
        if (present(include_paths)) then
            context%include_paths = include_paths
            context%include_path_count = size(include_paths)
        else
            allocate (character(len=1) :: context%include_paths(0))
            context%include_path_count = 0
        end if
        if (.not. prepare_liric_print_runtime(context%session, &
                                              context%i32_print_format_id, &
                                              context%str_print_format_id, &
                                              error_msg, &
                                              context%i64_print_format_id, &
                                              context%i8_print_format_id, &
                                              context%i16_print_format_id)) then
            call destroy(context%session)
            return
        end if
        if (.not. synthesize_real8_printer(context%session, error_msg)) then
            call destroy(context%session)
            return
        end if
        if (.not. synthesize_real4_printer(context%session, error_msg)) then
            call destroy(context%session)
            return
        end if
        if (.not. synthesize_get_arg_helper(context%session, error_msg)) then
            call destroy(context%session)
            return
        end if
        if (.not. synthesize_complex4_printer(context%session, error_msg)) then
            call destroy(context%session)
            return
        end if
        if (.not. synthesize_complex8_printer(context%session, error_msg)) then
            call destroy(context%session)
            return
        end if
        call collect_program_parameters(arena, root_index, context, error_msg)
        if (len_trim(error_msg) > 0) then
            call destroy(context%session)
            return
        end if
        call collect_derived_type_definitions(arena, root_index, context, &
                                              error_msg)
        if (len_trim(error_msg) > 0) then
            call destroy(context%session)
            return
        end if
        call collect_module_exports(arena, root_index, context, error_msg)
        if (len_trim(error_msg) > 0) then
            call destroy(context%session)
            return
        end if
        call collect_internal_function_names(arena, root_index, context, &
                                             error_msg)
        if (len_trim(error_msg) > 0) then
            call destroy(context%session)
            return
        end if
        call collect_statement_functions(arena, root_index, context, error_msg)
        if (len_trim(error_msg) > 0) then
            call destroy(context%session)
            return
        end if
        ! Saved-local globals must exist before any function body is compiled:
        ! a global created mid-compile is direct-address patched and skips its
        ! data-section initializer, so emit them all up front (#1541).
        call emit_saved_local_globals(arena, context, error_msg)
        if (len_trim(error_msg) > 0) then
            call destroy(context%session)
            return
        end if
        ! COMMON-block globals share the same up-front emission requirement so
        ! BLOCK DATA initialisers reach the data section (#1578, #1900).
        call emit_common_block_globals(arena, context, error_msg)
        if (len_trim(error_msg) > 0) then
            call destroy(context%session)
            return
        end if
        ! EQUIVALENCE groups share storage; emit their globals up front for the
        ! same data-section reason as COMMON (#280, issue_1745).
        call emit_equivalence_globals(arena, context, error_msg)
        if (len_trim(error_msg) > 0) then
            call destroy(context%session)
            return
        end if
        call lower_internal_functions(arena, root_index, context, error_msg)
        if (len_trim(error_msg) > 0) then
            call destroy(context%session)
            return
        end if
        ! Globals must exist before module procedures so their bodies can bind
        ! module variables to the interned global address (#263).
        call lower_module_variables(arena, root_index, context, error_msg)
        if (len_trim(error_msg) > 0) then
            call destroy(context%session)
            return
        end if
        call lower_module_procedures(arena, root_index, context, error_msg)
        if (len_trim(error_msg) > 0) then
            call destroy(context%session)
            return
        end if
        if (.not. begin_i32_main(context%session, error_msg, argc_vreg, &
                                 argv_vreg)) then
            call destroy(context%session)
            return
        end if
        context%argc_value = i32_vreg(context%session, argc_vreg)
        context%argv_value = ptr_vreg(context%session, argv_vreg)
        context%has_command_args = .true.
        call lower_program_return(arena, root_index, context, return_value, &
                                  error_msg)
        if (len_trim(error_msg) > 0) then
            call destroy(context%session)
            return
        end if
        if (.not. context%current_block_terminated) then
            if (.not. emit_ret_i32_operand(context%session, return_value, &
                                                           error_msg)) then
                call destroy(context%session)
                return
            end if
        end if
        if (emit_executable) then
            if (.not. finish_and_emit_exe(context%session, output_path, &
                                                          error_msg)) then
                call destroy(context%session)
                return
            end if
        else
            if (.not. finish_and_emit_object(context%session, output_path, &
                                                             error_msg)) then
                call destroy(context%session)
                return
            end if
            ! Emit a sibling .fmod for each module compiled with -c.
            if (context%module_export_count > 0) then
                call emit_module_fmod_artifacts(arena, context, output_path, &
                                                error_msg)
                if (len_trim(error_msg) > 0) then
                    call destroy(context%session)
                    return
                end if
            end if
        end if
        call destroy(context%session)
        call set_empty(error_msg)
    end subroutine lower_program_to_liric_path
    subroutine validate_program(arena, root_index, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: root_index
        character(len=:), allocatable, intent(out) :: error_msg
        if (root_index <= 0) then
            error_msg = 'FortFront did not return a root program index'
            return
        end if
        if (.not. node_exists(arena, root_index)) then
            error_msg = 'FortFront root index does not reference an AST node'
            return
        end if
        if (get_node_type_at(arena, root_index) == 'program_node' .or. &
            get_node_type_at(arena, root_index) == 'program') then
            continue
        else if (get_node_type_at(arena, root_index) == &
                 'mixed_construct_container') then
            ! Lazy source mixing a top-level procedure with bare main-program
            ! statements parses to a mixed_construct_container. It lowers like a
            ! program: explicit procedures become contained functions, implicit
            ! statements become the main body (#266, #275). Constructs the body
            ! still rejects (derived types, modules) keep their own diagnostics.
            continue
        else if (get_node_type_at(arena, root_index) == 'module_node' .or. &
                 get_node_type_at(arena, root_index) == 'module') then
            ! A bare module unit is accepted only when it declares no symbols;
            ! lower_program_return walks the body and emits targeted diagnostics
            ! for module-defined procedures, types, and constants.
            continue
        else if (get_node_type_at(arena, root_index) == 'block_data_node' .or. &
                 get_node_type_at(arena, root_index) == 'block_data') then
            ! A standalone BLOCK DATA unit only initialises COMMON storage; with
            ! no program to consume it, it lowers to an empty main (#2809).
            continue
        else
            call unsupported_feature_error('program unit', &
                                           get_node_line(arena, root_index), &
                                           get_node_column(arena, root_index), &
                                           'direct LIRIC session only lowers '// &
                                           'top-level programs', error_msg)
            return
        end if
        call set_empty(error_msg)
    end subroutine validate_program
    subroutine lower_program_return(arena, root_index, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: root_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: mod_name, ff_error
        integer, allocatable :: body_indices(:), mod_declaration_indices(:), &
                                mod_procedure_indices(:)
        integer :: i
        integer :: j
        logical :: has_executable_statements
        logical :: has_nested_program
        logical :: data_flushed
        value = i32_immediate(context%session, 0_c_int64_t)
        context%current_block_terminated = .false.
        has_executable_statements = .false.
        has_nested_program = .false.
        call set_empty(error_msg)
        if (get_node_type_at(arena, root_index) == 'program_node' .or. &
            get_node_type_at(arena, root_index) == 'program' .or. &
            get_node_type_at(arena, root_index) == 'mixed_construct_container') then
            call get_unit_body_indices(arena, root_index, body_indices, ff_error)
            if (len_trim(ff_error) > 0) then
                error_msg = ff_error
                return
            end if
            if (.not. allocated(body_indices)) return
            ! Check if there's a program_node in the body (multi-unit source)
            has_nested_program = .false.
            do i = 1, size(body_indices)
                if (is_program_node(arena, body_indices(i))) then
                    has_nested_program = .true.
                    exit
                end if
            end do
            ! A body carrying statement labels needs GOTO-aware lowering: labels
            ! become LIRIC blocks and scalar locals move to memory so values
            ! survive arbitrary branch edges without manual phi placement (#270).
            if (body_has_statement_labels(arena, body_indices)) then
                call lower_labeled_program_body(arena, body_indices, &
                                                has_nested_program, context, &
                                                value, error_msg)
                return
            end if
            ! Process body elements. DATA statements are specification-class:
            ! they initialise variables before execution begins regardless of
            ! their textual position, so they are deferred and flushed once,
            ! ahead of the first executable statement (#2349/#2251/#2252).
            data_flushed = .false.
            do i = 1, size(body_indices)
                if (is_internal_procedure_entry(arena, body_indices(i))) cycle
                if (get_node_type_at(arena, body_indices(i)) == 'module_node' .or. &
                    get_node_type_at(arena, body_indices(i)) == 'module') then
                    ! Skip module nodes only if there's a nested program
                    if (.not. has_nested_program) then
                        call unsupported_feature_error('module definition', &
                            arena%entries(body_indices(i))%node%line, &
                            arena%entries(body_indices(i))%node%column, &
                            'direct LIRIC session only supports modules as USE '// &
                            'targets, not as top-level program units', error_msg)
                        return
                    end if
                    cycle
                end if
                ! FortFront expands each DATA statement into the data_statement
                ! node plus suppress_codegen assignments at the DATA's textual
                ! position. Both are skipped here; DATA values are applied once
                ! by flush_data_statements as program initialisation so source
                ! ordering against executable assignments matches gfortran.
                if (is_data_statement_at(arena, body_indices(i))) cycle
                if (is_suppressed_assignment_at(arena, body_indices(i))) cycle
                if (.not. data_flushed .and. &
                    .not. is_specification_statement_at(arena, body_indices(i))) then
                    call flush_data_statements(arena, body_indices, context, &
                                               error_msg)
                    if (len_trim(error_msg) > 0) return
                    data_flushed = .true.
                end if
                call lower_statement(arena, body_indices(i), context, &
                                     value, error_msg)
                if (len_trim(error_msg) > 0) return
                if (context%current_block_terminated) return
                has_executable_statements = .true.
            end do
            if (.not. data_flushed) then
                call flush_data_statements(arena, body_indices, context, error_msg)
                if (len_trim(error_msg) > 0) return
            end if
        else if (get_node_type_at(arena, root_index) == 'module_node' .or. &
                 get_node_type_at(arena, root_index) == 'module') then
            call get_module_body_info(arena, root_index, &
                                      mod_name, mod_declaration_indices, &
                                      mod_procedure_indices, ff_error)
            if (len_trim(ff_error) > 0) then
                error_msg = ff_error
                return
            end if
            call lower_module_unit(arena, mod_name, mod_declaration_indices, &
                                   mod_procedure_indices, error_msg)
        else if (get_node_type_at(arena, root_index) == 'block_data_node' .or. &
                 get_node_type_at(arena, root_index) == 'block_data') then
            ! A standalone BLOCK DATA unit emits no executable code; its COMMON
            ! initialisers were folded into the slot globals by the pre-pass and
            ! the empty main just returns 0 (#2809).
            call set_empty(error_msg)
        else
            error_msg = 'ffc direct-session lowering only supports a program node'
        end if
    end subroutine lower_program_return

    subroutine lower_module_unit(arena, mod_name, declaration_indices, &
                                 procedure_indices, error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: mod_name
        integer, intent(in) :: declaration_indices(:)
        integer, intent(in) :: procedure_indices(:)
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: node_type
        integer :: i, value_kind, status

        ! A module of compile-time exports (named constants and derived-type
        ! definitions) lowers to no runtime code: the constants are folded at
        ! use sites and the types are layout-only. Module procedures are emitted
        ! upstream by lower_module_procedures (#119, #263), and supported scalar
        ! module variables are given globals by lower_module_variables (#263).
        ! Here the walk diagnoses any module declaration that is neither a named
        ! constant nor a supported scalar variable (e.g. a module array), which
        ! still has no storage. An empty module is a no-op.
        do i = 1, size(declaration_indices)
            if (.not. node_exists(arena, declaration_indices(i))) cycle
            node_type = get_node_type_at(arena, declaration_indices(i))
            if (node_type == 'implicit_statement_node' .or. &
                node_type == 'implicit_statement' .or. &
                node_type == 'derived_type_node' .or. &
                node_type == 'derived_type') cycle
            if (node_type == 'declaration') then
                if (module_declaration_is_parameter(arena, &
                                                    declaration_indices(i))) cycle
                call module_variable_kind(arena, declaration_indices(i), &
                                          value_kind, status, error_msg)
                if (status /= MODVAR_OK) return
            end if
        end do
        call set_empty(error_msg)
    end subroutine lower_module_unit

    logical function module_declaration_is_parameter(arena, node_index)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index

        module_declaration_is_parameter = .false.
        if (.not. node_exists(arena, node_index)) return
        select type (decl => arena%entries(node_index)%node)
        type is (declaration_node)
            module_declaration_is_parameter = decl%is_parameter
        end select
    end function module_declaration_is_parameter
    include 'session_program_lowering_functions.inc'
    recursive subroutine lower_statement(arena, node_index, context, value, &
                                         error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: node_type
        logical :: enum_handled
        context%current_block_terminated = .false.
        value = i32_immediate(context%session, 0_c_int64_t)
        if (.not. node_exists(arena, node_index)) then
            error_msg = 'program body index does not reference an AST node'
            return
        end if
        if (is_subroutine_call_statement(arena, node_index)) then
            call lower_subroutine_call(arena, node_index, context, error_msg)
            return
        end if
        node_type = get_node_type_at(arena, node_index)
        if (node_type == 'module_node' .or. node_type == 'module') then
            call set_empty(error_msg)
            return
        end if
        if (node_type == 'program_node' .or. node_type == 'program') then
            call lower_program_body(arena, get_node_as_program(arena, node_index), &
                                    context, value, error_msg)
            return
        end if
        select type (node => arena%entries(node_index)%node)
        type is (parameter_declaration_node)
            call lower_parameter_declaration(node, context, error_msg)
        type is (declaration_node)
            call lower_declaration(node, node_index, context, error_msg)
        type is (assignment_node)
            call lower_assignment(arena, node, context, error_msg)
        type is (pointer_assignment_node)
            call lower_pointer_assignment(arena, node, context, error_msg)
        type is (nullify_node)
            call lower_nullify(arena, node, context, error_msg)
        type is (data_statement_node)
            call lower_data_statement(arena, node, context, error_msg)
        type is (open_statement_node)
            call lower_open(arena, node, context, error_msg)
        type is (close_statement_node)
            call lower_close(node, context, error_msg)
        type is (rewind_statement_node)
            call lower_rewind(node, context, error_msg)
        type is (print_statement_node)
            call lower_print(arena, node, context, error_msg)
        type is (read_statement_node)
            if (allocated(node%namelist_group)) then
                call lower_namelist_read(node, context, error_msg)
                return
            end if
            if (is_internal_read(node, context)) then
                call lower_internal_read(arena, node, context, error_msg)
                return
            end if
            if (is_stdin_read(node)) then
                call lower_read_stdin(arena, node, context, error_msg)
                return
            end if
            if (is_file_unit_read(node, context)) then
                call lower_read_file(arena, node, context, error_msg)
                return
            end if
            call unsupported_feature_error('read statement', node%line, &
                                           node%column, &
                                           'direct LIRIC session does not '// &
                                           'support this read form yet', &
                                           error_msg)
        type is (write_statement_node)
            if (allocated(node%namelist_group)) then
                call lower_namelist_write(node, context, error_msg)
            else if (is_internal_write(node, context)) then
                call lower_internal_write(arena, node, context, error_msg)
            else if (is_stdout_write(node)) then
                call lower_write_stdout(arena, node, context, error_msg)
            else if (is_file_unit_write(node, context)) then
                call lower_write_file(arena, node, context, error_msg)
            else
                call unsupported_feature_error('write statement', node%line, &
                                               node%column, &
                                               'direct LIRIC session only '// &
                                               'supports minimal print output, '// &
                                               'internal write, and file write '// &
                                               '(#247)', error_msg)
            end if
        type is (allocate_statement_node)
            ! allocate(<type> :: x) on a class(*) allocatable populates the
            ! polymorphic descriptor; everything else uses the array path.
            if (allocate_targets_class_star(arena, node, context)) then
                call lower_class_star_allocate(arena, node, context, error_msg)
            else
                call lower_allocate_statement(arena, node, context, error_msg)
            end if
        type is (deallocate_statement_node)
            call lower_deallocate_statement(arena, node, context, error_msg)
        type is (return_node)
            if (context%in_internal_subroutine) then
                if (.not. emit_ret_void(context%session, error_msg)) return
                context%current_block_terminated = .true.
            else if (context%in_internal_function) then
                call lower_function_return(node, context, error_msg)
            else
                call unsupported_feature_error('return statement', node%line, &
                                               node%column, &
                                               'direct LIRIC session does not '// &
                                               'support return at top level', &
                                               error_msg)
            end if
        type is (stop_node)
            call lower_stop(arena, node, context, value, error_msg)
            if (len_trim(error_msg) == 0) then
                call emit_stop_banner(node, value, context, error_msg)
                if (len_trim(error_msg) > 0) return
                if (context%in_internal_function .or. &
                    context%in_internal_subroutine) then
                    ! stop terminates the whole program. From a contained
                    ! procedure, call libc exit() rather than returning the
                    ! code (which would only return from the procedure).
                    if (.not. emit_void_call(context%session, 'exit', [value], &
                                             error_msg)) return
                    if (context%in_internal_subroutine) then
                        if (.not. emit_ret_void(context%session, error_msg)) return
                    else
                        if (.not. emit_ret_i32_operand(context%session, value, &
                                                       error_msg)) return
                    end if
                else
                    if (.not. emit_ret_i32_operand(context%session, value, &
                                                                   error_msg)) return
                end if
                context%current_block_terminated = .true.
            end if
        type is (error_stop_node)
            call lower_error_stop(arena, node, context, value, error_msg)
            if (len_trim(error_msg) == 0) then
                call emit_error_stop_banner(node, value, context, error_msg)
                if (len_trim(error_msg) > 0) return
                ! ERROR STOP always terminates the whole program with the error
                ! code (default 1). From a contained procedure use libc exit();
                ! at top level return the code.
                if (context%in_internal_function .or. &
                    context%in_internal_subroutine) then
                    if (.not. emit_void_call(context%session, 'exit', [value], &
                                             error_msg)) return
                    if (context%in_internal_subroutine) then
                        if (.not. emit_ret_void(context%session, error_msg)) return
                    else
                        if (.not. emit_ret_i32_operand(context%session, value, &
                                                       error_msg)) return
                    end if
                else
                    if (.not. emit_ret_i32_operand(context%session, value, &
                                                   error_msg)) return
                end if
                context%current_block_terminated = .true.
            end if
        type is (exit_node)
            if (.not. context%in_counted_do) then
                call unsupported_feature_error('exit statement', node%line, &
                                               node%column, &
                                               'direct LIRIC session only '// &
                                               'supports exit inside a counted '// &
                                               'do', error_msg)
                return
            end if
            if (.not. emit_liric_br(context%session, &
                                    context%current_loop_exit_block, &
                                    error_msg)) return
            context%current_block_terminated = .true.
            context%current_block_exited_loop = .true.
        type is (cycle_node)
            if (.not. context%in_counted_do) then
                call unsupported_feature_error('cycle statement', node%line, &
                                               node%column, &
                                               'direct LIRIC session only '// &
                                               'supports cycle inside a counted '// &
                                               'do', error_msg)
                return
            end if
            if (.not. emit_liric_br(context%session, &
                                    context%current_loop_latch_block, &
                                    error_msg)) return
            context%current_block_terminated = .true.
            context%current_block_exited_loop = .true.
        type is (if_node)
            call lower_if(arena, node, context, value, error_msg)
        type is (do_loop_node)
            if (node%is_concurrent) then
                call lower_do_concurrent(arena, node, context, value, error_msg)
            else
                call lower_do_loop(arena, node, context, value, error_msg)
            end if
        type is (do_while_node)
            call lower_do_while(arena, node, context, value, error_msg)
        type is (select_case_node)
            call lower_select_case(arena, node, context, error_msg)
        type is (select_type_node)
            call lower_select_type(arena, node_index, context, error_msg)
        type is (select_rank_node)
            call lower_select_rank(arena, node, context, error_msg)
        type is (associate_node)
            call lower_associate(arena, node, context, error_msg)
        type is (block_construct_node)
            call lower_block_construct(arena, node, context, error_msg)
        type is (where_node)
            call lower_where_construct(arena, node, context, error_msg)
        type is (where_stmt_node)
            call lower_where_stmt(arena, node, context, error_msg)
        type is (forall_node)
            call lower_forall(arena, node, context, error_msg)
        type is (derived_type_node)
            call lower_derived_type_definition(node, context, error_msg)
        type is (use_statement_node)
            call import_module_derived_types(arena, node, context, error_msg)
        type is (interface_block_node)
            call lower_interface_block(arena, node, context, error_msg)
        type is (pause_node)
            ! PAUSE writes a banner to stderr and waits on stdin; end-of-input
            ! terminates the job, matching gfortran (#280).
            call lower_pause(node, context, error_msg)
        type is (continue_node)
            ! CONTINUE is a no-op; it commonly only anchors a statement label
            ! that a GOTO targets (the labeled-body driver owns the branch).
            call set_empty(error_msg)
        type is (goto_node)
            call lower_goto(arena, node_index, context, error_msg)
        type is (common_block_node)
            ! Reaching the COMMON statement during the body walk rebinds its
            ! variables to their shared slot globals (#1578, #2809).
            call bind_common_block_symbols(arena, context, error_msg)
        type is (comment_node)
            ! An EQUIVALENCE statement still survives as a comment; reaching it
            ! rebinds its members onto the shared group global (#280). Any other
            ! comment carries no executable semantics.
            if (is_equivalence_text(node%text)) then
                call bind_equivalence_symbols(arena, context, error_msg)
            else
                call set_empty(error_msg)
            end if
        type is (enum_node)
            ! An ENUM definition binds named integer constants; it emits no code
            ! at its position. The enumerators are bound here so later
            ! expressions resolve them (#1826).
            call lower_enum_block(arena, node_index, context, error_msg, &
                                  enum_handled)
        type is (block_data_node)
            ! A BLOCK DATA unit only initialises COMMON storage; its values were
            ! folded into the slot globals by the pre-pass, so it emits no code.
            call set_empty(error_msg)
        type is (namelist_statement_node)
            ! A NAMELIST declaration emits no code; it records the group's
            ! member list so a later WRITE(unit, nml=group) can format them.
            call record_namelist_group(node, context, error_msg)
        type is (statement_function_node)
            ! A statement-function definition emits no code on its own; it was
            ! recorded at setup and is inlined at each call site.
            call set_empty(error_msg)
        type is (end_statement_node)
            ! The implicit-program terminator carries no executable semantics.
            call set_empty(error_msg)
        class default
            node_type = get_node_type_at(arena, node_index)
            if (len_trim(node_type) == 0) node_type = 'unknown'
            call unsupported_statement_node(arena, node_index, node_type, error_msg)
        end select
    end subroutine lower_statement
    subroutine unsupported_statement_node(arena, node_index, node_type, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=*), intent(in) :: node_type
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: column
        integer :: line
        line = get_node_line(arena, node_index)
        column = get_node_column(arena, node_index)
        select case (trim(node_type))
        case ('implicit_statement')
            call set_empty(error_msg)
        case ('comment')
            ! Comments carry no executable semantics; skip them.
            call set_empty(error_msg)
        case ('continue_node')
            ! CONTINUE is a no-op statement (often only a loop label target).
            call set_empty(error_msg)
        case ('directive')
            ! OpenMP/OpenACC directives are conditional-compilation comments.
            ! The gfortran reference is built without -fopenmp/-fopenacc, so it
            ! also ignores them; lowering them as no-ops keeps output matching.
            call set_empty(error_msg)
        case ('intrinsic_statement')
            ! INTRINSIC only marks names as intrinsic procedures; no code.
            call set_empty(error_msg)
        case ('include_statement')
            ! FortFront preserves INCLUDE lines in the AST without textual
            ! expansion; there is nothing to lower at this stage.
            call set_empty(error_msg)
        case ('use_statement')
            call unsupported_feature_error('use statement', line, column, &
                                           'direct LIRIC session does not '// &
                                           'support module imports', error_msg)
        case ('interface_block')
            call unsupported_feature_error('interface block', line, column, &
                                           'direct LIRIC session does not '// &
                                           'support generic or explicit '// &
                                           'interfaces', error_msg)
        case default
            call unsupported_feature_error('statement node: '//trim(node_type), &
                                           line, column, &
                                           'direct LIRIC session does not '// &
                                           'support this AST node', error_msg)
        end select
    end subroutine unsupported_statement_node
    include 'session_program_lowering_control.inc'
    include 'session_program_lowering_goto.inc'
    include 'session_program_lowering_associate.inc'
    include 'session_program_lowering_loops.inc'
    include 'session_program_lowering_do_while.inc'
    include 'session_program_lowering_interface.inc'
    include 'session_program_lowering_derived_types.inc'
    include 'session_program_lowering_derived_type_ops.inc'
    include 'session_program_lowering_derived_module_ops.inc'
    include 'session_program_lowering_module_vars.inc'
    include 'session_program_lowering_save.inc'
    include 'session_program_lowering_common.inc'
    include 'session_program_lowering_equivalence.inc'
    include 'session_program_lowering_block_concurrent.inc'
    include 'session_program_lowering_where.inc'
    include 'session_program_lowering_forall.inc'
    include 'session_program_lowering_enum.inc'
    include 'session_program_lowering_namelist.inc'
    subroutine lower_statement_list(arena, node_indices, context, value, &
                                    terminated, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_indices(:)
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        logical, intent(out) :: terminated
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i
        terminated = .false.
        value = i32_immediate(context%session, 0_c_int64_t)
        call set_empty(error_msg)
        do i = 1, size(node_indices)
            ! A nested procedure definition (an internal procedure of the unit
            ! whose body this is) is lowered separately, not as an in-body
            ! statement. Skip it here so a procedure that contains internals is
            ! lowered the same way a program-contained procedure is.
            if (is_internal_procedure_entry(arena, node_indices(i))) cycle
            call lower_statement(arena, node_indices(i), context, value, error_msg)
            if (len_trim(error_msg) > 0) return
            if (context%current_block_terminated) then
                terminated = .true.
                return
            end if
        end do
    end subroutine lower_statement_list
