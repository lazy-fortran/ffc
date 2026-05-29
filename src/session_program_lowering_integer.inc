  recursive subroutine lower_i32_expression(arena, node_index, context, &
                                            value, error_msg)
      use liric_session_bindings, only: LR_OP_ADD
      type(ast_arena_t), intent(in) :: arena
      integer, intent(in) :: node_index
      type(lowering_context_t), intent(inout) :: context
      type(lr_operand_desc_t), intent(out) :: value
      character(len=:), allocatable, intent(out) :: error_msg
      integer(c_int64_t) :: literal_value
      type(lr_operand_desc_t) :: lhs
      type(lr_operand_desc_t) :: rhs
      integer :: opcode
      integer :: symbol_index

      if (.not. node_exists(arena, node_index)) then
          error_msg = 'expression index does not reference an AST node'
          return
      end if

      select type (node => arena%entries(node_index)%node)
      type is (literal_node)
          call parse_i32_literal(node%value, literal_value, error_msg)
          if (len_trim(error_msg) > 0) return
          value = i32_immediate(context%session, literal_value)
      type is (identifier_node)
          symbol_index = find_symbol(context, node%name)
          if (symbol_index <= 0) then
              error_msg = 'integer identifier was not declared: '//trim(node%name)
              return
          end if
          if (context%symbols(symbol_index)%is_array) then
              call unsupported_feature_error('array expression', &
                                             node%line, node%column, &
                                             'array reads require exactly one '// &
                                             'subscript', error_msg)
              return
          end if
          if (context%symbols(symbol_index)%value_kind /= VALUE_I32) then
              error_msg = 'integer expression used non-integer identifier: '// &
                          trim(node%name)
              return
          end if
          if (context%symbols(symbol_index)%has_address .and. &
              context%symbols(symbol_index)%is_reference) then
              if (.not. emit_i32_load(context%session,  &
                  context%symbols(symbol_index)%address, value, &
                  error_msg)) return
          else
              value = context%symbols(symbol_index)%value
              call set_empty(error_msg)
          end if
      type is (binary_op_node)
          call lower_i32_expression(arena, node%left_index, context, lhs, &
                                    error_msg)
          if (len_trim(error_msg) > 0) return
          if (trim(node%operator) == '**') then
              call lower_i32_pow(arena, node%right_index, node%line, &
                                 node%column, context, lhs, value, error_msg)
              return
          end if
          call lower_i32_expression(arena, node%right_index, context, rhs, &
                                    error_msg)
          if (len_trim(error_msg) > 0) return
          call integer_opcode(node%operator, opcode, error_msg)
          if (len_trim(error_msg) > 0) then
              call unsupported_feature_error('integer operator', &
                                             node%line, node%column, &
                                             'direct LIRIC session supports '// &
                                             '+, -, *, and / for integer '// &
                                             'expressions', error_msg)
              return
          end if
          if (.not. emit_i32_binary(context%session, opcode, lhs, rhs, value, &
                                                    error_msg)) return
      type is (call_or_subscript_node)
          if (node%is_array_access) then
              call lower_i32_array_element(arena, node, context, value, error_msg)
          else
              call lower_i32_call(arena, node, context, value, error_msg)
          end if
      type is (component_access_node)
          call lower_derived_component_load(arena, node, context, value, error_msg)
      type is (array_slice_node)
          call unsupported_array_subscript(arena, node_index, &
                                           'ranges and slices', error_msg)
      class default
          error_msg = 'ffc direct-session lowering only supports integer expressions'
      end select
  end subroutine lower_i32_expression

  subroutine lower_i32_pow(arena, exp_index, line, column, context, base, &
                           value, error_msg)
      use liric_session_bindings, only: LR_OP_MUL
      type(ast_arena_t), intent(in) :: arena
      integer, intent(in) :: exp_index
      integer, intent(in) :: line
      integer, intent(in) :: column
      type(lowering_context_t), intent(inout) :: context
      type(lr_operand_desc_t), intent(in) :: base
      type(lr_operand_desc_t), intent(out) :: value
      character(len=:), allocatable, intent(out) :: error_msg
      integer(c_int64_t) :: exponent
      character(len=:), allocatable :: parse_error
      type(lr_operand_desc_t) :: product
      integer(c_int64_t) :: i

      call parse_i32_constant(arena, exp_index, exponent, 'exponent', &
                              parse_error)
      if (len_trim(parse_error) > 0) then
          call unsupported_feature_error('integer exponent', line, column, &
                                         'direct LIRIC session only supports '// &
                                         'a non-negative integer literal '// &
                                         'exponent', error_msg)
          return
      end if
      if (exponent < 0_c_int64_t) then
          call unsupported_feature_error('negative integer exponent', line, &
                                         column, &
                                         'direct LIRIC session only supports '// &
                                         'a non-negative integer literal '// &
                                         'exponent', error_msg)
          return
      end if

      if (exponent == 0_c_int64_t) then
          value = i32_immediate(context%session, 1_c_int64_t)
          call set_empty(error_msg)
          return
      end if

      value = base
      do i = 2_c_int64_t, exponent
          if (.not. emit_i32_binary(context%session, LR_OP_MUL, value, base, &
                                    product, error_msg)) return
          value = product
      end do
      call set_empty(error_msg)
  end subroutine lower_i32_pow

  subroutine lower_i32_call(arena, node, context, value, error_msg)
      type(ast_arena_t), intent(in) :: arena
      type(call_or_subscript_node), intent(in) :: node
      type(lowering_context_t), intent(inout) :: context
      type(lr_operand_desc_t), intent(out) :: value
      character(len=:), allocatable, intent(out) :: error_msg
      type(lr_operand_desc_t), allocatable :: args(:)
      integer, allocatable :: copyback_indices(:)
      integer :: intrinsic_id

      if (node%is_array_access) then
          call unsupported_feature_error('array expression', &
                                         node%line, node%column, &
                                         'direct LIRIC session does not '// &
                                         'support reading array elements', &
                                         error_msg)
          return
      end if
      if (.not. allocated(node%name)) then
          error_msg = 'direct LIRIC session function call requires a name'
          return
      end if

      intrinsic_id = i32_intrinsic_id(node%name)
      if (.not. is_contained_i32_function(context, node%name)) then
          if (intrinsic_id /= I32_INTRINSIC_NONE) then
              call lower_i32_intrinsic_call(arena, node, intrinsic_id, &
                                            context, value, error_msg)
              return
          end if
          if (node%is_intrinsic) then
              call unsupported_intrinsic_error(node, error_msg)
              return
          end if
          call unsupported_feature_error( &
              'scalar function call or array expression', &
              node%line, node%column, &
              'direct LIRIC session only supports contained scalar '// &
              'functions, supported intrinsics, and scalar variables', &
              error_msg)
          return
      end if

      if (allocated(node%arg_indices)) then
          call prepare_reference_args(arena, node%arg_indices, context, &
                                      VALUE_I32, args, copyback_indices, &
                                      error_msg)
          if (len_trim(error_msg) > 0) return
      else
          allocate (args(0))
          allocate (copyback_indices(0))
      end if

      if (.not. emit_i32_call(context%session, node%name, args, value, &
                                              error_msg)) return
      call copy_back_reference_args(context, args, copyback_indices, error_msg)
  end subroutine lower_i32_call

  subroutine lower_i32_array_element(arena, node, context, value, error_msg)
      type(ast_arena_t), intent(in) :: arena
      type(call_or_subscript_node), intent(in) :: node
      type(lowering_context_t), intent(inout) :: context
      type(lr_operand_desc_t), intent(out) :: value
      character(len=:), allocatable, intent(out) :: error_msg
      type(lr_operand_desc_t) :: element_address

      call lower_i32_array_element_address(arena, node, context, element_address, &
                                           error_msg)
      if (len_trim(error_msg) > 0) return

      if (.not. emit_i32_load(context%session, element_address, value, error_msg)) then
          error_msg = 'arr_read_load: '//error_msg
          return
      end if
      call set_empty(error_msg)
  end subroutine lower_i32_array_element
