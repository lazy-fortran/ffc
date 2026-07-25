program test_session_symbol_table
    !! Unit tests for the binding-keyed symbol table.
    !!
    !! Why these exist as unit tests rather than compiler tests: review of #327
    !! showed that neutralizing resolve_symbol_at_node back to a plain text
    !! find_symbol left all 288 tests green, and a divergence probe over 1000
    !! corpus files found zero cases where binding-keyed lookup resolved
    !! differently from text lookup. The table is foundational - it exists so the
    !! lowering fragments can eventually hold private state - and its value shows
    !! up as the remaining scope issues (#329 through #332) migrate call sites to
    !! it. Until then the corpus cannot observe it, so the table's own semantics
    !! are what must be pinned.
    !!
    !! Each test below fails if the specific behaviour it names is removed. That
    !! is the property the corpus gate could not provide.
    use session_symbol_table, only: session_symbol_table_t
    implicit none

    integer :: n_pass, n_fail

    n_pass = 0
    n_fail = 0

    call require_lookup_misses_on_empty_table()
    call require_insert_then_find_round_trips()
    call require_same_declaration_different_scope_are_distinct()
    call require_same_scope_different_entity_are_distinct()
    call require_reused_slot_drops_the_stale_identity()
    call require_repeated_insert_is_idempotent()

    write (*, '(a,i0,a,i0,a)') 'session_symbol_table: ', n_pass, ' pass, ', &
        n_fail, ' fail'
    if (n_fail > 0) error stop 1

contains

    subroutine assert(cond, msg)
        logical, intent(in) :: cond
        character(len=*), intent(in) :: msg

        if (cond) then
            n_pass = n_pass + 1
        else
            n_fail = n_fail + 1
            write (*, '(a,a)') 'FAIL: ', msg
        end if
    end subroutine assert

    subroutine require_lookup_misses_on_empty_table()
        type(session_symbol_table_t) :: table

        call assert(table%find_binding(1, 1, 1) == 0, &
            'lookup on an empty table reports no symbol')
    end subroutine require_lookup_misses_on_empty_table

    subroutine require_insert_then_find_round_trips()
        type(session_symbol_table_t) :: table
        integer :: found

        call table%insert_binding(10, 1, 100, 7)
        found = table%find_binding(10, 1, 100)
        call assert(found == 7, &
            'a binding inserted for a declaration is found again')
    end subroutine require_insert_then_find_round_trips

    subroutine require_same_declaration_different_scope_are_distinct()
        !! The whole point of keying by binding rather than by text: one
        !! declaration reached from two scopes is two entities.
        type(session_symbol_table_t) :: table

        call table%insert_binding(10, 1, 100, 7)
        call table%insert_binding(10, 1, 200, 9)
        call assert(table%find_binding(10, 1, 100) == 7, &
            'the first scope keeps its own symbol')
        call assert(table%find_binding(10, 1, 200) == 9, &
            'the second scope resolves to a different symbol')
    end subroutine require_same_declaration_different_scope_are_distinct

    subroutine require_same_scope_different_entity_are_distinct()
        !! `integer :: a, b, c` is one declaration node carrying three entities,
        !! so the entity index has to participate in identity.
        type(session_symbol_table_t) :: table

        call table%insert_binding(10, 1, 100, 7)
        call table%insert_binding(10, 2, 100, 8)
        call table%insert_binding(10, 3, 100, 9)
        call assert(table%find_binding(10, 1, 100) == 7 .and. &
            table%find_binding(10, 2, 100) == 8, &
            'entities of one multi-name declaration stay distinct')
        call assert(table%find_binding(10, 3, 100) == 9, &
            'the third entity resolves to its own symbol')
    end subroutine require_same_scope_different_entity_are_distinct

    subroutine require_reused_slot_drops_the_stale_identity()
        !! Symbol slots are reused as the lowering context grows and shrinks. A
        !! slot that gets a new binding must not still answer to the old one, or
        !! a later lookup resolves to whatever previously occupied the slot.
        type(session_symbol_table_t) :: table

        call table%insert_binding(10, 1, 100, 7)
        call table%insert_binding(20, 1, 300, 7)
        call assert(table%find_binding(20, 1, 300) == 7, &
            'the slot answers to its new binding')
        call assert(table%find_binding(10, 1, 100) == 0, &
            'the slot no longer answers to the binding it replaced')
    end subroutine require_reused_slot_drops_the_stale_identity

    subroutine require_repeated_insert_is_idempotent()
        type(session_symbol_table_t) :: table

        call table%insert_binding(10, 1, 100, 7)
        call table%insert_binding(10, 1, 100, 7)
        call assert(table%find_binding(10, 1, 100) == 7, &
            'inserting the same binding twice resolves to the same symbol')
    end subroutine require_repeated_insert_is_idempotent

end program test_session_symbol_table
