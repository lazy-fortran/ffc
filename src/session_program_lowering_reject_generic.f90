submodule (session_program_lowering_impl) session_program_lowering_reject_generic
    implicit none
contains
    ! Malformed and ambiguous generic interfaces (#378).
    !
    ! FortFront keeps a generic interface block in the typed AST only when
    ! every interface body parses and every name resolves. The forms rejected
    ! here are dropped or merged before that point - a GENERIC binding with no
    ! binding list, an interface body terminated by a bare END, a MODULE
    ! PROCEDURE naming something that is not a module procedure, a generic name
    ! colliding with a contained procedure, and use association that makes a
    ! referenced name ambiguous - so the source text is the earliest layer that
    ! still holds enough information. Type, kind and rank questions are still
    ! answered from the typed AST through dummy_signature.
    module procedure check_generic_interface_forms
        character(len=:), allocatable :: source
        character(len=512), allocatable :: lines(:)
        integer :: count
        logical :: found

        call set_empty(error_msg)
        call get_source_text(arena, source, found)
        if (.not. found) return
        if (len_trim(source) == 0) return
        call generic_source_lines(source, lines, count)
        if (count == 0) return
        call check_generic_binding_syntax(lines, count, error_msg)
        if (len_trim(error_msg) > 0) return
        call check_implicit_interface_ambiguity(lines, count, error_msg)
        if (len_trim(error_msg) > 0) return
        call check_module_procedure_targets(lines, count, error_msg)
        if (len_trim(error_msg) > 0) return
        call check_generic_name_collisions(lines, count, error_msg)
        if (len_trim(error_msg) > 0) return
        call check_use_shadows_program_unit(lines, count, error_msg)
        if (len_trim(error_msg) > 0) return
        call check_ambiguous_use_association(lines, count, error_msg)
        if (len_trim(error_msg) > 0) return
        call check_typebound_generic_inheritance(arena, lines, count, error_msg)
        if (len_trim(error_msg) > 0) return
        call check_intrinsic_assignment_redefinition(arena, lines, count, &
                                                    error_msg)
        if (len_trim(error_msg) > 0) return
        call check_generic_call_resolves(arena, lines, count, error_msg)
    end procedure check_generic_interface_forms

    ! Source text split into comment-stripped, lowercased, left-adjusted lines.
    module procedure generic_source_lines
        character(len=:), allocatable :: raw, stripped
        integer :: pos, nl, total, i

        total = 1
        do i = 1, len(source)
            if (source(i:i) == new_line('a')) total = total + 1
        end do
        allocate (lines(total))
        lines = ''
        count = 0
        pos = 1
        do while (pos <= len(source))
            if (count >= total) exit
            nl = index(source(pos:), new_line('a'))
            if (nl == 0) then
                raw = source(pos:)
                pos = len(source) + 1
            else
                raw = source(pos:pos + nl - 2)
                pos = pos + nl
            end if
            call strip_line_comment(raw, stripped)
            count = count + 1
            lines(count) = adjustl(lowercase_text(stripped))
        end do
    end procedure generic_source_lines

    ! True when the statement text opens with the keyword kw as a whole word.
    module procedure stmt_starts
        character(len=:), allocatable :: text

        yes = .false.
        text = trim(line)
        if (len(text) < len(kw)) return
        if (text(1:len(kw)) /= kw) return
        if (len(text) == len(kw)) then
            yes = .true.
            return
        end if
        yes = .not. is_fortran_identifier_char(text(len(kw) + 1:len(kw) + 1))
    end procedure stmt_starts

    ! The first identifier in text at or after position from.
    module procedure identifier_after
        integer :: i, s, e

        name = ''
        s = 0
        do i = max(1, from), len_trim(text)
            if (is_fortran_identifier_char(text(i:i))) then
                s = i
                exit
            end if
        end do
        if (s == 0) return
        e = s
        do i = s, len_trim(text)
            if (.not. is_fortran_identifier_char(text(i:i))) exit
            e = i
        end do
        name = text(s:e)
    end procedure identifier_after

    module procedure append_owned_name

        if (len_trim(name) == 0) return
        if (n >= size(tab)) return
        n = n + 1
        tab(n) = name
        owners(n) = owner
    end procedure append_owned_name

    ! Whether name appears in tab; owner < 0 matches any owner.
    module procedure owned_name_present
        integer :: i

        yes = .false.
        do i = 1, n
            if (trim(tab(i)) /= trim(name)) cycle
            if (owner >= 0) then
                if (owners(i) /= owner) cycle
            end if
            yes = .true.
            return
        end do
    end procedure owned_name_present

    ! Interface nesting depth and enclosing named-generic name per line.
    module procedure interface_regions
        integer :: i, level
        character(len=64) :: current
        character(len=:), allocatable :: text, word

        allocate (depth(count))
        allocate (iface_name(count))
        level = 0
        current = ''
        do i = 1, count
            text = trim(lines(i))
            if (stmt_starts(text, 'end')) then
                word = identifier_after(text, 4)
                if (word == 'interface') then
                    level = 0
                    current = ''
                end if
            else if (stmt_starts(text, 'interface')) then
                level = 1
                current = identifier_after(text, 10)
            else if (stmt_starts(text, 'abstract')) then
                if (identifier_after(text, 9) == 'interface') then
                    level = 1
                    current = ''
                end if
            end if
            depth(i) = level
            iface_name(i) = current
        end do
    end procedure interface_regions

    ! A generic interface name proper: OPERATOR and ASSIGNMENT specifications
    ! are handled by their own rules.
    module procedure is_plain_generic_name

        yes = len_trim(name) > 0
        if (.not. yes) return
        if (trim(name) == 'operator') yes = .false.
        if (trim(name) == 'assignment') yes = .false.
        if (trim(name) == 'read') yes = .false.
        if (trim(name) == 'write') yes = .false.
    end procedure is_plain_generic_name

    ! F2018 R749: a generic binding is GENERIC [, access-spec] ::
    ! generic-spec => binding-name-list. A binding with no specification or no
    ! binding list names nothing and cannot be resolved.
    module procedure check_generic_binding_syntax
        character(len=:), allocatable :: text, rest, spec
        character(len=64) :: location
        integer :: i, p

        call set_empty(error_msg)
        do i = 1, count
            text = trim(lines(i))
            if (.not. stmt_starts(text, 'generic')) cycle
            rest = trim(adjustl(text(8:)))
            write (location, '(" at line ",I0)') i
            if (len(rest) == 0) then
                error_msg = 'malformed GENERIC binding: no generic '// &
                    'specification'//trim(location)
                return
            end if
            if (rest(1:1) /= ',' .and. rest(1:1) /= ':') cycle
            p = index(rest, '::')
            if (p == 0) then
                error_msg = 'malformed GENERIC binding: missing "::"'// &
                    trim(location)
                return
            end if
            spec = trim(adjustl(rest(p + 2:)))
            if (len(spec) == 0) then
                error_msg = 'malformed GENERIC binding: no generic '// &
                    'specification'//trim(location)
                return
            end if
            if (index(spec, '=>') == 0) then
                error_msg = 'malformed GENERIC binding: no binding name list'// &
                    trim(location)
                return
            end if
        end do
    end procedure check_generic_binding_syntax

    ! The procedure name and implicit-typing signature of an interface body
    ! header. ok is false when the line is not a procedure header.
    module procedure interface_body_header
        character(len=:), allocatable :: args, item
        integer :: p, lp, rp, start, i

        ok = .false.
        name = ''
        signature = ''
        if (stmt_starts(text, 'subroutine')) then
            p = 11
        else if (stmt_starts(text, 'function')) then
            p = 9
        else
            p = index(text, ' function ')
            if (p == 0) return
            p = p + 9
        end if
        name = identifier_after(text, p)
        if (len_trim(name) == 0) return
        ok = .true.
        lp = index(text, '(')
        rp = index(text, ')', back=.true.)
        if (lp == 0) return
        if (rp <= lp) return
        args = text(lp + 1:rp - 1)
        if (len_trim(args) == 0) return
        start = 1
        do i = 1, len(args) + 1
            if (i <= len(args)) then
                if (args(i:i) /= ',') cycle
            end if
            item = trim(adjustl(args(start:i - 1)))
            if (len(signature) > 0) signature = signature//','
            if (item == '*') then
                signature = signature//'*'
            else
                signature = signature//implicit_base_type(item)
            end if
            start = i + 1
        end do
    end procedure interface_body_header

    ! Two interface bodies of the same generic whose dummy arguments are all
    ! implicitly typed are distinguished by nothing when their implicit types
    ! agree position by position (F2018 C1514). Such bodies carry no
    ! specification part, so the typed AST cannot tell them apart either.
    module procedure check_implicit_interface_ambiguity
        integer, parameter :: MAXB = 128
        character(len=64) :: bname(MAXB), bsig(MAXB), bgen(MAXB)
        integer :: bline(MAXB)
        integer, allocatable :: depth(:)
        character(len=64), allocatable :: iface_name(:)
        character(len=:), allocatable :: text, name, signature
        character(len=64) :: location
        integer :: i, j, k, nb, spec_lines
        logical :: ok

        call set_empty(error_msg)
        call interface_regions(lines, count, depth, iface_name)
        nb = 0
        i = 0
        do while (i < count)
            i = i + 1
            if (depth(i) /= 1) cycle
            if (.not. is_plain_generic_name(iface_name(i))) cycle
            text = trim(lines(i))
            call interface_body_header(text, name, signature, ok)
            if (.not. ok) cycle
            spec_lines = 0
            j = i + 1
            do while (j <= count)
                if (stmt_starts(trim(lines(j)), 'end')) exit
                if (len_trim(lines(j)) > 0) spec_lines = spec_lines + 1
                j = j + 1
            end do
            if (spec_lines == 0 .and. nb < MAXB) then
                nb = nb + 1
                bname(nb) = name
                bsig(nb) = signature
                bgen(nb) = iface_name(i)
                bline(nb) = i
            end if
            i = j
        end do
        do j = 1, nb - 1
            do k = j + 1, nb
                if (bgen(j) /= bgen(k)) cycle
                if (bsig(j) /= bsig(k)) cycle
                write (location, '(" at line ",I0)') bline(k)
                error_msg = 'ambiguous interfaces '//trim(bname(j))//' and '// &
                    trim(bname(k))//' in generic interface '//trim(bgen(j))// &
                    trim(location)
                return
            end do
        end do
    end procedure check_implicit_interface_ambiguity

    ! Append every comma-separated identifier of rest to the table.
    module procedure append_name_list
        character(len=:), allocatable :: item
        integer :: i, start

        start = 1
        do i = 1, len_trim(rest) + 1
            if (i <= len_trim(rest)) then
                if (rest(i:i) /= ',') cycle
            end if
            item = identifier_after(rest(start:i - 1), 1)
            call append_owned_name(tab, owners, n, item, owner)
            start = i + 1
        end do
    end procedure append_name_list

    ! Program-unit index per line: MODULE and PROGRAM units partition the file.
    module procedure program_unit_ids
        character(len=:), allocatable :: text, word
        integer :: i, unit

        allocate (ids(count))
        unit = 1
        do i = 1, count
            text = trim(lines(i))
            if (stmt_starts(text, 'module')) then
                word = identifier_after(text, 7)
                if (word /= 'procedure' .and. word /= 'subroutine' .and. &
                    word /= 'function' .and. len_trim(word) > 0) unit = unit + 1
            else if (stmt_starts(text, 'program')) then
                unit = unit + 1
            else if (stmt_starts(text, 'submodule')) then
                unit = unit + 1
            end if
            ids(i) = unit
            if (stmt_starts(text, 'end')) then
                word = identifier_after(text, 4)
                if (word == 'module' .or. word == 'program' .or. &
                    word == 'submodule') unit = unit + 1
            end if
        end do
    end procedure program_unit_ids

    ! Collect the procedure definitions, interface bodies, generic names and
    ! EXTERNAL declarations of the whole file, tagged by program unit.
    module procedure collect_procedure_tables
        integer, allocatable :: depth(:), ids(:)
        character(len=64), allocatable :: iface_name(:)
        character(len=:), allocatable :: text, name, signature
        logical :: ok
        integer :: i

        call interface_regions(lines, count, depth, iface_name)
        call program_unit_ids(lines, count, ids)
        ndef = 0
        nbody = 0
        ngen = 0
        next = 0
        do i = 1, count
            text = trim(lines(i))
            if (stmt_starts(text, 'external')) then
                call append_name_list(exts, ext_owner, next, text(9:), ids(i))
                cycle
            end if
            if (stmt_starts(text, 'interface')) then
                name = identifier_after(text, 10)
                if (is_plain_generic_name(name)) then
                    call append_owned_name(gens, gen_owner, ngen, name, ids(i))
                end if
                cycle
            end if
            if (stmt_starts(text, 'end')) cycle
            if (stmt_starts(text, 'module')) cycle
            call interface_body_header(text, name, signature, ok)
            if (.not. ok) cycle
            if (depth(i) == 1) then
                call append_owned_name(bodies, body_owner, nbody, name, ids(i))
            else
                call append_owned_name(defs, def_owner, ndef, name, ids(i))
            end if
        end do
    end procedure collect_procedure_tables

    ! F2018 C1507: a MODULE PROCEDURE statement in a generic interface may only
    ! name a module procedure. A name that the file declares EXTERNAL, declares
    ! through an interface body, or uses as a generic name is not one.
    module procedure check_module_procedure_targets
        integer, parameter :: MAXN = 512
        character(len=64) :: defs(MAXN), bodies(MAXN), gens(MAXN), exts(MAXN)
        integer :: def_owner(MAXN), body_owner(MAXN), gen_owner(MAXN), &
                   ext_owner(MAXN)
        character(len=64) :: targets(MAXN)
        integer :: target_owner(MAXN)
        integer, allocatable :: depth(:)
        character(len=64), allocatable :: iface_name(:)
        character(len=:), allocatable :: text
        character(len=64) :: location
        integer :: ndef, nbody, ngen, next, ntarget, i, k
        logical :: bad

        call set_empty(error_msg)
        call interface_regions(lines, count, depth, iface_name)
        call collect_procedure_tables(lines, count, defs, def_owner, ndef, &
                                      bodies, body_owner, nbody, gens, &
                                      gen_owner, ngen, exts, ext_owner, next)
        do i = 1, count
            if (depth(i) /= 1) cycle
            if (.not. is_plain_generic_name(iface_name(i))) cycle
            text = trim(lines(i))
            if (.not. stmt_starts(text, 'module')) cycle
            if (identifier_after(text, 7) /= 'procedure') cycle
            ntarget = 0
            call append_name_list(targets, target_owner, ntarget, &
                                  text(index(text, 'procedure') + 9:), 0)
            do k = 1, ntarget
                if (owned_name_present(defs, def_owner, ndef, targets(k), -1)) &
                    cycle
                bad = owned_name_present(exts, ext_owner, next, targets(k), -1)
                if (.not. bad) bad = owned_name_present(bodies, body_owner, &
                                                        nbody, targets(k), -1)
                if (.not. bad) bad = owned_name_present(gens, gen_owner, ngen, &
                                                        targets(k), -1)
                if (.not. bad) cycle
                write (location, '(" at line ",I0)') i
                error_msg = 'MODULE PROCEDURE '//trim(targets(k))// &
                    ' in generic interface '//trim(iface_name(i))// &
                    ' is not a module procedure'//trim(location)
                return
            end do
        end do
    end procedure check_module_procedure_targets

    ! Specific procedure names listed by the generic interface block whose
    ! header is at line header_line.
    module procedure generic_block_specifics
        integer :: owners(size(specs))
        character(len=:), allocatable :: text, name, signature
        integer :: i
        logical :: ok

        nspec = 0
        do i = header_line + 1, count
            text = trim(lines(i))
            if (stmt_starts(text, 'end')) then
                if (identifier_after(text, 4) == 'interface') return
            end if
            if (stmt_starts(text, 'module')) then
                if (identifier_after(text, 7) == 'procedure') then
                    call append_name_list(specs, owners, nspec, &
                                          text(index(text, 'procedure') + 9:), 0)
                end if
                cycle
            end if
            call interface_body_header(text, name, signature, ok)
            if (ok) call append_owned_name(specs, owners, nspec, name, 0)
        end do
    end procedure generic_block_specifics

    ! A generic name and a procedure defined in the same scoping unit may only
    ! share a name when that procedure is one of the generic's own specifics
    ! (F2018 15.4.3.4.1). Otherwise the name is already defined as a generic.
    module procedure check_generic_name_collisions
        integer, parameter :: MAXN = 512
        character(len=64) :: defs(MAXN), bodies(MAXN), gens(MAXN), exts(MAXN)
        integer :: def_owner(MAXN), body_owner(MAXN), gen_owner(MAXN), &
                   ext_owner(MAXN)
        character(len=64) :: specs(MAXN)
        integer, allocatable :: depth(:), ids(:)
        character(len=64), allocatable :: iface_name(:)
        character(len=:), allocatable :: text, name
        character(len=64) :: location
        integer :: ndef, nbody, ngen, next, nspec, i, k

        call set_empty(error_msg)
        call interface_regions(lines, count, depth, iface_name)
        call program_unit_ids(lines, count, ids)
        call collect_procedure_tables(lines, count, defs, def_owner, ndef, &
                                      bodies, body_owner, nbody, gens, &
                                      gen_owner, ngen, exts, ext_owner, next)
        do i = 1, count
            text = trim(lines(i))
            if (.not. stmt_starts(text, 'interface')) cycle
            name = identifier_after(text, 10)
            if (.not. is_plain_generic_name(name)) cycle
            if (.not. owned_name_present(defs, def_owner, ndef, name, ids(i))) &
                cycle
            call generic_block_specifics(lines, count, i, specs, nspec)
            do k = 1, nspec
                if (trim(specs(k)) == trim(name)) return
            end do
            write (location, '(" at line ",I0)') i
            error_msg = trim(name)//' is already defined as a generic '// &
                'interface and cannot also name a procedure in the same '// &
                'scoping unit'//trim(location)
            return
        end do
    end procedure check_generic_name_collisions

    ! Names a module in this file makes accessible: its generic names, its
    ! interface bodies, and the procedures it defines.
    module procedure module_export_table
        integer, allocatable :: depth(:)
        character(len=64), allocatable :: iface_name(:)
        character(len=:), allocatable :: text, name, signature, word
        integer :: i, current
        logical :: ok, in_proc

        call interface_regions(lines, count, depth, iface_name)
        nmod = 0
        nname = 0
        current = 0
        in_proc = .false.
        do i = 1, count
            text = trim(lines(i))
            ! Everything inside a module procedure body, an interface block of
            ! its own included, is local to that procedure and exports nothing.
            if (in_proc) then
                if (depth(i) == 0) then
                    if (stmt_starts(text, 'end')) then
                        word = identifier_after(text, 4)
                        if (len_trim(word) == 0 .or. word == 'subroutine' .or. &
                            word == 'function') in_proc = .false.
                        if (word == 'module') then
                            in_proc = .false.
                            current = 0
                        end if
                    end if
                end if
                cycle
            end if
            if (stmt_starts(text, 'end')) then
                if (identifier_after(text, 4) == 'module') current = 0
                cycle
            end if
            if (stmt_starts(text, 'module')) then
                name = identifier_after(text, 7)
                if (name /= 'procedure' .and. name /= 'subroutine' .and. &
                    name /= 'function' .and. len_trim(name) > 0) then
                    if (nmod < size(mods)) then
                        nmod = nmod + 1
                        mods(nmod) = name
                        current = nmod
                    end if
                end if
                cycle
            end if
            if (current == 0) cycle
            if (stmt_starts(text, 'interface')) then
                name = identifier_after(text, 10)
                if (is_plain_generic_name(name)) then
                    call append_owned_name(names, owner, nname, name, current)
                end if
                cycle
            end if
            call interface_body_header(text, name, signature, ok)
            if (ok) then
                call append_owned_name(names, owner, nname, name, current)
                if (depth(i) == 0) in_proc = .true.
            end if
        end do
    end procedure module_export_table

    ! The module index of a USE statement's module, or 0.
    module procedure used_module_index
        character(len=:), allocatable :: name
        integer :: i

        idx = 0
        name = identifier_after(text, 4)
        do i = 1, nmod
            if (trim(mods(i)) == trim(name)) then
                idx = i
                return
            end if
        end do
    end procedure used_module_index

    ! A USE statement must not make accessible a name that is also the name of
    ! the current program unit; the two entities then collide in the same
    ! scope. The enclosing unit is the nearest preceding unit header, since a
    ! USE statement stands at the head of the specification part.
    module procedure check_use_shadows_program_unit
        integer, parameter :: MAXN = 512
        character(len=64) :: mods(MAXN), names(MAXN)
        integer :: owner(MAXN)
        integer, allocatable :: depth(:)
        character(len=64), allocatable :: iface_name(:)
        character(len=:), allocatable :: text, unit_name, signature, hname
        character(len=64) :: location
        integer :: nmod, nname, i, j, midx
        logical :: ok

        call set_empty(error_msg)
        call interface_regions(lines, count, depth, iface_name)
        call module_export_table(lines, count, mods, nmod, names, owner, nname)
        if (nmod == 0) return
        do i = 1, count
            if (depth(i) /= 0) cycle
            text = trim(lines(i))
            if (.not. stmt_starts(text, 'use')) cycle
            midx = used_module_index(text, mods, nmod)
            if (midx == 0) cycle
            unit_name = ''
            do j = i - 1, 1, -1
                if (depth(j) /= 0) cycle
                hname = trim(lines(j))
                if (stmt_starts(hname, 'module')) exit
                if (stmt_starts(hname, 'program')) exit
                if (stmt_starts(hname, 'end')) exit
                call interface_body_header(hname, unit_name, signature, ok)
                if (ok) exit
                unit_name = ''
            end do
            if (len_trim(unit_name) == 0) cycle
            if (.not. owned_name_present(names, owner, nname, unit_name, midx)) &
                cycle
            if (.not. use_makes_name_accessible(text, unit_name)) cycle
            write (location, '(" at line ",I0)') i
            error_msg = trim(unit_name)//' from module '//trim(mods(midx))// &
                ' is also the name of the current program unit'//trim(location)
            return
        end do
    end procedure check_use_shadows_program_unit

    ! Whether a USE statement makes name accessible under that same name. A
    ! rename moves the module entity to a different local name, and an ONLY
    ! clause admits nothing that it does not list, so in both cases the name
    ! itself never enters the local scope (F2018 14.2.2).
    module procedure use_makes_name_accessible
        character(len=:), allocatable :: tail, item, use_name
        logical :: only_form
        integer :: comma, p, start, i

        yes = .true.
        comma = index(text, ',')
        if (comma == 0) return
        tail = text(comma + 1:)
        only_form = .false.
        p = index(tail, 'only')
        if (p > 0) then
            p = index(tail, ':')
            if (p > 0) then
                only_form = .true.
                tail = tail(p + 1:)
            end if
        end if
        if (only_form) yes = .false.
        start = 1
        do i = 1, len(tail) + 1
            if (i <= len(tail)) then
                if (tail(i:i) /= ',') cycle
            end if
            item = trim(adjustl(tail(start:i - 1)))
            start = i + 1
            if (len_trim(item) == 0) cycle
            p = index(item, '=>')
            if (p > 0) then
                use_name = trim(adjustl(item(p + 2:)))
                if (use_name == trim(name)) then
                    yes = .false.
                    return
                end if
                cycle
            end if
            if (.not. only_form) cycle
            if (item == trim(name)) yes = .true.
        end do
    end procedure use_makes_name_accessible

    ! True when name occurs as a whole word on the line.
    module procedure line_references_name
        integer :: p, base

        yes = .false.
        base = 0
        do
            p = index(line(base + 1:), trim(name))
            if (p == 0) return
            p = base + p
            base = p
            if (p > 1) then
                if (is_fortran_identifier_char(line(p - 1:p - 1))) cycle
                ! A component or type-bound reference names a part of an
                ! object, never the use associated entity of that name.
                if (line(p - 1:p - 1) == '%') cycle
            end if
            ! A component or type-bound name after % resolves through the
            ! declared type, not through use association, so it never makes
            ! an ambiguous use-associated name referenced.
            if (preceded_by_component_selector(line, p)) cycle
            if (p + len_trim(name) <= len(line)) then
                if (is_fortran_identifier_char(line(p + len_trim(name): &
                                                    p + len_trim(name)))) cycle
            end if
            yes = .true.
            return
        end do
    end procedure line_references_name

    ! Whether the identifier starting at pos is preceded by a % selector.
    module procedure preceded_by_component_selector
        integer :: i

        yes = .false.
        do i = pos - 1, 1, -1
            if (line(i:i) == ' ') cycle
            yes = line(i:i) == '%'
            return
        end do
    end procedure preceded_by_component_selector

    ! Two modules that make the same name accessible to one scoping unit leave
    ! that name ambiguous; referencing it is an error (F2018 19.5.1.4).
    module procedure check_ambiguous_use_association
        integer, parameter :: MAXN = 512
        character(len=64) :: mods(MAXN), names(MAXN)
        integer :: owner(MAXN)
        integer, allocatable :: depth(:), region(:)
        character(len=64), allocatable :: iface_name(:)
        integer :: nmod, nname, i, j, k, midx, first, last
        integer :: used(64), nused
        character(len=:), allocatable :: text
        character(len=64) :: location

        call set_empty(error_msg)
        call interface_regions(lines, count, depth, iface_name)
        call module_export_table(lines, count, mods, nmod, names, owner, nname)
        if (nmod < 2) return
        call scoping_regions(lines, count, depth, region)
        first = 1
        do while (first <= count)
            last = first
            do while (last < count)
                if (region(last + 1) /= region(first)) exit
                last = last + 1
            end do
            nused = 0
            do i = first, last
                if (depth(i) /= 0) cycle
                text = trim(lines(i))
                if (.not. stmt_starts(text, 'use')) cycle
                if (index(text, 'only') > 0) cycle
                midx = used_module_index(text, mods, nmod)
                if (midx == 0) cycle
                if (nused < size(used)) then
                    nused = nused + 1
                    used(nused) = midx
                end if
            end do
            if (nused >= 2) then
                do j = 1, nname
                    if (.not. any(used(1:nused) == owner(j))) cycle
                    do k = 1, nname
                        if (k == j) cycle
                        if (trim(names(k)) /= trim(names(j))) cycle
                        if (owner(k) == owner(j)) cycle
                        if (.not. any(used(1:nused) == owner(k))) cycle
                        if (generic_extends_own_name(lines, count, mods, &
                                                     owner(j), names(j))) cycle
                        if (generic_extends_own_name(lines, count, mods, &
                                                     owner(k), names(k))) cycle
                        call ambiguous_reference_line(lines, first, last, &
                                                      names(j), i)
                        if (i == 0) cycle
                        write (location, '(" at line ",I0)') i
                        error_msg = 'ambiguous reference to '//trim(names(j))// &
                            ': it is use associated from more than one '// &
                            'module'//trim(location)
                        return
                    end do
                end do
            end if
            first = last + 1
        end do
    end procedure check_ambiguous_use_association

    ! True when the module makes name generic through an interface block that
    ! lists name itself as a specific. The generic then extends the very entity
    ! of that name it inherits, so USE of both modules names one generic
    ! (F2018 15.4.3.4.1) rather than two conflicting entities.
    module procedure generic_extends_own_name
        character(len=64) :: specs(128)
        character(len=:), allocatable :: text, word
        integer :: i, k, nspec
        logical :: inside

        yes = .false.
        inside = .false.
        do i = 1, count
            text = trim(lines(i))
            if (stmt_starts(text, 'end')) then
                if (identifier_after(text, 4) == 'module') inside = .false.
                cycle
            end if
            if (stmt_starts(text, 'module')) then
                word = identifier_after(text, 7)
                inside = word == trim(mods(midx))
                cycle
            end if
            if (.not. inside) cycle
            if (.not. stmt_starts(text, 'interface')) cycle
            if (identifier_after(text, 10) /= trim(name)) cycle
            call generic_block_specifics(lines, count, i, specs, nspec)
            do k = 1, nspec
                if (trim(specs(k)) == trim(name)) then
                    yes = .true.
                    return
                end if
            end do
        end do
    end procedure generic_extends_own_name

    ! The first line of the region that references name outside a USE
    ! statement, or 0.
    module procedure ambiguous_reference_line
        integer :: i
        character(len=:), allocatable :: text

        found_line = 0
        do i = first, last
            text = trim(lines(i))
            if (stmt_starts(text, 'use')) cycle
            if (.not. line_references_name(text, name)) cycle
            found_line = i
            return
        end do
    end procedure ambiguous_reference_line

    ! Region index per line: a new region starts at every program unit header
    ! and after every unit END statement, so each scoping unit gets its own.
    module procedure scoping_regions
        character(len=:), allocatable :: text, name, signature
        integer :: i, current
        logical :: ok

        allocate (region(count))
        current = 1
        do i = 1, count
            text = trim(lines(i))
            if (depth(i) == 0) then
                if (stmt_starts(text, 'program') .or. &
                    stmt_starts(text, 'module')) then
                    current = current + 1
                else
                    call interface_body_header(text, name, signature, ok)
                    if (ok) current = current + 1
                end if
            end if
            region(i) = current
            if (depth(i) == 0 .and. stmt_starts(text, 'end')) then
                current = current + 1
            end if
        end do
    end procedure scoping_regions

    ! A derived type that extends another must not bind the same generic
    ! operator when the inherited and the new specific are not distinguishable:
    ! an actual of the extension type matches both (F2018 C1514).
    module procedure check_typebound_generic_inheritance
        integer, parameter :: MAXT = 128
        character(len=64) :: tname(MAXT), tparent(MAXT)
        character(len=64) :: gspec(MAXT), gtarget(MAXT)
        integer :: gtype(MAXT), gline(MAXT)
        character(len=:), allocatable :: text
        character(len=64) :: location
        integer :: i, j, k, nt, ng

        call set_empty(error_msg)
        call collect_type_generics(lines, count, tname, tparent, nt, gspec, &
                                   gtarget, gtype, gline, ng)
        do j = 1, ng
            do k = 1, ng
                if (j == k) cycle
                if (gspec(j) /= gspec(k)) cycle
                if (gtype(j) == gtype(k)) cycle
                if (.not. type_extends_type(tname, tparent, nt, gtype(j), &
                                            gtype(k))) cycle
                if (.not. bindings_indistinguishable(arena, trim(gtarget(j)), &
                                                     trim(gtarget(k)), tname, &
                                                     tparent, nt)) cycle
                write (location, '(" at line ",I0)') gline(j)
                error_msg = 'type-bound generic '//trim(gspec(j))//' of type '// &
                    trim(tname(gtype(j)))//' and the binding inherited from '// &
                    trim(tname(gtype(k)))//' are ambiguous'//trim(location)
                return
            end do
        end do
    end procedure check_typebound_generic_inheritance

    ! Like specifics_indistinguishable, but a declared type and any of its
    ! extensions do not distinguish two bindings: an actual of the extension
    ! type is type compatible with both dummies.
    module procedure bindings_indistinguishable
        integer :: count_a, count_b, pos
        integer :: kind_a, kind_b, rank_a, rank_b
        logical :: known_a, known_b, proc_a, proc_b, any_a, any_b
        character(len=:), allocatable :: base_a, base_b

        same = .false.
        count_a = arena_proc_param_count(arena, name_a)
        count_b = arena_proc_param_count(arena, name_b)
        if (count_a < 0 .or. count_b < 0) return
        if (count_a /= count_b) return
        do pos = 1, count_a
            call dummy_signature(arena, name_a, pos, known_a, base_a, kind_a, &
                                 rank_a, proc_a, any_a)
            call dummy_signature(arena, name_b, pos, known_b, base_b, kind_b, &
                                 rank_b, proc_b, any_b)
            if (.not. known_a) return
            if (.not. known_b) return
            if (proc_a .neqv. proc_b) return
            if (proc_a) cycle
            if (any_a .or. any_b) cycle
            if (rank_a /= rank_b) return
            if (base_a == base_b) cycle
            if (.not. base_types_related(base_a, base_b, tname, tparent, nt)) &
                return
        end do
        same = .true.
    end procedure bindings_indistinguishable

    ! Whether two declared derived types are related by type extension.
    module procedure base_types_related
        integer :: ia, ib

        related = .false.
        ia = declared_type_index(base_a, tname, nt)
        ib = declared_type_index(base_b, tname, nt)
        if (ia == 0 .or. ib == 0) return
        related = type_extends_type(tname, tparent, nt, ia, ib)
        if (.not. related) then
            related = type_extends_type(tname, tparent, nt, ib, ia)
        end if
    end procedure base_types_related

    ! Index in the collected type table of the declared type spec type(x).
    module procedure declared_type_index
        character(len=:), allocatable :: name
        integer :: i

        idx = 0
        if (len_trim(base) < 6) return
        if (base(1:5) /= 'type(') return
        name = identifier_after(base, 6)
        do i = 1, nt
            if (trim(tname(i)) /= trim(name)) cycle
            idx = i
            return
        end do
    end procedure declared_type_index

    module procedure collect_type_generics
        character(len=:), allocatable :: text, spec
        integer :: i, p, current

        nt = 0
        ng = 0
        current = 0
        do i = 1, count
            text = trim(lines(i))
            if (stmt_starts(text, 'end')) then
                if (identifier_after(text, 4) == 'type') current = 0
                cycle
            end if
            if (stmt_starts(text, 'type')) then
                p = index(text, '::')
                if (p == 0) cycle
                if (nt >= size(tname)) cycle
                nt = nt + 1
                tname(nt) = identifier_after(text, p + 2)
                tparent(nt) = ''
                p = index(text, 'extends(')
                if (p > 0) tparent(nt) = identifier_after(text, p + 8)
                current = nt
                cycle
            end if
            if (current == 0) cycle
            if (.not. stmt_starts(text, 'generic')) cycle
            p = index(text, '::')
            if (p == 0) cycle
            spec = trim(adjustl(text(p + 2:)))
            p = index(spec, '=>')
            if (p == 0) cycle
            if (ng >= size(gspec)) cycle
            ng = ng + 1
            gspec(ng) = squeeze_blanks(spec(1:p - 1))
            gtarget(ng) = identifier_after(spec(p + 2:), 1)
            gtype(ng) = current
            gline(ng) = i
        end do
    end procedure collect_type_generics

    module procedure squeeze_blanks
        integer :: i

        packed = ''
        do i = 1, len_trim(text)
            if (text(i:i) == ' ') cycle
            packed = packed//text(i:i)
        end do
    end procedure squeeze_blanks

    ! Whether type child is an extension of type ancestor.
    module procedure type_extends_type
        integer :: i

        yes = .false.
        if (child < 1 .or. child > nt) return
        if (len_trim(tparent(child)) == 0) return
        do i = 1, nt
            if (trim(tname(i)) /= trim(tparent(child))) cycle
            if (i == ancestor) then
                yes = .true.
                return
            end if
            yes = type_extends_type(tname, tparent, nt, i, ancestor)
            return
        end do
    end procedure type_extends_type

    ! F2018 C1503: a defined assignment must not redefine an assignment that
    ! is already defined intrinsically for its two operands.
    module procedure check_intrinsic_assignment_redefinition
        integer, parameter :: MAXN = 64
        character(len=64) :: specs(MAXN)
        character(len=:), allocatable :: text, base_l, base_r
        character(len=64) :: location
        integer :: i, k, nspec, kind_l, kind_r, rank_l, rank_r
        logical :: known_l, known_r, proc_l, proc_r, any_l, any_r

        call set_empty(error_msg)
        do i = 1, count
            text = trim(lines(i))
            if (.not. stmt_starts(text, 'interface')) cycle
            if (identifier_after(text, 10) /= 'assignment') cycle
            call generic_block_specifics(lines, count, i, specs, nspec)
            do k = 1, nspec
                call dummy_signature(arena, trim(specs(k)), 1, known_l, base_l, &
                                     kind_l, rank_l, proc_l, any_l)
                call dummy_signature(arena, trim(specs(k)), 2, known_r, base_r, &
                                     kind_r, rank_r, proc_r, any_r)
                if (.not. known_l) cycle
                if (.not. known_r) cycle
                if (proc_l .or. proc_r) cycle
                if (rank_l /= rank_r) cycle
                if (.not. intrinsic_assignment_defined(base_type_root(base_l), &
                                                       base_type_root(base_r))) &
                    cycle
                write (location, '(" at line ",I0)') i
                error_msg = trim(specs(k))//' would redefine an INTRINSIC '// &
                    'type assignment between '//base_l//' and '//base_r// &
                    trim(location)
                return
            end do
        end do
    end procedure check_intrinsic_assignment_redefinition

    module procedure intrinsic_assignment_defined

        yes = .false.
        if (is_numeric_base_type(base_l) .and. is_numeric_base_type(base_r)) then
            yes = .true.
            return
        end if
        if (base_l == 'character' .and. base_r == 'character') yes = .true.
        if (base_l == 'logical' .and. base_r == 'logical') yes = .true.
    end procedure intrinsic_assignment_defined

    module procedure is_numeric_base_type

        yes = base == 'integer' .or. base == 'real' .or. base == 'complex'
    end procedure is_numeric_base_type

    ! A reference to a generic name must match one of its specific procedures
    ! (F2018 15.5.5). Only calls whose actual arguments are all literal
    ! constants are judged, so a mismatch is beyond doubt.
    module procedure check_generic_call_resolves
        integer, parameter :: MAXN = 64
        character(len=64) :: specs(MAXN), actuals(MAXN)
        character(len=:), allocatable :: text, callee
        character(len=64) :: location
        integer :: i, nspec, nactual
        logical :: matched, resolvable

        call set_empty(error_msg)
        do i = 1, count
            text = trim(lines(i))
            if (.not. stmt_starts(text, 'call')) cycle
            callee = identifier_after(text, 5)
            if (len_trim(callee) == 0) cycle
            call literal_actual_types(text, actuals, nactual)
            if (nactual <= 0) cycle
            call generic_union_specifics(lines, count, callee, specs, nspec)
            if (nspec == 0) cycle
            call generic_call_matches(arena, specs, nspec, actuals, nactual, &
                                      matched, resolvable)
            if (.not. resolvable) cycle
            if (matched) cycle
            write (location, '(" at line ",I0)') i
            error_msg = 'no specific subroutine of the generic interface '// &
                trim(callee)//' matches this reference'//trim(location)
            return
        end do
    end procedure check_generic_call_resolves

    ! The union of the specific procedures of every generic interface block
    ! that declares name. One reference may see several of them at once: a
    ! generic can be extended by host association and by USE association in
    ! the same scope (F2018 15.4.3.4.1), so judging a call against a single
    ! block would reject valid references.
    module procedure generic_union_specifics
        character(len=64) :: block_specs(64)
        character(len=:), allocatable :: text
        integer :: i, j, k, nblock

        nspec = 0
        do i = 1, count
            text = trim(lines(i))
            if (.not. stmt_starts(text, 'interface')) cycle
            if (identifier_after(text, 10) /= trim(name)) cycle
            call generic_block_specifics(lines, count, i, block_specs, nblock)
            do k = 1, nblock
                if (nspec >= size(specs)) exit
                ! One specific name declared by two blocks names two distinct
                ! procedures, which a name lookup cannot tell apart. Judge
                ! nothing rather than judge the wrong signature.
                do j = 1, nspec
                    if (trim(specs(j)) == trim(block_specs(k))) then
                        nspec = 0
                        return
                    end if
                end do
                nspec = nspec + 1
                specs(nspec) = block_specs(k)
            end do
        end do
    end procedure generic_union_specifics

    ! Base types of a call's actual arguments when every one is a literal
    ! constant; nactual is -1 otherwise.
    module procedure literal_actual_types
        character(len=:), allocatable :: args, item, base
        integer :: lp, rp, i, start

        nactual = -1
        lp = index(text, '(')
        rp = index(text, ')', back=.true.)
        if (lp == 0) return
        if (rp <= lp) return
        args = text(lp + 1:rp - 1)
        if (len_trim(args) == 0) return
        nactual = 0
        start = 1
        do i = 1, len(args) + 1
            if (i <= len(args)) then
                if (args(i:i) /= ',') cycle
            end if
            item = trim(adjustl(args(start:i - 1)))
            base = literal_base_type(item)
            if (len_trim(base) == 0) then
                nactual = -1
                return
            end if
            if (nactual >= size(actuals)) then
                nactual = -1
                return
            end if
            nactual = nactual + 1
            actuals(nactual) = base
            start = i + 1
        end do
    end procedure literal_actual_types

    module procedure literal_base_type
        integer :: i
        logical :: has_digit, has_dot, has_other

        base = ''
        if (len_trim(item) == 0) return
        if (item(1:1) == '''' .or. item(1:1) == '"') then
            base = 'character'
            return
        end if
        if (item == '.true.' .or. item == '.false.') then
            base = 'logical'
            return
        end if
        has_digit = .false.
        has_dot = .false.
        has_other = .false.
        do i = 1, len_trim(item)
            select case (item(i:i))
            case ('0':'9')
                has_digit = .true.
            case ('.')
                has_dot = .true.
            case ('+', '-')
                continue
            case default
                has_other = .true.
            end select
        end do
        if (has_other) return
        if (.not. has_digit) return
        if (has_dot) then
            base = 'real'
        else
            base = 'integer'
        end if
    end procedure literal_base_type

    module procedure generic_call_matches
        character(len=:), allocatable :: base
        integer :: k, pos, params, kind_value, rank
        logical :: known, is_proc, is_any, fits

        matched = .false.
        resolvable = .false.
        do k = 1, nspec
            params = arena_proc_param_count(arena, trim(specs(k)))
            if (params < 0) return
            resolvable = .true.
            if (params /= nactual) cycle
            fits = .true.
            do pos = 1, nactual
                call dummy_signature(arena, trim(specs(k)), pos, known, base, &
                                     kind_value, rank, is_proc, is_any)
                if (.not. known) return
                if (is_proc) return
                if (is_any) cycle
                if (rank /= 0) then
                    fits = .false.
                    exit
                end if
                if (base_type_root(base) /= trim(actuals(pos))) then
                    fits = .false.
                    exit
                end if
            end do
            if (fits) then
                matched = .true.
                return
            end if
        end do
    end procedure generic_call_matches

end submodule session_program_lowering_reject_generic
