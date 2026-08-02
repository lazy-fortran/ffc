module session_program_lowering_reject_text
    implicit none
    private
    public :: normalized_base_type
    public :: base_type_root
    public :: implicit_base_type
    public :: starts_with_word

contains

    ! Canonicalize a declaration type for generic distinguishability.
    ! CLASS(T) and TYPE(T) name the same declared type; retain kind/length
    ! selectors because callers may use them to distinguish specifics.
    function normalized_base_type(type_name) result(base)
        character(len=*), intent(in) :: type_name
        character(len=:), allocatable :: base

        base = squeeze_blanks(lowercase_text(trim(adjustl(type_name))))
        if (base == 'class(*)') return
        if (len(base) > 6) then
            if (base(1:6) == 'class(') base = 'type('//base(7:)
        end if
    end function normalized_base_type

    ! Return the intrinsic root of a canonical type name, without a kind or
    ! length selector. Derived type names remain intact.
    function base_type_root(base) result(root)
        character(len=*), intent(in) :: base
        character(len=:), allocatable :: root
        integer :: left_paren

        root = trim(base)
        if (root == 'class(*)') return
        if (len(root) > 5) then
            if (root(1:5) == 'type(') return
        end if
        left_paren = index(root, '(')
        if (left_paren > 1) root = root(1:left_paren - 1)
    end function base_type_root

    ! Apply the standard implicit typing rule to an identifier.
    function implicit_base_type(name) result(base)
        character(len=*), intent(in) :: name
        character(len=:), allocatable :: base
        character(len=1) :: first

        base = ''
        if (len_trim(name) == 0) return
        first = lowercase_text(name(1:1))
        if (first < 'a' .or. first > 'z') return
        if (first >= 'i' .and. first <= 'n') then
            base = 'integer'
        else
            base = 'real'
        end if
    end function implicit_base_type

    logical function starts_with_word(text, word) result(matches)
        character(len=*), intent(in) :: text, word
        integer :: word_length

        word_length = len(word)
        matches = .false.
        if (len(text) < word_length) return
        if (text(1:word_length) /= word) return
        if (len(text) == word_length) then
            matches = .true.
        else
            matches = scan(text(word_length + 1:word_length + 1), &
                           'abcdefghijklmnopqrstuvwxyz_') == 0
        end if
    end function starts_with_word

    function lowercase_text(text) result(lowered)
        character(len=*), intent(in) :: text
        character(len=len(text)) :: lowered
        integer :: i, code

        lowered = text
        do i = 1, len(text)
            code = iachar(lowered(i:i))
            if (code >= iachar('A') .and. code <= iachar('Z')) then
                lowered(i:i) = achar(code + iachar('a') - iachar('A'))
            end if
        end do
    end function lowercase_text

    function squeeze_blanks(text) result(packed)
        character(len=*), intent(in) :: text
        character(len=:), allocatable :: packed
        integer :: i, n

        packed = ''
        n = len_trim(text)
        do i = 1, n
            if (text(i:i) /= ' ') packed = packed//text(i:i)
        end do
    end function squeeze_blanks

end module session_program_lowering_reject_text
