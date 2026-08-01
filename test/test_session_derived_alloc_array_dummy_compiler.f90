program test_session_derived_alloc_array_dummy_compiler
    ! Arrays whose elements carry allocatable components crossing a procedure
    ! boundary (#406). The outer array keeps the canonical contiguous derived
    ! layout, so an explicit-shape or assumed-shape dummy binds the caller's
    ! storage directly and each element's component descriptor stays owned by
    ! the actual: the dummy borrows it, reads its extent through size(), and
    ! reaches its heap elements through the same data pointer. Negative cases
    ! keep the compiler honest: a type mismatch, a rank mismatch, and a write
    ! through an INTENT(IN) dummy are all diagnosed instead of lowered.
    use ffc_test_support, only: expect_output, expect_exit_status, &
                                expect_error_contains
    implicit none

    logical :: all_passed

    print *, '=== direct session derived allocatable array dummy test ==='

    all_passed = .true.
    if (.not. test_assumed_shape_rank1()) all_passed = .false.
    if (.not. test_explicit_shape_rank1()) all_passed = .false.
    if (.not. test_assumed_shape_rank2()) all_passed = .false.
    if (.not. test_reject_type_mismatch()) all_passed = .false.
    if (.not. test_reject_rank_mismatch()) all_passed = .false.
    if (.not. test_reject_intent_in_mutation()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: derived allocatable array dummies lower through session'

contains

    logical function test_assumed_shape_rank1()
        ! A rank-1 assumed-shape dummy sees the actual's element count and each
        ! element's component extent and values.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: box_t'//new_line('a')// &
            '    integer, allocatable :: v(:)'//new_line('a')// &
            '  end type box_t'//new_line('a')// &
            '  type(box_t) :: b(3)'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  do i = 1, 3'//new_line('a')// &
            '    allocate(b(i)%v(i))'//new_line('a')// &
            '    b(i)%v(1) = i * 100'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  call show(b)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine show(arr)'//new_line('a')// &
            '    type(box_t), intent(in) :: arr(:)'//new_line('a')// &
            '    integer :: k, s'//new_line('a')// &
            '    s = 0'//new_line('a')// &
            '    do k = 1, size(arr)'//new_line('a')// &
            '      s = s + 1000 * size(arr(k)%v) + arr(k)%v(1)'//new_line('a')// &
            '    end do'//new_line('a')// &
            '    print *, size(arr), s'//new_line('a')// &
            '  end subroutine show'//new_line('a')// &
            'end program main'

        test_assumed_shape_rank1 = expect_output( &
            source, '           3        6600'//new_line('a'), &
            '/tmp/ffc_derived_alloc_arrayarg_as1')
    end function test_assumed_shape_rank1

    logical function test_explicit_shape_rank1()
        ! An explicit-shape dummy borrows the same storage: component values
        ! written by the caller are visible, and a write through an INTENT(OUT)
        ! element component reaches the caller's heap block.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: box_t'//new_line('a')// &
            '    integer, allocatable :: v(:)'//new_line('a')// &
            '  end type box_t'//new_line('a')// &
            '  type(box_t) :: b(2)'//new_line('a')// &
            '  allocate(b(1)%v(4))'//new_line('a')// &
            '  allocate(b(2)%v(2))'//new_line('a')// &
            '  b(1)%v(1) = 11'//new_line('a')// &
            '  b(2)%v(1) = 22'//new_line('a')// &
            '  call touch(b)'//new_line('a')// &
            '  if (b(2)%v(2) /= 99) error stop 3'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine touch(arr)'//new_line('a')// &
            '    type(box_t), intent(inout) :: arr(2)'//new_line('a')// &
            '    if (size(arr(1)%v) /= 4) error stop 1'//new_line('a')// &
            '    if (arr(1)%v(1) + arr(2)%v(1) /= 33) error stop 2'// &
            new_line('a')// &
            '    arr(2)%v(2) = 99'//new_line('a')// &
            '  end subroutine touch'//new_line('a')// &
            'end program main'

        test_explicit_shape_rank1 = expect_exit_status( &
            source, 0, '/tmp/ffc_derived_alloc_arrayarg_es1')
    end function test_explicit_shape_rank1

    logical function test_assumed_shape_rank2()
        ! A rank-2 assumed-shape dummy linearises its subscripts over the same
        ! contiguous element block.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: box_t'//new_line('a')// &
            '    integer, allocatable :: v(:)'//new_line('a')// &
            '  end type box_t'//new_line('a')// &
            '  type(box_t) :: b(2, 3)'//new_line('a')// &
            '  integer :: i, j'//new_line('a')// &
            '  do j = 1, 3'//new_line('a')// &
            '    do i = 1, 2'//new_line('a')// &
            '      allocate(b(i, j)%v(3))'//new_line('a')// &
            '      b(i, j)%v(1) = 10 * i + j'//new_line('a')// &
            '    end do'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  call show(b)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine show(arr)'//new_line('a')// &
            '    type(box_t), intent(in) :: arr(:, :)'//new_line('a')// &
            '    integer :: p, s'//new_line('a')// &
            '    s = 0'//new_line('a')// &
            '    do p = 1, 2'//new_line('a')// &
            '      s = s + arr(p, 3)%v(1) + size(arr(p, 3)%v)'// &
            new_line('a')// &
            '    end do'//new_line('a')// &
            '    print *, size(arr), s'//new_line('a')// &
            '  end subroutine show'//new_line('a')// &
            'end program main'

        test_assumed_shape_rank2 = expect_output( &
            source, '           6          42'//new_line('a'), &
            '/tmp/ffc_derived_alloc_arrayarg_as2')
    end function test_assumed_shape_rank2

    logical function test_reject_type_mismatch()
        ! The dummy borrows the actual's slot layout, so a different declared
        ! type is a diagnostic, never a reinterpretation.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: box_t'//new_line('a')// &
            '    integer, allocatable :: v(:)'//new_line('a')// &
            '  end type box_t'//new_line('a')// &
            '  type :: other_t'//new_line('a')// &
            '    integer :: q'//new_line('a')// &
            '  end type other_t'//new_line('a')// &
            '  type(other_t) :: b(2)'//new_line('a')// &
            '  call show(b)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine show(arr)'//new_line('a')// &
            '    type(box_t), intent(in) :: arr(:)'//new_line('a')// &
            '    print *, size(arr)'//new_line('a')// &
            '  end subroutine show'//new_line('a')// &
            'end program main'

        test_reject_type_mismatch = expect_error_contains( &
            source, 'Type mismatch in argument to show', &
            '/tmp/ffc_derived_alloc_arrayarg_type')
    end function test_reject_type_mismatch

    logical function test_reject_rank_mismatch()
        ! A rank-1 actual cannot supply a rank-2 assumed-shape dummy's shape.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: box_t'//new_line('a')// &
            '    integer, allocatable :: v(:)'//new_line('a')// &
            '  end type box_t'//new_line('a')// &
            '  type(box_t) :: b(2)'//new_line('a')// &
            '  call show(b)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine show(arr)'//new_line('a')// &
            '    type(box_t), intent(in) :: arr(:, :)'//new_line('a')// &
            '    print *, size(arr)'//new_line('a')// &
            '  end subroutine show'//new_line('a')// &
            'end program main'

        test_reject_rank_mismatch = expect_error_contains( &
            source, 'assumed-shape derived array', &
            '/tmp/ffc_derived_alloc_arrayarg_rank')
    end function test_reject_rank_mismatch

    logical function test_reject_intent_in_mutation()
        ! Writing an element's component through an INTENT(IN) dummy is a
        ! variable definition context on the dummy itself.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: box_t'//new_line('a')// &
            '    integer, allocatable :: v(:)'//new_line('a')// &
            '  end type box_t'//new_line('a')// &
            '  type(box_t) :: b(2)'//new_line('a')// &
            '  allocate(b(1)%v(1))'//new_line('a')// &
            '  call bad(b)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine bad(arr)'//new_line('a')// &
            '    type(box_t), intent(in) :: arr(:)'//new_line('a')// &
            '    arr(1)%v(1) = 5'//new_line('a')// &
            '  end subroutine bad'//new_line('a')// &
            'end program main'

        test_reject_intent_in_mutation = expect_error_contains( &
            source, 'INTENT(IN) dummy argument ''arr''', &
            '/tmp/ffc_derived_alloc_arrayarg_intent')
    end function test_reject_intent_in_mutation

end program test_session_derived_alloc_array_dummy_compiler
