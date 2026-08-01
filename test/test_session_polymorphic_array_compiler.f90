program test_session_polymorphic_array_compiler
    use ffc_test_support, only: expect_output, expect_error_contains
    implicit none

    logical :: all_passed

    print *, '=== polymorphic array passing tests ==='

    all_passed = .true.
    if (.not. test_base_actual_unchanged()) all_passed = .false.
    if (.not. test_extension_actual_uses_concrete_stride()) all_passed = .false.
    if (.not. test_extension_components_are_addressable()) all_passed = .false.
    if (.not. test_rank2_extension_actual()) all_passed = .false.
    if (.not. test_monomorphic_dummy_is_unaffected()) all_passed = .false.
    if (.not. test_rank_mismatch_is_rejected()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: polymorphic array passing'

contains

    character(len=:) function hierarchy() result(text)
        allocatable :: text
        text = 'module m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type :: base_t'//new_line('a')// &
            '    integer :: x'//new_line('a')// &
            '  end type base_t'//new_line('a')// &
            '  type, extends(base_t) :: ext_t'//new_line('a')// &
            '    integer :: y'//new_line('a')// &
            '  end type ext_t'//new_line('a')// &
            'contains'//new_line('a')
    end function hierarchy

    logical function test_base_actual_unchanged()
        ! Negative control: a base actual already worked and must keep working.
        character(len=:), allocatable :: source

        source = hierarchy()// &
            '  subroutine show(a)'//new_line('a')// &
            '    class(base_t), intent(in) :: a(:)'//new_line('a')// &
            '    print *, size(a), a(1)%x, a(2)%x, a(3)%x'//new_line('a')// &
            '  end subroutine show'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type(base_t) :: b(3)'//new_line('a')// &
            '  b(1)%x = 1'//new_line('a')// &
            '  b(2)%x = 2'//new_line('a')// &
            '  b(3)%x = 3'//new_line('a')// &
            '  call show(b)'//new_line('a')// &
            'end program main'

        test_base_actual_unchanged = expect_output(source, &
            '           3           1           2           3'// &
            new_line('a'), '/tmp/ffc_polyarr_base')
    end function test_base_actual_unchanged

    logical function test_extension_actual_uses_concrete_stride()
        ! The heart of the slice: an extension actual has a wider element than
        ! the dummy's declared type, so the callee must step by the dynamic
        ! element size. Stepping by the declared size reads the extension's
        ! own component of element 1 as element 2's inherited component.
        character(len=:), allocatable :: source

        source = hierarchy()// &
            '  subroutine show(a)'//new_line('a')// &
            '    class(base_t), intent(in) :: a(:)'//new_line('a')// &
            '    print *, size(a), a(1)%x, a(2)%x, a(3)%x'//new_line('a')// &
            '  end subroutine show'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type(ext_t) :: e(3)'//new_line('a')// &
            '  e(1)%x = 10'//new_line('a')// &
            '  e(1)%y = 11'//new_line('a')// &
            '  e(2)%x = 20'//new_line('a')// &
            '  e(2)%y = 21'//new_line('a')// &
            '  e(3)%x = 30'//new_line('a')// &
            '  e(3)%y = 31'//new_line('a')// &
            '  call show(e)'//new_line('a')// &
            'end program main'

        test_extension_actual_uses_concrete_stride = expect_output(source, &
            '           3          10          20          30'// &
            new_line('a'), '/tmp/ffc_polyarr_stride')
    end function test_extension_actual_uses_concrete_stride

    logical function test_extension_components_are_addressable()
        ! Writing through the declared-type prefix of one element must not
        ! touch the neighbouring element or the extension's own component.
        character(len=:), allocatable :: source

        source = hierarchy()// &
            '  subroutine bump(a)'//new_line('a')// &
            '    class(base_t), intent(inout) :: a(:)'//new_line('a')// &
            '    a(2)%x = a(2)%x + 100'//new_line('a')// &
            '  end subroutine bump'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type(ext_t) :: e(3)'//new_line('a')// &
            '  e(1)%x = 1'//new_line('a')// &
            '  e(1)%y = 2'//new_line('a')// &
            '  e(2)%x = 3'//new_line('a')// &
            '  e(2)%y = 4'//new_line('a')// &
            '  e(3)%x = 5'//new_line('a')// &
            '  e(3)%y = 6'//new_line('a')// &
            '  call bump(e)'//new_line('a')// &
            '  print *, e(1)%x, e(1)%y, e(2)%x, e(2)%y, e(3)%x, e(3)%y'// &
            new_line('a')// &
            'end program main'

        test_extension_components_are_addressable = expect_output(source, &
            '           1           2         103           4           5'// &
            '           6'//new_line('a'), '/tmp/ffc_polyarr_write')
    end function test_extension_components_are_addressable

    logical function test_rank2_extension_actual()
        ! Column-major addressing scales by the dynamic element size in every
        ! dimension, not only the first.
        character(len=:), allocatable :: source

        source = hierarchy()// &
            '  subroutine show(a)'//new_line('a')// &
            '    class(base_t), intent(in) :: a(:,:)'//new_line('a')// &
            '    print *, a(1,1)%x, a(2,1)%x, a(1,2)%x, a(2,2)%x'//new_line('a')// &
            '  end subroutine show'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type(ext_t) :: e(2,2)'//new_line('a')// &
            '  e(1,1)%x = 11'//new_line('a')// &
            '  e(1,1)%y = 1'//new_line('a')// &
            '  e(2,1)%x = 21'//new_line('a')// &
            '  e(2,1)%y = 2'//new_line('a')// &
            '  e(1,2)%x = 12'//new_line('a')// &
            '  e(1,2)%y = 3'//new_line('a')// &
            '  e(2,2)%x = 22'//new_line('a')// &
            '  e(2,2)%y = 4'//new_line('a')// &
            '  call show(e)'//new_line('a')// &
            'end program main'

        test_rank2_extension_actual = expect_output(source, &
            '          11          21          12          22'// &
            new_line('a'), '/tmp/ffc_polyarr_rank2')
    end function test_rank2_extension_actual

    logical function test_monomorphic_dummy_is_unaffected()
        ! Negative control: a type(t) assumed-shape dummy is not polymorphic,
        ! keeps its compile-time element size, and is unchanged by this slice.
        character(len=:), allocatable :: source

        source = hierarchy()// &
            '  subroutine show(a)'//new_line('a')// &
            '    type(ext_t), intent(in) :: a(:)'//new_line('a')// &
            '    print *, size(a), a(1)%x, a(2)%y'//new_line('a')// &
            '  end subroutine show'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type(ext_t) :: e(2)'//new_line('a')// &
            '  e(1)%x = 4'//new_line('a')// &
            '  e(1)%y = 5'//new_line('a')// &
            '  e(2)%x = 6'//new_line('a')// &
            '  e(2)%y = 7'//new_line('a')// &
            '  call show(e)'//new_line('a')// &
            'end program main'

        test_monomorphic_dummy_is_unaffected = expect_output(source, &
            '           2           4           7'//new_line('a'), &
            '/tmp/ffc_polyarr_mono')
    end function test_monomorphic_dummy_is_unaffected

    logical function test_rank_mismatch_is_rejected()
        ! A rank-2 actual for a rank-1 polymorphic dummy is rejected rather
        ! than miscompiled into a wrong stride.
        character(len=:), allocatable :: source

        source = hierarchy()// &
            '  subroutine show(a)'//new_line('a')// &
            '    class(base_t), intent(in) :: a(:)'//new_line('a')// &
            '    print *, size(a)'//new_line('a')// &
            '  end subroutine show'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type(ext_t) :: e(2,2)'//new_line('a')// &
            '  e(1,1)%x = 1'//new_line('a')// &
            '  call show(e)'//new_line('a')// &
            'end program main'

        test_rank_mismatch_is_rejected = expect_error_contains(source, &
            'assumed-shape derived', '/tmp/ffc_polyarr_rankmismatch')
    end function test_rank_mismatch_is_rejected

end program test_session_polymorphic_array_compiler
