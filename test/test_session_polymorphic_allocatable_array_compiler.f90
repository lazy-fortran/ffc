program test_session_polymorphic_allocatable_array_compiler
    use ffc_test_support, only: expect_output
    implicit none

    print *, '=== polymorphic allocatable array allocation test ==='
    if (.not. test_allocate_extension_array()) stop 1
    print *, 'PASS: polymorphic allocatable array allocation'

contains

    logical function test_allocate_extension_array()
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  type :: base_t'//new_line('a')// &
            '    integer :: x'//new_line('a')// &
            '  end type base_t'//new_line('a')// &
            '  type, extends(base_t) :: ext_t'//new_line('a')// &
            '    integer :: y'//new_line('a')// &
            '  end type ext_t'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  class(base_t), allocatable :: a(:)'//new_line('a')// &
            '  allocate(ext_t :: a(2))'//new_line('a')// &
            '  select type (a)'//new_line('a')// &
            '  type is (ext_t)'//new_line('a')// &
            '    a(1)%x = 10'//new_line('a')// &
            '    a(1)%y = 11'//new_line('a')// &
            '    a(2)%x = 20'//new_line('a')// &
            '    a(2)%y = 21'//new_line('a')// &
            '    print *, size(a), a(1)%x, a(1)%y, a(2)%x, a(2)%y'//new_line('a')// &
            '  end select'//new_line('a')// &
            '  deallocate(a)'//new_line('a')// &
            'end program main'

        test_allocate_extension_array = expect_output(source, &
            '           2          10          11          20          21'// &
            new_line('a'), '/tmp/ffc_poly_alloc_array_ext')
    end function test_allocate_extension_array

end program test_session_polymorphic_allocatable_array_compiler
