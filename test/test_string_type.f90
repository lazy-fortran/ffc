program test_string_type
    use string_type
    implicit none

    type(string_t) :: str1, str2
    logical :: test_passed

    test_passed = .true.

    ! Test: Creating string_t with allocated string
    str1%s = "Hello, World!"
    if (.not. allocated(str1%s)) then
        print *, "FAIL: str1%s not allocated"
        test_passed = .false.
    else if (str1%s /= "Hello, World!") then
        print *, "FAIL: str1%s has wrong value"
        test_passed = .false.
    end if

    ! Test: Creating empty string_t
    if (allocated(str2%s)) then
        print *, "FAIL: str2%s should not be allocated initially"
        test_passed = .false.
    end if

    ! Test: Assigning to string_t
    str2%s = "Test string"
    if (.not. allocated(str2%s)) then
        print *, "FAIL: str2%s not allocated after assignment"
        test_passed = .false.
    else if (str2%s /= "Test string") then
        print *, "FAIL: str2%s has wrong value after assignment"
        test_passed = .false.
    end if

    ! Test: Deallocating string
    deallocate(str1%s)
    if (allocated(str1%s)) then
        print *, "FAIL: str1%s still allocated after deallocation"
        test_passed = .false.
    end if

    if (test_passed) then
        print *, "All tests passed!"
    else
        print *, "Some tests failed!"
        stop 1
    end if

end program test_string_type