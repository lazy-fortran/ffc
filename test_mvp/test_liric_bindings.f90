program test_liric_bindings
    use liric_bindings, only: liric_compiler_t, liric_create
    implicit none

    type(liric_compiler_t) :: compiler
    character(len=:), allocatable :: error_msg
    logical :: ok

    print *, '=== LIRIC binding tests ==='

    call liric_create(compiler, error_msg)
    if (len_trim(error_msg) > 0) then
        print *, 'FAIL: create returned ', trim(error_msg)
        stop 1
    end if
    if (.not. compiler%is_open()) then
        print *, 'FAIL: compiler handle was not opened'
        stop 1
    end if
    call compiler%destroy()
    if (compiler%is_open()) then
        print *, 'FAIL: compiler handle was not closed'
        stop 1
    end if

    call liric_create(compiler, error_msg)
    ok = compiler%feed_ll('define i32 @main( {', error_msg)
    call compiler%destroy()
    if (ok) then
        print *, 'FAIL: invalid LLVM IR was accepted'
        stop 1
    end if
    if (len_trim(error_msg) == 0) then
        print *, 'FAIL: invalid LLVM IR returned no error message'
        stop 1
    end if

    print *, 'PASS: LIRIC binding tests'
end program test_liric_bindings
