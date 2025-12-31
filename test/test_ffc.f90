program test_ffc
    use ffc, only: ffc_version
    implicit none
    if (ffc_version /= "retired") error stop "unexpected version"
    print *, "PASS: ffc is retired"
end program test_ffc
