program test_ffc
    use ffc, only: ffc_version
    implicit none
    if (ffc_version /= "0.1.0-mlir") error stop "unexpected version"
    print *, "PASS: ffc module loads correctly"
end program test_ffc
