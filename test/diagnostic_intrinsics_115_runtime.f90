program diagnostic_intrinsics_115_runtime
    real(4) :: arr1(3) = [123.41_4, 4.23_4, -31.0_4]
    real(8) :: arr2(3) = [123.41_8, 4.23_8, -31.0_8]
    real(4) :: out1(3)
    real(8) :: out2(3)
    out1 = anint(arr1)
    out2 = anint(arr2)
    if (any(out1 /= [123.0_4, 4.0_4, -31.0_4])) error stop
    if (any(out2 /= [123.0_8, 4.0_8, -31.0_8])) error stop
    print *, anint(arr1)
    if (any(anint(arr1) /= [123, 4, -31])) error stop
    print *, anint(arr2)
end program diagnostic_intrinsics_115_runtime
