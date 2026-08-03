program diagnostic_intrinsics_115_param_f64
    real(8), parameter :: ar2(3) = anint([123.41_8, 4.23_8, -31.0_8])
    if (any(ar2 /= [123.0_8, 4.0_8, -31.0_8])) error stop
    print *, ar2
end program diagnostic_intrinsics_115_param_f64
