program diagnostic_intrinsics_115_param_f32
    real(4), parameter :: ar1(3) = anint([123.41_4, 4.23_4, -31.0_4])
    if (any(ar1 /= [123.0_4, 4.0_4, -31.0_4])) error stop
    print *, ar1
end program diagnostic_intrinsics_115_param_f32
