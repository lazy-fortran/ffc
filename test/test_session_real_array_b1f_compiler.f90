program test_session_real_array_b1f_compiler
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, &
        compile_frontend_from_string, INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    logical :: all_passed

    print *, '=== B1f: real-element array sections and intrinsics ==='

    all_passed = .true.

    ! real(4) 2-D element access
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  real :: a(2,2)'//new_line('a')// &
        '  a(1,1) = 1.0'//new_line('a')// &
        '  a(2,1) = 2.0'//new_line('a')// &
        '  a(1,2) = 3.0'//new_line('a')// &
        '  a(2,2) = 4.0'//new_line('a')// &
        '  print *, a(1,1), a(2,1), a(1,2), a(2,2)'//new_line('a')// &
        'end program main', 'b1f_f32_rank2_elem')) all_passed = .false.

    ! real(4) sum/maxval/minval on 2-D
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  real :: a(2,2)'//new_line('a')// &
        '  a(1,1) = 1.0'//new_line('a')// &
        '  a(2,1) = 2.0'//new_line('a')// &
        '  a(1,2) = 3.0'//new_line('a')// &
        '  a(2,2) = 4.0'//new_line('a')// &
        '  print *, sum(a), maxval(a), minval(a)'//new_line('a')// &
        'end program main', 'b1f_f32_rank2_reductions')) all_passed = .false.

    ! real(4) 1-D section print
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  real :: a(4)'//new_line('a')// &
        '  a = [10.0, 20.0, 30.0, 40.0]'//new_line('a')// &
        '  print *, a(2:3)'//new_line('a')// &
        'end program main', 'b1f_f32_section_print')) all_passed = .false.

    ! real(4) 1-D section assignment
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  real :: a(4), b(2)'//new_line('a')// &
        '  a = [10.0, 20.0, 30.0, 40.0]'//new_line('a')// &
        '  b = a(2:3)'//new_line('a')// &
        '  print *, b(1), b(2)'//new_line('a')// &
        'end program main', 'b1f_f32_section_assign')) all_passed = .false.

    ! real(4) 1-D strided section print
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  real :: a(4)'//new_line('a')// &
        '  a = [10.0, 20.0, 30.0, 40.0]'//new_line('a')// &
        '  print *, a(1:4:2)'//new_line('a')// &
        'end program main', 'b1f_f32_section_stride')) all_passed = .false.

    ! real(4) column section (2-D)
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  real :: a(2,3), b(2)'//new_line('a')// &
        '  integer :: j'//new_line('a')// &
        '  do j = 1, 3'//new_line('a')// &
        '    a(1,j) = real(j)'//new_line('a')// &
        '    a(2,j) = real(j) * 10.0'//new_line('a')// &
        '  end do'//new_line('a')// &
        '  b = a(:,2)'//new_line('a')// &
        '  print *, b(1), b(2)'//new_line('a')// &
        'end program main', 'b1f_f32_col_section')) all_passed = .false.

    ! real(4) matmul
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  real :: a(2,2), b(2,2), c(2,2)'//new_line('a')// &
        '  a(1,1) = 1.0; a(2,1) = 2.0'//new_line('a')// &
        '  a(1,2) = 3.0; a(2,2) = 4.0'//new_line('a')// &
        '  b(1,1) = 1.0; b(2,1) = 0.0'//new_line('a')// &
        '  b(1,2) = 0.0; b(2,2) = 1.0'//new_line('a')// &
        '  c = matmul(a, b)'//new_line('a')// &
        '  print *, c(1,1), c(2,1), c(1,2), c(2,2)'//new_line('a')// &
        'end program main', 'b1f_f32_matmul')) all_passed = .false.

    ! real(4) transpose
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  real :: a(2,3), b(3,2)'//new_line('a')// &
        '  a(1,1) = 1.0; a(2,1) = 2.0'//new_line('a')// &
        '  a(1,2) = 3.0; a(2,2) = 4.0'//new_line('a')// &
        '  a(1,3) = 5.0; a(2,3) = 6.0'//new_line('a')// &
        '  b = transpose(a)'//new_line('a')// &
        '  print *, b(1,1), b(2,1), b(3,1)'//new_line('a')// &
        '  print *, b(1,2), b(2,2), b(3,2)'//new_line('a')// &
        'end program main', 'b1f_f32_transpose')) all_passed = .false.

    ! real(4) reshape
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  real :: a(6), b(2,3)'//new_line('a')// &
        '  a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]'//new_line('a')// &
        '  b = reshape(a, [2, 3])'//new_line('a')// &
        '  print *, b(1,1), b(2,1), b(1,2), b(2,2), b(1,3), b(2,3)'//new_line('a')// &
        'end program main', 'b1f_f32_reshape')) all_passed = .false.

    ! real(4) dot_product
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  real :: a(3), b(3)'//new_line('a')// &
        '  a = [1.0, 2.0, 3.0]'//new_line('a')// &
        '  b = [4.0, 5.0, 6.0]'//new_line('a')// &
        '  print *, dot_product(a, b)'//new_line('a')// &
        'end program main', 'b1f_f32_dot_product')) all_passed = .false.

    ! real(8) 2-D element access
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  real(8) :: a(2,2)'//new_line('a')// &
        '  a(1,1) = 1.0d0'//new_line('a')// &
        '  a(2,1) = 2.0d0'//new_line('a')// &
        '  a(1,2) = 3.0d0'//new_line('a')// &
        '  a(2,2) = 4.0d0'//new_line('a')// &
        '  print *, a(1,1), a(2,1), a(1,2), a(2,2)'//new_line('a')// &
        'end program main', 'b1f_f64_rank2_elem')) all_passed = .false.

    ! real(8) 1-D section print
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  real(8) :: a(4)'//new_line('a')// &
        '  a = [10.0d0, 20.0d0, 30.0d0, 40.0d0]'//new_line('a')// &
        '  print *, a(2:3)'//new_line('a')// &
        'end program main', 'b1f_f64_section_print')) all_passed = .false.

    ! real(8) section assignment
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  real(8) :: a(4), b(2)'//new_line('a')// &
        '  a = [10.0d0, 20.0d0, 30.0d0, 40.0d0]'//new_line('a')// &
        '  b = a(2:3)'//new_line('a')// &
        '  print *, b(1), b(2)'//new_line('a')// &
        'end program main', 'b1f_f64_section_assign')) all_passed = .false.

    ! real(8) matmul
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  real(8) :: a(2,2), b(2,1), c(2,1)'//new_line('a')// &
        '  a(1,1) = 1.0d0; a(2,1) = 2.0d0'//new_line('a')// &
        '  a(1,2) = 3.0d0; a(2,2) = 4.0d0'//new_line('a')// &
        '  b(1,1) = 1.0d0'//new_line('a')// &
        '  b(2,1) = 1.0d0'//new_line('a')// &
        '  c = matmul(a, b)'//new_line('a')// &
        '  print *, c(1,1), c(2,1)'//new_line('a')// &
        'end program main', 'b1f_f64_matmul')) all_passed = .false.

    ! real(8) transpose
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  real(8) :: a(2,2), b(2,2)'//new_line('a')// &
        '  a(1,1) = 1.0d0; a(2,1) = 2.0d0'//new_line('a')// &
        '  a(1,2) = 3.0d0; a(2,2) = 4.0d0'//new_line('a')// &
        '  b = transpose(a)'//new_line('a')// &
        '  print *, b(1,1), b(2,1), b(1,2), b(2,2)'//new_line('a')// &
        'end program main', 'b1f_f64_transpose')) all_passed = .false.

    ! real(8) reshape
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  real(8) :: a(4), b(2,2)'//new_line('a')// &
        '  a = [1.0d0, 2.0d0, 3.0d0, 4.0d0]'//new_line('a')// &
        '  b = reshape(a, [2, 2])'//new_line('a')// &
        '  print *, b(1,1), b(2,1), b(1,2), b(2,2)'//new_line('a')// &
        'end program main', 'b1f_f64_reshape')) all_passed = .false.

    ! real(8) dot_product
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  real(8) :: a(3), b(3)'//new_line('a')// &
        '  a = [1.0d0, 2.0d0, 3.0d0]'//new_line('a')// &
        '  b = [4.0d0, 5.0d0, 6.0d0]'//new_line('a')// &
        '  print *, dot_product(a, b)'//new_line('a')// &
        'end program main', 'b1f_f64_dot_product')) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: B1f real array sections and intrinsics'

contains

    logical function matches_gfortran(source, stem)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: stem
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result
        character(len=:), allocatable :: error_msg
        character(len=:), allocatable :: base, src, exe, ref, ffc_out, ref_out
        integer :: unit, exit_stat, status

        matches_gfortran = .false.
        base = '/tmp/ffc_b1f_'//stem
        src = base//'.f90'
        exe = base//'.ffc'
        ref = base//'.gf'
        ffc_out = base//'.ffc.out'
        ref_out = base//'.gf.out'

        options = compiler_frontend_options_t()
        options%run_semantics = .true.
        options%input_mode = INPUT_MODE_STANDARD
        call compile_frontend_from_string(source, frontend_result, options)
        if (.not. frontend_result%success()) then
            print *, 'FAIL[', stem, ']: FortFront rejected source: ', &
                trim(frontend_result%diagnostic_text)
            return
        end if

        call lower_program_to_liric_exe(frontend_result%arena, &
            frontend_result%root_index, exe, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL[', stem, ']: ffc lowering failed: ', trim(error_msg)
            return
        end if

        open (newunit=unit, file=src, status='replace', action='write')
        write (unit, '(A)') source
        close (unit)
        call execute_command_line('gfortran -w '//src//' -o '//ref, &
            exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL[', stem, ']: gfortran rejected source'
            return
        end if

        call execute_command_line(exe//' > '//ffc_out, exitstat=exit_stat)
        call execute_command_line(ref//' > '//ref_out, exitstat=exit_stat)
        call execute_command_line('diff '//ffc_out//' '//ref_out// &
            ' > /dev/null 2>&1', exitstat=status)
        if (status /= 0) then
            print *, 'FAIL[', stem, ']: ffc output differs from gfortran'
            call execute_command_line('diff '//ffc_out//' '//ref_out)
        else
            matches_gfortran = .true.
        end if
        call execute_command_line('rm -f '//src//' '//exe//' '//ref//' '// &
            ffc_out//' '//ref_out)
    end function matches_gfortran

end program test_session_real_array_b1f_compiler
