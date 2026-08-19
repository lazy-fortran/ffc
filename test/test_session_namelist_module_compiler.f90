program test_session_namelist_module_compiler
    ! Reduced issue #628 leaf: a fixed-length module character member is
    ! exported with its NAMELIST groups, and a same-name program group merges
    ! its additional scalar members. The witness emits no stdout, so its
    ! pinned behavioral oracle is empty stdout plus normal exit status. The two input
    ! records carry the witness payload; formatted character output and an
    ! explicit IOSTAT declaration keep this test independent of unrelated
    ! list-directed character-write and implicit-name paths.
    use ffc_test_support, only: expect_output
    implicit none

    character(len=*), parameter :: source = &
        'module global'//new_line('a')// &
        '  character(4) :: aa'//new_line('a')// &
        '  integer :: ii'//new_line('a')// &
        '  real :: rr'//new_line('a')// &
        '  namelist /nml1/ aa, ii, rr'//new_line('a')// &
        '  namelist /nml2/ aa'//new_line('a')// &
        'end module global'//new_line('a')// &
        'program namelist_use'//new_line('a')// &
        '  use global'//new_line('a')// &
        '  real :: rrr'//new_line('a')// &
        '  integer :: i'//new_line('a')// &
        '  namelist /nml2/ ii, rrr'//new_line('a')// &
        '  open (10, status="scratch")'//new_line('a')// &
        '  write (10,''(A)'') "&NML1 aa=''lmno'' ii=1 rr=2.5 /"'// &
        new_line('a')// &
        '  write (10,''(A)'') "&NML2 aa=''pqrs'' ii=2 rrr=3.5 /"'// &
        new_line('a')// &
        '  rewind (10)'//new_line('a')// &
        '  read (10,nml=nml1,iostat=i)'//new_line('a')// &
        '  if ((i.ne.0).or.(aa.ne."lmno").or.(ii.ne.1).or.(rr.ne.2.5)) '// &
        'stop 1'//new_line('a')// &
        '  read (10,nml=nml2,iostat=i)'//new_line('a')// &
        '  if ((i.ne.0).or.(aa.ne."pqrs").or.(ii.ne.2).or.(rrr.ne.3.5)) '// &
        'stop 2'//new_line('a')// &
        '  close (10)'//new_line('a')// &
        'end program namelist_use'

    if (.not. expect_output(source, '', '/tmp/ffc_namelist_module_scalar')) &
        stop 1
end program test_session_namelist_module_compiler
