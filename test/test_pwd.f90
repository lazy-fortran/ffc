program test_pwd
    implicit none
    character(len=256) :: cwd
    call getcwd(cwd)
    print '(A)', trim(cwd)
end program test_pwd
