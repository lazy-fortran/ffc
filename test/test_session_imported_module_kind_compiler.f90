program test_session_imported_module_kind_compiler
    ! A module kind parameter must be available while its derived types are
    ! laid out, so a separately compiled user can import both the parameter and
    ! the type. Compare the complete two-object behavior against gfortran.
    implicit none

    character(len=*), parameter :: root = '/tmp/ffc_imported_module_kind'
    character(len=*), parameter :: module_source = root//'/module.f90'
    character(len=*), parameter :: child_source = root//'/child.f90'
    character(len=*), parameter :: module_object = root//'/module.o'
    character(len=*), parameter :: ffc_executable = root//'/child.ffc'
    character(len=*), parameter :: gfortran_executable = root//'/child.gf'
    character(len=*), parameter :: ffc_output = root//'/child.ffc.out'
    character(len=*), parameter :: gfortran_output = root//'/child.gf.out'

    print *, '=== imported module kind parameter regression ==='
    call execute_command_line('rm -rf '//root)
    call execute_command_line('mkdir -p '//root)

    if (.not. write_source(module_source, &
            'module ffc_imported_kind_module'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer, parameter :: ilp = 4'//new_line('a')// &
            '  type :: state_type'//new_line('a')// &
            '    integer(ilp) :: state = 0'//new_line('a')// &
            '  end type state_type'//new_line('a')// &
            'end module ffc_imported_kind_module')) stop 1

    if (.not. write_source(child_source, &
            'module ffc_imported_kind_child'//new_line('a')// &
            '  use ffc_imported_kind_module, only: ilp, state_type'// &
            new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type, extends(state_type) :: child_type'//new_line('a')// &
            '  end type child_type'//new_line('a')// &
            'contains'//new_line('a')// &
            '  pure function state_message(flag) result(message)'//new_line('a')// &
            '    integer(ilp), intent(in) :: flag'//new_line('a')// &
            '    character(len=:), allocatable :: message'//new_line('a')// &
            '    if (flag == 0) then'//new_line('a')// &
            "      message = 'ok'"//new_line('a')// &
            '    else'//new_line('a')// &
            "      message = 'error'"//new_line('a')// &
            '    end if'//new_line('a')// &
            '  end function state_message'//new_line('a')// &
            'end module ffc_imported_kind_child'//new_line('a')// &
            'program imported_module_kind_main'//new_line('a')// &
            '  use ffc_imported_kind_child, only: child_type, state_message'// &
            new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type(child_type) :: value'//new_line('a')// &
            '  character(len=:), allocatable :: message'//new_line('a')// &
            '  value%state = 0'//new_line('a')// &
            '  message = state_message(value%state)'//new_line('a')// &
            "  if (message /= 'ok') error stop 1"//new_line('a')// &
            '  print *, message'//new_line('a')// &
            'end program imported_module_kind_main')) stop 1

    if (.not. compile_and_compare()) stop 1
    call execute_command_line('rm -rf '//root)
    print *, 'PASS: imported module kind parameter matches gfortran'

contains

    logical function compile_and_compare() result(ok)
        integer :: cmd_stat, exit_stat

        ok = .false.
        call execute_command_line( &
            "sh -c 'ffc=$(ls -t build/*/app/ffc build/fo/bin/ffc "// &
            "2>/dev/null | head -n 1); test -x $ffc || exit 90; $ffc -c "// &
            module_source//" -o "//module_object//" || exit 91; $ffc "// &
            child_source//" -I "//root//" "//module_object//" -o "// &
            ffc_executable//" || exit 92'", &
            exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0 .or. exit_stat /= 0) then
            print *, 'FAIL: ffc separate compilation failed, code ', exit_stat
            return
        end if

        call execute_command_line('gfortran -w -J '//root//' '//module_source// &
            ' '//child_source//' -o '//gfortran_executable, &
            exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0 .or. exit_stat /= 0) then
            print *, 'FAIL: gfortran oracle failed, code ', exit_stat
            return
        end if

        call execute_command_line(ffc_executable//' > '//ffc_output// &
            ' 2>&1', exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0 .or. exit_stat /= 0) then
            print *, 'FAIL: ffc executable failed, code ', exit_stat
            return
        end if
        call execute_command_line(gfortran_executable//' > '//gfortran_output// &
            ' 2>&1', exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0 .or. exit_stat /= 0) then
            print *, 'FAIL: gfortran oracle executable failed, code ', exit_stat
            return
        end if

        call execute_command_line('diff -u '//gfortran_output//' '//ffc_output// &
            ' > /dev/null 2>&1', exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0 .or. exit_stat /= 0) then
            print *, 'FAIL: ffc output differs from gfortran'
            return
        end if
        ok = .true.
    end function compile_and_compare

    logical function write_source(path, contents) result(ok)
        character(len=*), intent(in) :: path
        character(len=*), intent(in) :: contents
        integer :: io_stat, unit

        open (newunit=unit, file=path, status='replace', action='write', &
              iostat=io_stat)
        if (io_stat /= 0) then
            ok = .false.
            return
        end if
        write (unit, '(A)', iostat=io_stat) contents
        close (unit)
        ok = io_stat == 0
    end function write_source

end program test_session_imported_module_kind_compiler
