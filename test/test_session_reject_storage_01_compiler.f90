program test_session_reject_storage_01_compiler
    ! #392: storage-association restrictions on COMMON, EQUIVALENCE, SAVE and
    ! BLOCK DATA. Each invalid form is rejected with its own source
    ! diagnostic, while the corrected neighbour still compiles and runs.
    use ffc_test_support, only: expect_cli_error_contains, expect_cli_no_error, &
        expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== storage association restriction rejection test ==='

    all_passed = .true.
    if (.not. test_common_nonsequence_derived_rejected()) all_passed = .false.
    if (.not. test_common_ultimate_allocatable_rejected()) all_passed = .false.
    if (.not. test_block_save_common_rejected()) all_passed = .false.
    if (.not. test_equivalence_bind_c_rejected()) all_passed = .false.
    if (.not. test_common_program_unit_name_rejected()) all_passed = .false.
    if (.not. test_common_save_conflict_rejected()) all_passed = .false.
    if (.not. test_block_data_object_not_in_common_rejected()) all_passed = .false.
    if (.not. test_sequence_common_accepted()) all_passed = .false.
    if (.not. test_block_save_local_accepted()) all_passed = .false.
    if (.not. test_equivalence_without_bind_c_accepted()) all_passed = .false.
    if (.not. test_distinct_common_name_accepted()) all_passed = .false.
    if (.not. test_common_without_save_accepted()) all_passed = .false.
    if (.not. test_block_data_common_object_accepted()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: storage association restrictions are enforced'

contains

    logical function test_common_nonsequence_derived_rejected()
        ! gfortran.dg/common_10.f90: common /c/ t3 with plain derived type.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type mytype3'//new_line('a')// &
            '    integer :: x'//new_line('a')// &
            '  end type mytype3'//new_line('a')// &
            '  type(mytype3) :: t3'//new_line('a')// &
            '  common /c/ t3'//new_line('a')// &
            'end program main'

        test_common_nonsequence_derived_rejected = expect_cli_error_contains( &
            source, 'has neither the SEQUENCE nor the BIND(C) attribute', &
            '/tmp/ffc_session_reject_storage01_nonseq')
    end function test_common_nonsequence_derived_rejected

    logical function test_common_ultimate_allocatable_rejected()
        ! gfortran.dg/common_10.f90: common /f/ t7, nested allocatable.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type mytype4'//new_line('a')// &
            '    sequence'//new_line('a')// &
            '    integer, allocatable, dimension(:) :: x'//new_line('a')// &
            '  end type mytype4'//new_line('a')// &
            '  type mytype7'//new_line('a')// &
            '    sequence'//new_line('a')// &
            '    type(mytype4) :: t'//new_line('a')// &
            '  end type mytype7'//new_line('a')// &
            '  type(mytype7) :: t7'//new_line('a')// &
            '  common /f/ t7'//new_line('a')// &
            'end program main'

        test_common_ultimate_allocatable_rejected = expect_cli_error_contains( &
            source, 'has an ultimate component that is allocatable', &
            '/tmp/ffc_session_reject_storage01_alloc')
    end function test_common_ultimate_allocatable_rejected

    logical function test_block_save_common_rejected()
        ! gfortran.dg/common_31.f90: SAVE /argmnt2/ inside a BLOCK construct.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real r'//new_line('a')// &
            '  common /argmnt2/ r'//new_line('a')// &
            '  block'//new_line('a')// &
            '    save /argmnt2/'//new_line('a')// &
            '  end block'//new_line('a')// &
            'end program main'

        test_block_save_common_rejected = expect_cli_error_contains( &
            source, 'not allowed in a BLOCK construct', &
            '/tmp/ffc_session_reject_storage01_blocksave')
    end function test_block_save_common_rejected

    logical function test_equivalence_bind_c_rejected()
        ! gfortran.dg/equiv_constraint_bind_c.f90.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: i1, i2'//new_line('a')// &
            '  bind(C) :: i2'//new_line('a')// &
            '  equivalence(i1,i2)'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            'end program main'

        test_equivalence_bind_c_rejected = expect_cli_error_contains( &
            source, 'conflicts with BIND(C) attribute', &
            '/tmp/ffc_session_reject_storage01_equiv')
    end function test_equivalence_bind_c_rejected

    logical function test_common_program_unit_name_rejected()
        ! gfortran.dg/pr103259.f90: the program unit name p is in COMMON.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  integer :: p'//new_line('a')// &
            '  common /c/ p'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program p'//new_line('a')// &
            '  use m'//new_line('a')// &
            'end program p'

        test_common_program_unit_name_rejected = expect_cli_error_contains( &
            source, 'cannot appear in a COMMON block', &
            '/tmp/ffc_session_reject_storage01_unitname')
    end function test_common_program_unit_name_rejected

    logical function test_common_save_conflict_rejected()
        ! gfortran.dg/save_common.f90.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, save :: x'//new_line('a')// &
            '  common /com/ x'//new_line('a')// &
            'end program main'

        test_common_save_conflict_rejected = expect_cli_error_contains( &
            source, 'conflicts with SAVE attribute', &
            '/tmp/ffc_session_reject_storage01_savecommon')
    end function test_common_save_conflict_rejected

    logical function test_block_data_object_not_in_common_rejected()
        ! gfortran.dg/uncommon_block_data_1.f90.
        character(len=*), parameter :: source = &
            'block data d'//new_line('a')// &
            '  integer i'//new_line('a')// &
            '  data i /1/'//new_line('a')// &
            'end block data'//new_line('a')// &
            'program main'//new_line('a')// &
            'end program main'

        test_block_data_object_not_in_common_rejected = expect_cli_error_contains( &
            source, 'must be in COMMON', &
            '/tmp/ffc_session_reject_storage01_blockdata')
    end function test_block_data_object_not_in_common_rejected

    logical function test_sequence_common_accepted()
        ! Corrected neighbour for both derived-type rules: a SEQUENCE type
        ! without an allocatable ultimate component is valid in COMMON, so
        ! neither storage diagnostic may fire. Lowering a derived-type COMMON
        ! object is a separate unimplemented feature, so what the compiler
        ! reports here is that unsupported-feature diagnostic; if either
        ! storage rule misfired this exact string would be replaced.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type mytype2'//new_line('a')// &
            '    sequence'//new_line('a')// &
            '    integer :: x'//new_line('a')// &
            '  end type mytype2'//new_line('a')// &
            '  type(mytype2) :: t2'//new_line('a')// &
            '  common /b/ t2'//new_line('a')// &
            '  t2%x = 5'//new_line('a')// &
            '  stop t2%x'//new_line('a')// &
            'end program main'

        test_sequence_common_accepted = expect_cli_error_contains( &
            source, 'unsupported scalar type', &
            '/tmp/ffc_session_storage01_seq_ok')
    end function test_sequence_common_accepted

    logical function test_block_save_local_accepted()
        ! Corrected neighbour: SAVE of a local entity inside BLOCK is fine.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real r'//new_line('a')// &
            '  common /argmnt2/ r'//new_line('a')// &
            '  block'//new_line('a')// &
            '    integer :: k'//new_line('a')// &
            '    save :: k'//new_line('a')// &
            '    k = 7'//new_line('a')// &
            '    stop k'//new_line('a')// &
            '  end block'//new_line('a')// &
            'end program main'

        test_block_save_local_accepted = expect_cli_no_error( &
            source, '/tmp/ffc_session_storage01_blocksave_ok_cli')
        if (.not. expect_exit_status(source, 7, &
                                     '/tmp/ffc_session_storage01_blocksave_ok')) &
            test_block_save_local_accepted = .false.
    end function test_block_save_local_accepted

    logical function test_equivalence_without_bind_c_accepted()
        ! Corrected neighbour: EQUIVALENCE of two ordinary variables.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: i1, i2'//new_line('a')// &
            '  equivalence(i1,i2)'//new_line('a')// &
            '  i1 = 9'//new_line('a')// &
            '  stop i2'//new_line('a')// &
            'end program main'

        test_equivalence_without_bind_c_accepted = expect_cli_no_error( &
            source, '/tmp/ffc_session_storage01_equiv_ok_cli')
        if (.not. expect_exit_status(source, 9, &
                                     '/tmp/ffc_session_storage01_equiv_ok')) &
            test_equivalence_without_bind_c_accepted = .false.
    end function test_equivalence_without_bind_c_accepted

    logical function test_distinct_common_name_accepted()
        ! Corrected neighbour: the COMMON object name differs from every
        ! program unit name.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  integer :: q'//new_line('a')// &
            '  common /c/ q'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program p'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  q = 4'//new_line('a')// &
            '  stop q'//new_line('a')// &
            'end program p'

        test_distinct_common_name_accepted = expect_cli_no_error( &
            source, '/tmp/ffc_session_storage01_unitname_ok_cli')
        if (.not. expect_exit_status(source, 4, &
                                     '/tmp/ffc_session_storage01_unitname_ok')) &
            test_distinct_common_name_accepted = .false.
    end function test_distinct_common_name_accepted

    logical function test_common_without_save_accepted()
        ! Corrected neighbour: drop SAVE from the COMMON object.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  common /com/ x'//new_line('a')// &
            '  x = 6'//new_line('a')// &
            '  stop x'//new_line('a')// &
            'end program main'

        test_common_without_save_accepted = expect_cli_no_error( &
            source, '/tmp/ffc_session_storage01_save_ok_cli')
        if (.not. expect_exit_status(source, 6, &
                                     '/tmp/ffc_session_storage01_save_ok')) &
            test_common_without_save_accepted = .false.
    end function test_common_without_save_accepted

    logical function test_block_data_common_object_accepted()
        ! Corrected neighbour: the BLOCK DATA object is in COMMON.
        character(len=*), parameter :: source = &
            'block data d'//new_line('a')// &
            '  integer i'//new_line('a')// &
            '  common /blk/ i'//new_line('a')// &
            '  data i /1/'//new_line('a')// &
            'end block data'//new_line('a')// &
            'program main'//new_line('a')// &
            '  integer i'//new_line('a')// &
            '  common /blk/ i'//new_line('a')// &
            '  stop i'//new_line('a')// &
            'end program main'

        test_block_data_common_object_accepted = expect_cli_no_error( &
            source, '/tmp/ffc_session_storage01_blockdata_ok_cli')
        if (.not. expect_exit_status(source, 1, &
                                     '/tmp/ffc_session_storage01_blockdata_ok')) &
            test_block_data_common_object_accepted = .false.
    end function test_block_data_common_object_accepted

end program test_session_reject_storage_01_compiler
