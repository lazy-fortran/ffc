program test_docs_no_drift_claims
  implicit none

  character(len=256), parameter :: readme_path = "README.md"
  character(len=256), parameter :: api_path = "docs/API_REFERENCE.md"
  character(len=256), parameter :: dev_path = "docs/DEVELOPER_GUIDE.md"

  integer :: unit, iostat
  character(len=4096) :: line
  character(len=1024) :: content
  logical :: found_readme, found_api, found_dev
  logical :: found_support_contract, found_fpm_build, found_fpm_test
  logical :: fail

  ! Forbidden phrases that must not appear
  character(len=1024), parameter :: forbidden_1 = "currently 40 programs"
  character(len=1024), parameter :: forbidden_2 = "Stored but not yet consumed by lowering"
  character(len=1024), parameter :: forbidden_3 = "CI runs the same workflow"

  ! Required content
  character(len=1024), parameter :: required_1 = "SUPPORT_CONTRACT.md"
  character(len=1024), parameter :: required_2 = "fpm build"
  character(len=1024), parameter :: required_3 = "fpm test"

  fail = .false.
  found_readme = .false.
  found_api = .false.
  found_dev = .false.
  found_support_contract = .false.
  found_fpm_build = .false.
  found_fpm_test = .false.

  ! Check README.md
  unit = 10
  open(unit=unit, file=readme_path, status="old", iostat=iostat)
  if (iostat == 0) then
    found_readme = .true.
    do
      read(unit, '(a)', iostat=iostat) line
      if (iostat /= 0) exit
      if (index(line, trim(forbidden_1)) > 0) then
        write(*, '(a)') "FAIL: README.md contains forbidden phrase: " // trim(forbidden_1)
        fail = .true.
      end if
      if (index(line, trim(forbidden_2)) > 0) then
        write(*, '(a)') "FAIL: README.md contains forbidden phrase: " // trim(forbidden_2)
        fail = .true.
      end if
      if (index(line, trim(forbidden_3)) > 0) then
        write(*, '(a)') "FAIL: README.md contains forbidden phrase: " // trim(forbidden_3)
        fail = .true.
      end if
      if (index(line, trim(required_1)) > 0) found_support_contract = .true.
      if (index(line, trim(required_2)) > 0) found_fpm_build = .true.
      if (index(line, trim(required_3)) > 0) found_fpm_test = .true.
    end do
    close(unit)
  else
    write(*, '(a)') "FAIL: Cannot open " // trim(readme_path)
    fail = .true.
  end if

  ! Check docs/API_REFERENCE.md
  unit = 20
  open(unit=unit, file=api_path, status="old", iostat=iostat)
  if (iostat == 0) then
    found_api = .true.
    do
      read(unit, '(a)', iostat=iostat) line
      if (iostat /= 0) exit
      if (index(line, trim(forbidden_1)) > 0) then
        write(*, '(a)') "FAIL: API_REFERENCE.md contains forbidden phrase: " // trim(forbidden_1)
        fail = .true.
      end if
      if (index(line, trim(forbidden_2)) > 0) then
        write(*, '(a)') "FAIL: API_REFERENCE.md contains forbidden phrase: " // trim(forbidden_2)
        fail = .true.
      end if
      if (index(line, trim(forbidden_3)) > 0) then
        write(*, '(a)') "FAIL: API_REFERENCE.md contains forbidden phrase: " // trim(forbidden_3)
        fail = .true.
      end if
      if (index(line, trim(required_1)) > 0) found_support_contract = .true.
    end do
    close(unit)
  else
    write(*, '(a)') "FAIL: Cannot open " // trim(api_path)
    fail = .true.
  end if

  ! Check docs/DEVELOPER_GUIDE.md
  unit = 30
  open(unit=unit, file=dev_path, status="old", iostat=iostat)
  if (iostat == 0) then
    found_dev = .true.
    do
      read(unit, '(a)', iostat=iostat) line
      if (iostat /= 0) exit
      if (index(line, trim(forbidden_1)) > 0) then
        write(*, '(a)') "FAIL: DEVELOPER_GUIDE.md contains forbidden phrase: " // trim(forbidden_1)
        fail = .true.
      end if
      if (index(line, trim(forbidden_2)) > 0) then
        write(*, '(a)') "FAIL: DEVELOPER_GUIDE.md contains forbidden phrase: " // trim(forbidden_2)
        fail = .true.
      end if
      if (index(line, trim(forbidden_3)) > 0) then
        write(*, '(a)') "FAIL: DEVELOPER_GUIDE.md contains forbidden phrase: " // trim(forbidden_3)
        fail = .true.
      end if
      if (index(line, trim(required_1)) > 0) found_support_contract = .true.
      if (index(line, trim(required_2)) > 0) found_fpm_build = .true.
      if (index(line, trim(required_3)) > 0) found_fpm_test = .true.
    end do
    close(unit)
  else
    write(*, '(a)') "FAIL: Cannot open " // trim(dev_path)
    fail = .true.
  end if

  ! Check required content presence
  if (.not. found_support_contract) then
    write(*, '(a)') "FAIL: No doc mentions SUPPORT_CONTRACT.md"
    fail = .true.
  end if

  if (.not. found_fpm_build) then
    write(*, '(a)') "FAIL: No doc mentions fpm build"
    fail = .true.
  end if

  if (.not. found_fpm_test) then
    write(*, '(a)') "FAIL: No doc mentions fpm test"
    fail = .true.
  end if

  if (fail) then
    stop 1
  else
    write(*, '(a)') "PASS: No drift-prone claims found"
    stop 0
  end if

end program test_docs_no_drift_claims
