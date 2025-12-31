module test_harness
    use iso_c_binding
    implicit none
    private

    public :: test_suite_t, test_case_t
    public :: run_test_suite, create_test_suite
    public :: add_test_case
    
    ! Test case type
    type :: test_case_t
        character(len=128) :: name = ""
        procedure(test_function_interface), pointer, nopass :: test_func => null()
        logical :: enabled = .true.
    end type test_case_t
    
    ! Test suite type
    type :: test_suite_t
        character(len=128) :: name = ""
        type(test_case_t), allocatable :: tests(:)
        integer :: test_count = 0
        integer :: max_tests = 1000
        ! Statistics
        integer :: passed = 0
        integer :: failed = 0
        integer :: skipped = 0
        real :: start_time = 0.0
        real :: end_time = 0.0
    contains
        procedure :: init => suite_init
        procedure :: cleanup => suite_cleanup
        procedure :: add_test => suite_add_test
        procedure :: run => suite_run
        procedure :: print_summary => suite_print_summary
    end type test_suite_t
    
    ! Test function interface
    abstract interface
        function test_function_interface() result(passed)
            logical :: passed
        end function test_function_interface
    end interface
    
contains

    ! Create a new test suite
    function create_test_suite(name) result(suite)
        character(len=*), intent(in) :: name
        type(test_suite_t) :: suite
        
        suite%name = name
        call suite%init()
    end function create_test_suite
    
    ! Initialize test suite
    subroutine suite_init(this)
        class(test_suite_t), intent(inout) :: this
        
        if (.not. allocated(this%tests)) then
            allocate(this%tests(this%max_tests))
        end if
        this%test_count = 0
        this%passed = 0
        this%failed = 0
        this%skipped = 0
    end subroutine suite_init
    
    ! Cleanup test suite
    subroutine suite_cleanup(this)
        class(test_suite_t), intent(inout) :: this
        
        if (allocated(this%tests)) then
            deallocate(this%tests)
        end if
    end subroutine suite_cleanup
    
    ! Add a test case to the suite
    subroutine suite_add_test(this, name, test_func, enabled)
        class(test_suite_t), intent(inout) :: this
        character(len=*), intent(in) :: name
        procedure(test_function_interface) :: test_func
        logical, intent(in), optional :: enabled
        
        if (this%test_count >= this%max_tests) return
        
        this%test_count = this%test_count + 1
        this%tests(this%test_count)%name = name
        this%tests(this%test_count)%test_func => test_func
        
        if (present(enabled)) then
            this%tests(this%test_count)%enabled = enabled
        else
            this%tests(this%test_count)%enabled = .true.
        end if
    end subroutine suite_add_test
    
    ! Helper subroutine for adding test cases
    subroutine add_test_case(suite, name, test_func, enabled)
        type(test_suite_t), intent(inout) :: suite
        character(len=*), intent(in) :: name
        procedure(test_function_interface) :: test_func
        logical, intent(in), optional :: enabled
        
        call suite%add_test(name, test_func, enabled)
    end subroutine add_test_case
    
    ! Run all tests in the suite
    subroutine suite_run(this, verbose)
        class(test_suite_t), intent(inout) :: this
        logical, intent(in), optional :: verbose
        logical :: is_verbose
        logical :: test_passed
        integer :: i
        real :: test_start, test_end
        
        is_verbose = .false.
        if (present(verbose)) is_verbose = verbose
        
        print '(A)', repeat("=", 70)
        print '(A,A)', "Running Test Suite: ", trim(this%name)
        print '(A)', repeat("=", 70)
        
        call cpu_time(this%start_time)
        
        do i = 1, this%test_count
            if (.not. this%tests(i)%enabled) then
                this%skipped = this%skipped + 1
                if (is_verbose) then
                    print '(A,A,A)', "[ SKIP ] ", trim(this%tests(i)%name), " (disabled)"
                end if
                cycle
            end if
            
            call cpu_time(test_start)
            test_passed = this%tests(i)%test_func()
            call cpu_time(test_end)
            
            if (test_passed) then
                this%passed = this%passed + 1
                if (is_verbose) then
                    print '(A,A,A,F6.3,A)', "[ PASS ] ", trim(this%tests(i)%name), &
                          " (", test_end - test_start, "s)"
                end if
            else
                this%failed = this%failed + 1
                print '(A,A)', "[ FAIL ] ", trim(this%tests(i)%name)
            end if
        end do
        
        call cpu_time(this%end_time)
        
        call this%print_summary()
    end subroutine suite_run
    
    ! Print test suite summary
    subroutine suite_print_summary(this)
        class(test_suite_t), intent(in) :: this
        real :: total_time
        
        total_time = this%end_time - this%start_time
        
        print '(A)', repeat("-", 70)
        print '(A,I0,A,I0,A,I0,A,I0,A)', "Tests: ", this%test_count, &
              " | Passed: ", this%passed, &
              " | Failed: ", this%failed, &
              " | Skipped: ", this%skipped, &
              " |"
        print '(A,F6.3,A)', "Total time: ", total_time, " seconds"
        print '(A)', repeat("=", 70)
        
        if (this%failed > 0) then
            print '(A)', "SUITE FAILED"
        else
            print '(A)', "SUITE PASSED"
        end if
        print *
    end subroutine suite_print_summary
    
    ! Run a test suite (convenience function)
    subroutine run_test_suite(suite, verbose)
        type(test_suite_t), intent(inout) :: suite
        logical, intent(in), optional :: verbose
        
        call suite%run(verbose)
    end subroutine run_test_suite

end module test_harness