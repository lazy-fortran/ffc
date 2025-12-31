module memory_tracker
    use iso_c_binding
    implicit none
    private

    public :: memory_tracker_t

    ! Memory allocation record
    type :: allocation_record_t
        character(len=64) :: name = ""
        integer(8) :: size = 0
        logical :: freed = .false.
        integer(8) :: timestamp = 0
    end type allocation_record_t

    ! REFACTOR: Memory tracker type with improved organization and documentation
    type :: memory_tracker_t
        private
        logical :: initialized = .false.
        integer(8) :: current_usage = 0
        integer(8) :: peak_usage = 0
        logical :: peak_tracking_enabled = .false.
        type(allocation_record_t), allocatable :: allocations(:)
        integer :: allocation_count = 0
        integer :: max_allocations = 10000
        integer(8) :: phase_start_usage = 0
        character(len=64) :: current_phase = ""
        ! Statistics
        integer(8) :: total_allocations = 0
        integer(8) :: total_deallocations = 0
        integer(8) :: allocation_failures = 0
    contains
        ! Lifecycle
        procedure :: init => tracker_init
        procedure :: cleanup => tracker_cleanup
        procedure :: is_initialized => tracker_is_initialized
        ! Memory tracking
        procedure :: record_allocation => tracker_record_allocation
        procedure :: record_deallocation => tracker_record_deallocation
        ! Queries
        procedure :: get_current_usage => tracker_get_current_usage
        procedure :: get_peak_usage => tracker_get_peak_usage
        procedure :: has_memory_leaks => tracker_has_memory_leaks
        procedure :: verify_all_freed => tracker_verify_all_freed
        ! Configuration
        procedure :: enable_peak_tracking => tracker_enable_peak_tracking
        ! Phase management
        procedure :: start_phase => tracker_start_phase
        procedure :: end_phase => tracker_end_phase
        ! Reporting
        procedure :: print_leak_report => tracker_print_leak_report
        procedure :: print_statistics => tracker_print_statistics
    end type memory_tracker_t

contains

    ! REFACTOR: Initialize tracker with error checking
    subroutine tracker_init(this)
        class(memory_tracker_t), intent(inout) :: this
        integer :: stat
        
        if (this%initialized) return
        
        allocate(this%allocations(this%max_allocations), stat=stat)
        if (stat /= 0) then
            this%initialized = .false.
            return
        end if
        
        this%allocation_count = 0
        this%current_usage = 0
        this%peak_usage = 0
        this%peak_tracking_enabled = .false.
        this%total_allocations = 0
        this%total_deallocations = 0
        this%allocation_failures = 0
        this%current_phase = ""
        this%initialized = .true.
    end subroutine tracker_init

    subroutine tracker_cleanup(this)
        class(memory_tracker_t), intent(inout) :: this
        
        if (allocated(this%allocations)) then
            deallocate(this%allocations)
        end if
        
        this%initialized = .false.
        this%allocation_count = 0
        this%current_usage = 0
        this%peak_usage = 0
    end subroutine tracker_cleanup

    function tracker_is_initialized(this) result(is_init)
        class(memory_tracker_t), intent(in) :: this
        logical :: is_init
        
        is_init = this%initialized
    end function tracker_is_initialized

    function tracker_get_current_usage(this) result(usage)
        class(memory_tracker_t), intent(in) :: this
        integer(8) :: usage
        
        usage = this%current_usage
    end function tracker_get_current_usage

    function tracker_get_peak_usage(this) result(usage)
        class(memory_tracker_t), intent(in) :: this
        integer(8) :: usage
        
        usage = this%peak_usage
    end function tracker_get_peak_usage

    subroutine tracker_enable_peak_tracking(this)
        class(memory_tracker_t), intent(inout) :: this
        
        this%peak_tracking_enabled = .true.
    end subroutine tracker_enable_peak_tracking

    ! REFACTOR: Record allocation with improved error handling and statistics
    subroutine tracker_record_allocation(this, name, size)
        class(memory_tracker_t), intent(inout) :: this
        character(len=*), intent(in) :: name
        integer(8), intent(in) :: size
        integer :: i
        
        if (.not. this%initialized) return
        
        ! Validate input
        if (size <= 0) then
            this%allocation_failures = this%allocation_failures + 1
            return
        end if
        
        ! Find empty slot or expand array
        if (this%allocation_count >= this%max_allocations) then
            call expand_allocations_array(this)
            if (this%allocation_count >= this%max_allocations) then
                ! Expansion failed
                this%allocation_failures = this%allocation_failures + 1
                return
            end if
        end if
        
        ! Add new allocation
        this%allocation_count = this%allocation_count + 1
        this%allocations(this%allocation_count)%name = name
        this%allocations(this%allocation_count)%size = size
        this%allocations(this%allocation_count)%freed = .false.
        this%allocations(this%allocation_count)%timestamp = get_timestamp()
        
        ! Update statistics
        this%total_allocations = this%total_allocations + 1
        this%current_usage = this%current_usage + size
        
        ! Update peak if needed
        if (this%peak_tracking_enabled .and. this%current_usage > this%peak_usage) then
            this%peak_usage = this%current_usage
        end if
    end subroutine tracker_record_allocation

    ! REFACTOR: Record deallocation with validation
    subroutine tracker_record_deallocation(this, name, size)
        class(memory_tracker_t), intent(inout) :: this
        character(len=*), intent(in) :: name
        integer(8), intent(in) :: size
        integer :: i
        logical :: found
        
        if (.not. this%initialized) return
        
        found = .false.
        
        ! Find matching allocation
        do i = 1, this%allocation_count
            if (trim(this%allocations(i)%name) == trim(name) .and. &
                this%allocations(i)%size == size .and. &
                .not. this%allocations(i)%freed) then
                this%allocations(i)%freed = .true.
                found = .true.
                exit
            end if
        end do
        
        if (found) then
            this%total_deallocations = this%total_deallocations + 1
            this%current_usage = this%current_usage - size
            if (this%current_usage < 0) this%current_usage = 0
        else
            ! Deallocation without matching allocation
            this%allocation_failures = this%allocation_failures + 1
        end if
    end subroutine tracker_record_deallocation

    function tracker_has_memory_leaks(this) result(has_leaks)
        class(memory_tracker_t), intent(in) :: this
        logical :: has_leaks
        integer :: i
        
        has_leaks = .false.
        
        if (.not. this%initialized) return
        
        do i = 1, this%allocation_count
            if (.not. this%allocations(i)%freed) then
                has_leaks = .true.
                exit
            end if
        end do
    end function tracker_has_memory_leaks

    subroutine tracker_print_leak_report(this)
        class(memory_tracker_t), intent(in) :: this
        integer :: i
        integer :: leak_count
        integer(8) :: leaked_bytes
        
        if (.not. this%initialized) return
        
        leak_count = 0
        leaked_bytes = 0
        
        print *, "=== Memory Leak Report ==="
        
        do i = 1, this%allocation_count
            if (.not. this%allocations(i)%freed) then
                leak_count = leak_count + 1
                leaked_bytes = leaked_bytes + this%allocations(i)%size
                print '(A,A,A,I0,A)', "  Leaked: ", trim(this%allocations(i)%name), &
                     " (", this%allocations(i)%size, " bytes)"
            end if
        end do
        
        print '(A,I0,A,I0,A)', "Total: ", leak_count, " leaks, ", leaked_bytes, " bytes"
        print *, "========================="
    end subroutine tracker_print_leak_report

    subroutine tracker_start_phase(this, phase_name)
        class(memory_tracker_t), intent(inout) :: this
        character(len=*), intent(in) :: phase_name
        
        this%current_phase = phase_name
        this%phase_start_usage = this%current_usage
    end subroutine tracker_start_phase

    subroutine tracker_end_phase(this, phase_name)
        class(memory_tracker_t), intent(inout) :: this
        character(len=*), intent(in) :: phase_name
        integer(8) :: phase_usage
        
        if (trim(this%current_phase) == trim(phase_name)) then
            phase_usage = this%current_usage - this%phase_start_usage
            ! Could log phase memory usage here
            this%current_phase = ""
        end if
    end subroutine tracker_end_phase

    function tracker_verify_all_freed(this) result(all_freed)
        class(memory_tracker_t), intent(in) :: this
        logical :: all_freed
        
        all_freed = .not. this%has_memory_leaks() .and. (this%current_usage == 0)
    end function tracker_verify_all_freed

    ! Helper subroutines

    subroutine expand_allocations_array(this)
        class(memory_tracker_t), intent(inout) :: this
        type(allocation_record_t), allocatable :: new_allocations(:)
        integer :: new_size
        
        new_size = this%max_allocations * 2
        allocate(new_allocations(new_size))
        
        new_allocations(1:this%max_allocations) = this%allocations
        
        deallocate(this%allocations)
        this%allocations = new_allocations
        this%max_allocations = new_size
    end subroutine expand_allocations_array

    function get_timestamp() result(timestamp)
        integer(8) :: timestamp
        ! Simple counter for now
        integer(8), save :: counter = 0
        counter = counter + 1
        timestamp = counter
    end function get_timestamp

    ! REFACTOR: Print comprehensive statistics
    subroutine tracker_print_statistics(this)
        class(memory_tracker_t), intent(in) :: this
        
        if (.not. this%initialized) return
        
        print *, "=== Memory Tracker Statistics ==="
        print '(A,I0)', "Current memory usage: ", this%current_usage
        print '(A,I0)', "Peak memory usage: ", this%peak_usage
        print '(A,I0)', "Total allocations: ", this%total_allocations
        print '(A,I0)', "Total deallocations: ", this%total_deallocations
        print '(A,I0)', "Active allocations: ", this%allocation_count
        print '(A,I0)', "Allocation failures: ", this%allocation_failures
        print '(A,L1)', "Has memory leaks: ", this%has_memory_leaks()
        print *, "================================="
    end subroutine tracker_print_statistics

end module memory_tracker