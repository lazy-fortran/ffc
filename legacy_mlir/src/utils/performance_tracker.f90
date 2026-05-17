module performance_tracker
    use iso_fortran_env, only: real64
    implicit none
    private
    
    public :: performance_tracker_t
    
    type :: performance_entry_t
        character(len=64) :: name = ""
        real(real64) :: start_time = 0.0_real64
        real(real64) :: end_time = 0.0_real64
        real(real64) :: duration = 0.0_real64
        integer :: call_count = 0
        logical :: active = .false.
    end type performance_entry_t
    
    type :: performance_tracker_t
        private
        logical :: initialized = .false.
        type(performance_entry_t), allocatable :: entries(:)
        integer :: entry_count = 0
        integer :: max_entries = 1000
        logical :: logging_enabled = .true.
        character(len=256) :: log_file = "performance.log"
    contains
        ! Lifecycle
        procedure :: init => tracker_init
        procedure :: cleanup => tracker_cleanup
        procedure :: is_initialized => tracker_is_initialized
        ! Performance tracking
        procedure :: start_timer => tracker_start_timer
        procedure :: end_timer => tracker_end_timer
        procedure :: record_measurement => tracker_record_measurement
        ! Data access
        procedure :: get_duration => tracker_get_duration
        procedure :: get_call_count => tracker_get_call_count
        procedure :: get_average_duration => tracker_get_average_duration
        ! Reporting
        procedure :: print_summary => tracker_print_summary
        procedure :: write_log => tracker_write_log
        procedure :: enable_logging => tracker_enable_logging
        procedure :: disable_logging => tracker_disable_logging
    end type performance_tracker_t
    
contains

    subroutine tracker_init(this)
        class(performance_tracker_t), intent(inout) :: this
        integer :: stat
        
        if (this%initialized) return
        
        allocate(this%entries(this%max_entries), stat=stat)
        if (stat /= 0) then
            error stop "Failed to allocate performance tracker entries"
        end if
        
        this%entry_count = 0
        this%initialized = .true.
    end subroutine tracker_init
    
    subroutine tracker_cleanup(this)
        class(performance_tracker_t), intent(inout) :: this
        
        if (.not. this%initialized) return
        
        if (allocated(this%entries)) then
            deallocate(this%entries)
        end if
        
        this%entry_count = 0
        this%initialized = .false.
    end subroutine tracker_cleanup
    
    function tracker_is_initialized(this) result(is_init)
        class(performance_tracker_t), intent(in) :: this
        logical :: is_init
        
        is_init = this%initialized
    end function tracker_is_initialized
    
    subroutine tracker_start_timer(this, name)
        class(performance_tracker_t), intent(inout) :: this
        character(len=*), intent(in) :: name
        integer :: i
        logical :: found
        
        if (.not. this%initialized) return
        
        ! Find existing entry or create new one
        found = .false.
        do i = 1, this%entry_count
            if (trim(this%entries(i)%name) == trim(name)) then
                found = .true.
                exit
            end if
        end do
        
        if (.not. found) then
            if (this%entry_count >= this%max_entries) then
                ! Simple replacement strategy - replace oldest
                i = 1
            else
                this%entry_count = this%entry_count + 1
                i = this%entry_count
            end if
            this%entries(i)%name = name
            this%entries(i)%call_count = 0
            this%entries(i)%duration = 0.0_real64
        end if
        
        call cpu_time(this%entries(i)%start_time)
        this%entries(i)%active = .true.
    end subroutine tracker_start_timer
    
    subroutine tracker_end_timer(this, name)
        class(performance_tracker_t), intent(inout) :: this
        character(len=*), intent(in) :: name
        integer :: i
        real(real64) :: current_duration
        
        if (.not. this%initialized) return
        
        ! Find the entry
        do i = 1, this%entry_count
            if (trim(this%entries(i)%name) == trim(name) .and. this%entries(i)%active) then
                call cpu_time(this%entries(i)%end_time)
                current_duration = this%entries(i)%end_time - this%entries(i)%start_time
                this%entries(i)%duration = this%entries(i)%duration + current_duration
                this%entries(i)%call_count = this%entries(i)%call_count + 1
                this%entries(i)%active = .false.
                exit
            end if
        end do
    end subroutine tracker_end_timer
    
    subroutine tracker_record_measurement(this, name, duration)
        class(performance_tracker_t), intent(inout) :: this
        character(len=*), intent(in) :: name
        real(real64), intent(in) :: duration
        integer :: i
        logical :: found
        
        if (.not. this%initialized) return
        
        ! Find existing entry or create new one
        found = .false.
        do i = 1, this%entry_count
            if (trim(this%entries(i)%name) == trim(name)) then
                found = .true.
                exit
            end if
        end do
        
        if (.not. found) then
            if (this%entry_count >= this%max_entries) return
            this%entry_count = this%entry_count + 1
            i = this%entry_count
            this%entries(i)%name = name
            this%entries(i)%call_count = 0
            this%entries(i)%duration = 0.0_real64
        end if
        
        this%entries(i)%duration = this%entries(i)%duration + duration
        this%entries(i)%call_count = this%entries(i)%call_count + 1
    end subroutine tracker_record_measurement
    
    function tracker_get_duration(this, name) result(duration)
        class(performance_tracker_t), intent(in) :: this
        character(len=*), intent(in) :: name
        real(real64) :: duration
        integer :: i
        
        duration = 0.0_real64
        if (.not. this%initialized) return
        
        do i = 1, this%entry_count
            if (trim(this%entries(i)%name) == trim(name)) then
                duration = this%entries(i)%duration
                return
            end if
        end do
    end function tracker_get_duration
    
    function tracker_get_call_count(this, name) result(count)
        class(performance_tracker_t), intent(in) :: this
        character(len=*), intent(in) :: name
        integer :: count
        integer :: i
        
        count = 0
        if (.not. this%initialized) return
        
        do i = 1, this%entry_count
            if (trim(this%entries(i)%name) == trim(name)) then
                count = this%entries(i)%call_count
                return
            end if
        end do
    end function tracker_get_call_count
    
    function tracker_get_average_duration(this, name) result(avg_duration)
        class(performance_tracker_t), intent(in) :: this
        character(len=*), intent(in) :: name
        real(real64) :: avg_duration
        integer :: i
        
        avg_duration = 0.0_real64
        if (.not. this%initialized) return
        
        do i = 1, this%entry_count
            if (trim(this%entries(i)%name) == trim(name)) then
                if (this%entries(i)%call_count > 0) then
                    avg_duration = this%entries(i)%duration / real(this%entries(i)%call_count, real64)
                end if
                return
            end if
        end do
    end function tracker_get_average_duration
    
    subroutine tracker_print_summary(this)
        class(performance_tracker_t), intent(in) :: this
        integer :: i
        
        if (.not. this%initialized) return
        
        print *, "======================================================================="
        print *, "Performance Summary"
        print *, "======================================================================="
        print '(A20,A12,A12,A15)', "Operation", "Calls", "Total (s)", "Average (s)"
        print *, "-----------------------------------------------------------------------"
        
        do i = 1, this%entry_count
            if (this%entries(i)%call_count > 0) then
                print '(A20,I12,F12.6,F15.6)', &
                    trim(this%entries(i)%name), &
                    this%entries(i)%call_count, &
                    this%entries(i)%duration, &
                    this%entries(i)%duration / real(this%entries(i)%call_count, real64)
            end if
        end do
        print *, "======================================================================="
    end subroutine tracker_print_summary
    
    subroutine tracker_write_log(this)
        class(performance_tracker_t), intent(in) :: this
        integer :: unit, stat, i
        
        if (.not. this%initialized .or. .not. this%logging_enabled) return
        
        open(newunit=unit, file=trim(this%log_file), action='write', iostat=stat)
        if (stat /= 0) return
        
        write(unit, '(A)') "# FortFC Performance Log"
        write(unit, '(A)') "# Operation,Calls,Total_Duration(s),Average_Duration(s)"
        
        do i = 1, this%entry_count
            if (this%entries(i)%call_count > 0) then
                write(unit, '(A,",",I0,",",F0.6,",",F0.6)') &
                    trim(this%entries(i)%name), &
                    this%entries(i)%call_count, &
                    this%entries(i)%duration, &
                    this%entries(i)%duration / real(this%entries(i)%call_count, real64)
            end if
        end do
        
        close(unit)
    end subroutine tracker_write_log
    
    subroutine tracker_enable_logging(this)
        class(performance_tracker_t), intent(inout) :: this
        this%logging_enabled = .true.
    end subroutine tracker_enable_logging
    
    subroutine tracker_disable_logging(this)
        class(performance_tracker_t), intent(inout) :: this  
        this%logging_enabled = .false.
    end subroutine tracker_disable_logging

end module performance_tracker