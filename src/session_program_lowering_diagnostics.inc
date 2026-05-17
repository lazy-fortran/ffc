    subroutine set_empty(value)
        character(len=:), allocatable, intent(out) :: value

        allocate (character(len=0) :: value)
    end subroutine set_empty

    subroutine unsupported_feature_error(feature, line, column, limitation, &
                                         error_msg)
        character(len=*), intent(in) :: feature
        integer, intent(in) :: line
        integer, intent(in) :: column
        character(len=*), intent(in) :: limitation
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=64) :: location

        if (line > 0 .and. column > 0) then
            write (location, '(" at line ",I0,", column ",I0)') line, column
            error_msg = 'unsupported '//trim(feature)//trim(location)//': '// &
                        trim(limitation)
        else
            error_msg = 'unsupported '//trim(feature)//': '//trim(limitation)
        end if
    end subroutine unsupported_feature_error
