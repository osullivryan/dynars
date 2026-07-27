! Parse an LS-DYNA deck and validate it, from Fortran.
!
! Fortran talks to Rust through the C ABI: the `dynars` module below declares
! iso_c_binding interfaces to the same symbols in dynars.h, and the program
! drives them exactly like the C example.
!
! Build (from examples/ffi/):  make f_example
! Run:                         ./f_example path/to/main.k

module dynars
  use, intrinsic :: iso_c_binding
  implicit none
  private

  ! Finding severities (mirror of the DynarsSeverity enum in dynars.h).
  integer(c_int), parameter, public :: DYNARS_ERROR   = 0_c_int
  integer(c_int), parameter, public :: DYNARS_WARNING = 1_c_int
  integer(c_int), parameter, public :: DYNARS_INFO    = 2_c_int

  ! Public API mirrored from dynars.h.
  public :: dynars_parse_deck, dynars_deck_free
  public :: dynars_deck_file_count, dynars_deck_total_bytes
  public :: dynars_ruleset_new, dynars_ruleset_free
  public :: dynars_ruleset_add_references_resolve
  public :: dynars_ruleset_add_references_resolve_with_connectivity
  public :: dynars_ruleset_add_include_missing
  public :: dynars_ruleset_add_keyword_forbidden
  public :: dynars_deck_validate, dynars_report_free
  public :: dynars_report_len, dynars_report_count, dynars_report_is_clean
  public :: dynars_report_finding_severity, dynars_report_finding_line
  public :: dynars_report_finding_file, dynars_report_finding_message
  public :: dynars_last_error
  ! Helpers.
  public :: c_to_f_string, severity_name

  interface
    function dynars_parse_deck(path) bind(C, name="dynars_parse_deck") result(deck)
      import :: c_ptr, c_char
      character(kind=c_char), dimension(*), intent(in) :: path
      type(c_ptr) :: deck
    end function

    subroutine dynars_deck_free(deck) bind(C, name="dynars_deck_free")
      import :: c_ptr
      type(c_ptr), value :: deck
    end subroutine

    function dynars_deck_file_count(deck) bind(C, name="dynars_deck_file_count") result(n)
      import :: c_ptr, c_size_t
      type(c_ptr), value :: deck
      integer(c_size_t) :: n
    end function

    function dynars_deck_total_bytes(deck) bind(C, name="dynars_deck_total_bytes") result(n)
      import :: c_ptr, c_size_t
      type(c_ptr), value :: deck
      integer(c_size_t) :: n
    end function

    function dynars_last_error() bind(C, name="dynars_last_error") result(msg)
      import :: c_ptr
      type(c_ptr) :: msg
    end function

    function dynars_ruleset_new() bind(C, name="dynars_ruleset_new") result(rules)
      import :: c_ptr
      type(c_ptr) :: rules
    end function

    subroutine dynars_ruleset_free(rules) bind(C, name="dynars_ruleset_free")
      import :: c_ptr
      type(c_ptr), value :: rules
    end subroutine

    subroutine dynars_ruleset_add_references_resolve(rules) &
        bind(C, name="dynars_ruleset_add_references_resolve")
      import :: c_ptr
      type(c_ptr), value :: rules
    end subroutine

    subroutine dynars_ruleset_add_references_resolve_with_connectivity(rules) &
        bind(C, name="dynars_ruleset_add_references_resolve_with_connectivity")
      import :: c_ptr
      type(c_ptr), value :: rules
    end subroutine

    subroutine dynars_ruleset_add_include_missing(rules) &
        bind(C, name="dynars_ruleset_add_include_missing")
      import :: c_ptr
      type(c_ptr), value :: rules
    end subroutine

    function dynars_ruleset_add_keyword_forbidden(rules, keyword) &
        bind(C, name="dynars_ruleset_add_keyword_forbidden") result(rc)
      import :: c_ptr, c_char, c_int
      type(c_ptr), value :: rules
      character(kind=c_char), dimension(*), intent(in) :: keyword
      integer(c_int) :: rc
    end function

    function dynars_deck_validate(deck, rules) bind(C, name="dynars_deck_validate") result(report)
      import :: c_ptr
      type(c_ptr), value :: deck
      type(c_ptr), value :: rules
      type(c_ptr) :: report
    end function

    subroutine dynars_report_free(report) bind(C, name="dynars_report_free")
      import :: c_ptr
      type(c_ptr), value :: report
    end subroutine

    function dynars_report_len(report) bind(C, name="dynars_report_len") result(n)
      import :: c_ptr, c_size_t
      type(c_ptr), value :: report
      integer(c_size_t) :: n
    end function

    function dynars_report_count(report, severity) bind(C, name="dynars_report_count") result(n)
      import :: c_ptr, c_size_t, c_int
      type(c_ptr), value :: report
      integer(c_int), value :: severity
      integer(c_size_t) :: n
    end function

    function dynars_report_is_clean(report) bind(C, name="dynars_report_is_clean") result(c)
      import :: c_ptr, c_int
      type(c_ptr), value :: report
      integer(c_int) :: c
    end function

    function dynars_report_finding_severity(report, i) &
        bind(C, name="dynars_report_finding_severity") result(s)
      import :: c_ptr, c_size_t, c_int
      type(c_ptr), value :: report
      integer(c_size_t), value :: i
      integer(c_int) :: s
    end function

    function dynars_report_finding_line(report, i) &
        bind(C, name="dynars_report_finding_line") result(l)
      import :: c_ptr, c_size_t
      type(c_ptr), value :: report
      integer(c_size_t), value :: i
      integer(c_size_t) :: l
    end function

    function dynars_report_finding_file(report, i) &
        bind(C, name="dynars_report_finding_file") result(p)
      import :: c_ptr, c_size_t
      type(c_ptr), value :: report
      integer(c_size_t), value :: i
      type(c_ptr) :: p
    end function

    function dynars_report_finding_message(report, i) &
        bind(C, name="dynars_report_finding_message") result(p)
      import :: c_ptr, c_size_t
      type(c_ptr), value :: report
      integer(c_size_t), value :: i
      type(c_ptr) :: p
    end function

    ! libc strlen — lets us size a NUL-terminated string the library handed back.
    function c_strlen(s) bind(C, name="strlen") result(n)
      import :: c_ptr, c_size_t
      type(c_ptr), value :: s
      integer(c_size_t) :: n
    end function
  end interface

contains

  ! Copy a C `const char*` (NUL-terminated) into a Fortran allocatable string.
  function c_to_f_string(cptr) result(f)
    type(c_ptr), intent(in) :: cptr
    character(len=:), allocatable :: f
    character(kind=c_char), pointer :: chars(:)
    integer(c_size_t) :: n, i
    if (.not. c_associated(cptr)) then
      f = ""
      return
    end if
    n = c_strlen(cptr)
    call c_f_pointer(cptr, chars, [n])
    allocate(character(len=n) :: f)
    do i = 1, n
      f(i:i) = chars(i)
    end do
  end function

  function severity_name(s) result(name)
    integer(c_int), intent(in) :: s
    character(len=7) :: name
    select case (s)
    case (0); name = "ERROR  "
    case (1); name = "WARNING"
    case (2); name = "INFO   "
    case default; name = "?      "
    end select
  end function

end module dynars


program validate_deck
  use, intrinsic :: iso_c_binding
  use dynars
  implicit none

  character(len=4096) :: path
  type(c_ptr) :: deck, rules, report
  integer(c_size_t) :: nfind, i
  integer(c_int) :: sev, clean
  integer :: nargs

  nargs = command_argument_count()
  if (nargs < 1) then
    write (*, '(a)') "usage: f_example <path-to-deck.k>"
    call exit(2)
  end if
  call get_command_argument(1, path)

  deck = dynars_parse_deck(trim(path) // c_null_char)
  if (.not. c_associated(deck)) then
    write (*, '(a,a)') "parse failed: ", c_to_f_string(dynars_last_error())
    call exit(1)
  end if
  write (*, '(a,i0,a,i0,a)') "parsed ", dynars_deck_file_count(deck), &
      " file(s), ", dynars_deck_total_bytes(deck), " bytes"

  ! Assemble the checks we want to run.
  rules = dynars_ruleset_new()
  call dynars_ruleset_add_references_resolve(rules)
  call dynars_ruleset_add_include_missing(rules)

  report = dynars_deck_validate(deck, rules)
  nfind = dynars_report_len(report)
  write (*, '(i0,a,i0,a,i0,a,i0,a)') nfind, " finding(s): ", &
      dynars_report_count(report, DYNARS_ERROR), " error, ", &
      dynars_report_count(report, DYNARS_WARNING), " warning, ", &
      dynars_report_count(report, DYNARS_INFO), " info"

  do i = 0, nfind - 1
    sev = dynars_report_finding_severity(report, i)
    write (*, '(a,a,a,a,a,i0,a,a)') "  [", trim(severity_name(sev)), "] ", &
        c_to_f_string(dynars_report_finding_file(report, i)), ":", &
        dynars_report_finding_line(report, i), "  ", &
        c_to_f_string(dynars_report_finding_message(report, i))
  end do

  clean = dynars_report_is_clean(report)
  if (clean == 1) then
    write (*, '(a)') "deck is clean (no errors)"
  else
    write (*, '(a)') "deck has errors"
  end if

  call dynars_report_free(report)
  call dynars_ruleset_free(rules)
  call dynars_deck_free(deck)

  ! Match the C example: non-zero exit if any error-severity finding was seen.
  if (clean /= 1) call exit(1)
end program validate_deck
