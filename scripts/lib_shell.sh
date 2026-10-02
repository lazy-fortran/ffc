#!/usr/bin/env bash
# Entry points call this before sourcing helpers that require Bash 4.3.

ffc_require_bash() {
    if [ "${BASH_VERSINFO[0]:-0}" -gt 4 ] || \
            { [ "${BASH_VERSINFO[0]:-0}" -eq 4 ] && \
              [ "${BASH_VERSINFO[1]:-0}" -ge 3 ]; }; then
        return 0
    fi
    local candidate
    for candidate in "${FFC_BASH:-}" "$(command -v bash 2>/dev/null)" \
            /opt/homebrew/bin/bash /usr/local/bin/bash; do
        [ -n "$candidate" ] && [ -x "$candidate" ] || continue
        if "$candidate" -c \
                '(( BASH_VERSINFO[0] > 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] >= 3) ))' \
                >/dev/null 2>&1; then
            exec "$candidate" -- "$0" "$@"
        fi
    done
    printf 'ERROR: Bash 4.3 or newer is required; install it or set FFC_BASH\n' >&2
    return 1
}
