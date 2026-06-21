#!/usr/bin/env bash
# lib_conformance.sh: shared helpers for the conformance gauntlet.
# Source this file; do not execute it directly.
#
# Public functions:
#   find_ffc               resolve ffc binary path
#   compile_with_ffc       compile a source file through ffc
#   compile_object_with_ffc  compile a source file through ffc -c
#   compile_with_gfortran  compile a source file with gfortran -w
#   run_capture            run an executable with timeout, capture stdout+stderr
#   compare_outputs        compare stdout files and exit statuses
#   build_module_index     map sibling module/submodule names to their files
#   resolve_prerequisites  order the sibling files a source needs compiled first

set -uo pipefail

# Resolve the ffc binary. Priority: --ffc arg > FFC_BIN env > build/ tree > PATH.
find_ffc() {
    if [ -n "${FFC_BIN:-}" ]; then
        echo "$FFC_BIN"
        return 0
    fi
    local candidate
    # Pick the most recently built ffc. Several may coexist (build/fo/bin/ffc
    # from the fo backend, build/gfortran_*/app/ffc from fpm); an arbitrary
    # head -1 can return a stale one whose lowering predates recent fixes.
    candidate=$(find build -name ffc -type f -executable -printf '%T@ %p\n' \
        2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-) || true
    if [ -n "$candidate" ]; then
        echo "$candidate"
        return 0
    fi
    local in_path
    in_path=$(command -v ffc 2>/dev/null) || true
    if [ -n "$in_path" ]; then
        echo "$in_path"
        return 0
    fi
    echo "ERROR: ffc binary not found. Set FFC_BIN or run 'fpm build' first." >&2
    return 1
}

# compile_with_ffc <source> <exe> <ffc_path> [extra_args...]
# Returns 0 on success, non-zero on failure. Extra arguments (e.g. -I dir and
# prerequisite .o files for separate compilation) are passed straight to ffc.
compile_with_ffc() {
    local source="$1" exe="$2" ffc="$3"
    shift 3
    timeout "${FFC_COMPILE_TIMEOUT:-10}" "$ffc" "$source" -o "$exe" "$@" 2>/dev/null
    return $?
}

# modules_defined_in <source>
# Emit the lowercase names of modules DEFINED in the file, one per line.
# Matches a bare `module <name>` definition line; excludes `module procedure`
# and `submodule(...)`.
modules_defined_in() {
    local source="$1"
    grep -iE '^[[:space:]]*module[[:space:]]+[a-zA-Z][a-zA-Z0-9_]*[[:space:]]*$' \
        "$source" 2>/dev/null \
        | sed -E 's/^[[:space:]]*[mM][oO][dD][uU][lL][eE][[:space:]]+//' \
        | sed -E 's/[[:space:]]*$//' \
        | tr 'A-Z' 'a-z'
}

# submodule_parents_in <source>
# Emit the lowercase parent-module names for any `submodule(parent) name` units
# in the file. The leftmost identifier in the parenthesised ancestor list is the
# module whose interfaces the submodule implements.
submodule_parents_in() {
    local source="$1"
    grep -iE '^[[:space:]]*submodule[[:space:]]*\([[:space:]]*[a-zA-Z]' \
        "$source" 2>/dev/null \
        | sed -E 's/^[[:space:]]*[sS][uU][bB][mM][oO][dD][uU][lL][eE][[:space:]]*\([[:space:]]*([a-zA-Z][a-zA-Z0-9_]*).*/\1/' \
        | tr 'A-Z' 'a-z'
}

# modules_used_in <source>
# Emit the lowercase names of modules USEd by the file, one per line.
modules_used_in() {
    local source="$1"
    grep -iE '^[[:space:]]*use[[:space:]]' "$source" 2>/dev/null \
        | sed -E 's/^[[:space:]]*[uU][sS][eE][[:space:]]+//' \
        | sed -E 's/^,[^:]*::[[:space:]]*//' \
        | sed -E 's/[[:space:]]*[,!].*$//' \
        | sed -E 's/[[:space:]]*$//' \
        | tr 'A-Z' 'a-z' \
        | awk 'NF'
}

# build_module_index <suite_dir> <index_file>
# Write a "mod\t<modname>\t<file>" line for every module defined by a sibling
# file in the suite directory, and a "sub\t<parent>\t<file>" line for every
# submodule unit. Matching is case-insensitive (names already lowercased).
build_module_index() {
    local dir="$1" index="$2"
    : > "$index"
    local f modname parent
    while IFS= read -r f; do
        [ -z "$f" ] && continue
        while IFS= read -r modname; do
            [ -z "$modname" ] && continue
            printf 'mod\t%s\t%s\n' "$modname" "$(basename "$f")" >> "$index"
        done < <(modules_defined_in "$f")
        while IFS= read -r parent; do
            [ -z "$parent" ] && continue
            printf 'sub\t%s\t%s\n' "$parent" "$(basename "$f")" >> "$index"
        done < <(submodule_parents_in "$f")
    done < <(find "$dir" -maxdepth 1 \( -name '*.f90' -o -name '*.lf' \) -type f | sort)
}

# index_file_for_module <index> <modname>
# Echo the first sibling file basename that DEFINES the given module.
index_file_for_module() {
    local index="$1" modname="$2"
    awk -F'\t' -v m="$modname" '$1=="mod" && $2==m {print $3; exit}' "$index"
}

# index_submodule_files <index> <modname>
# Echo all sibling file basenames whose submodules name the given parent module.
index_submodule_files() {
    local index="$1" modname="$2"
    awk -F'\t' -v m="$modname" '$1=="sub" && $2==m {print $3}' "$index"
}

# resolve_prerequisites <source> <suite_dir> <index> <out_list>
# Resolve, in dependency order, the sibling files that must be compiled before
# <source> so the modules it USEs are available. A module is a prerequisite only
# when it is not defined in <source> itself and a sibling file defines it.
# Module-to-module dependencies are followed transitively. Submodule files that
# implement a pulled-in module's interfaces are appended so the final link
# resolves their procedures. Writes one path per line, in compile order
# (dependencies first); the list is empty when <source> is self-contained.
resolve_prerequisites() {
    local source="$1" dir="$2" index="$3" out="$4"
    : > "$out"
    local seen_files="" pulled_mods=""
    _rp_collect() {
        local f="$1"
        local self_mods u dep
        self_mods=$(modules_defined_in "$f")
        while IFS= read -r u; do
            [ -z "$u" ] && continue
            if printf '%s\n' "$self_mods" | grep -Fxq -- "$u"; then
                continue
            fi
            dep=$(index_file_for_module "$index" "$u")
            [ -z "$dep" ] && continue
            [ "$dep" = "$(basename "$f")" ] && continue
            case " $pulled_mods " in *" $u "*) ;; *) pulled_mods="$pulled_mods $u" ;; esac
            case " $seen_files " in
                *" $dep "*) ;;
                *)
                    seen_files="$seen_files $dep"
                    _rp_collect "$dir/$dep"
                    printf '%s/%s\n' "$dir" "$dep" >> "$out"
                    ;;
            esac
        done < <(modules_used_in "$f")
    }
    _rp_collect "$source"
    # Append submodule implementation files for every pulled-in module. A
    # submodule may declare further child modules, so iterate to a fixpoint.
    local changed=1 m sf nm newmods
    while [ "$changed" -eq 1 ]; do
        changed=0
        for m in $pulled_mods; do
            while IFS= read -r sf; do
                [ -z "$sf" ] && continue
                case " $seen_files " in
                    *" $sf "*) continue ;;
                esac
                seen_files="$seen_files $sf"
                printf '%s/%s\n' "$dir" "$sf" >> "$out"
                newmods=$(modules_defined_in "$dir/$sf")
                while IFS= read -r nm; do
                    [ -z "$nm" ] && continue
                    case " $pulled_mods " in *" $nm "*) ;; *) pulled_mods="$pulled_mods $nm"; changed=1 ;; esac
                done <<< "$newmods"
            done < <(index_submodule_files "$index" "$m")
        done
    done
    unset -f _rp_collect
}

# compile_object_with_ffc <source> <object> <ffc_path>
# Returns 0 on success, non-zero on failure.
compile_object_with_ffc() {
    local source="$1" object="$2" ffc="$3"
    timeout "${FFC_COMPILE_TIMEOUT:-10}" "$ffc" "$source" -c -o "$object" 2>/dev/null
    return $?
}

# compile_object_with_ffc_inc <source> <object> <ffc_path> <inc_dir>
# Compile a prerequisite module/submodule file to an object, with <inc_dir> on
# the include path so it can USE earlier prerequisites' .fmod files. ffc writes
# each module's .fmod next to the object, so passing -o inside <inc_dir> places
# the .fmod where the main file's -I search later finds it.
compile_object_with_ffc_inc() {
    local source="$1" object="$2" ffc="$3" inc_dir="$4"
    timeout "${FFC_COMPILE_TIMEOUT:-10}" \
        "$ffc" "$source" -c -o "$object" -I "$inc_dir" 2>/dev/null
    return $?
}

# compile_with_gfortran <source> <exe> [extra_sources...]
# Returns 0 on success, non-zero on failure. Extra source files (prerequisite
# module/submodule files for separate compilation) are compiled together with
# <source>; they are placed BEFORE the main source on the command line so the
# program's USE statements see freshly emitted .mod files. Each call writes its
# .mod files into a private temp directory so concurrent or sequential tests
# cannot collide on a shared module-output directory.
compile_with_gfortran() {
    local source="$1" exe="$2"
    shift 2
    local mod_dir status
    mod_dir=$(mktemp -d "${TMPDIR:-/tmp}/ffc_gfmod_XXXXXX") || return 1
    timeout "${GFORTRAN_COMPILE_TIMEOUT:-${FFC_COMPILE_TIMEOUT:-10}}" \
        gfortran -w -J "$mod_dir" "$@" "$source" -o "$exe" 2>/dev/null
    status=$?
    rm -rf "$mod_dir"
    return $status
}

# run_capture <exe> <out_file> <timeout_seconds>
# Runs the executable, captures stdout+stderr to out_file, respects timeout.
# Returns the exit status of the executable (or 124 if timed out).
# stdin is redirected from /dev/null: the caller drives the file list on the
# loop's stdin, and a test program that reads stdin (PAUSE, READ) must not steal
# those lines and corrupt later iterations.
run_capture() {
    local exe="$1" out_file="$2" timeout="$3"
    timeout "$timeout" "$exe" > "$out_file" 2>&1 < /dev/null
    return $?
}

# compare_outputs <ffc_out> <ref_out> <ffc_exit> <ref_exit>
# Returns 0 if both stdout and exit codes match, 1 otherwise.
compare_outputs() {
    local ffc_out="$1" ref_out="$2" ffc_exit="$3" ref_exit="$4"
    if [ "$ffc_exit" != "$ref_exit" ]; then
        return 1
    fi
    diff -q "$ffc_out" "$ref_out" > /dev/null 2>&1
    return $?
}

# numeric_structure <file>
# Emit the file's output with every numeric literal collapsed to a single 'N'
# token and runs of whitespace collapsed to one space. Two outputs share a
# numeric structure when they differ only in the magnitude and field width of
# their numbers, which is the signature of a nondeterministic timing/random
# program.
numeric_structure() {
    local file="$1"
    sed -E 's/[+-]?[0-9]+\.?[0-9]*([eEdD][+-]?[0-9]+)?/N/g; s/[[:space:]]+/ /g; s/^ //; s/ $//' \
        "$file"
}

# reference_is_nondeterministic <exe> <timeout>
# True when two runs of the reference executable produce different stdout but a
# matching numeric structure. Such programs (CPU_TIME, SYSTEM_CLOCK, RANDOM_*)
# cannot be compared byte-for-byte; they are compared structurally instead.
reference_is_nondeterministic() {
    local exe="$1" timeout="$2"
    local a b
    a=$(timeout "$timeout" "$exe" 2>&1 < /dev/null) || return 1
    b=$(timeout "$timeout" "$exe" 2>&1 < /dev/null) || return 1
    [ "$a" != "$b" ]
}

# compare_structural <ffc_out> <ref_out> <ffc_exit> <ref_exit>
# Like compare_outputs but ignores numeric magnitudes and field widths. Used
# only when the reference program is nondeterministic.
compare_structural() {
    local ffc_out="$1" ref_out="$2" ffc_exit="$3" ref_exit="$4"
    if [ "$ffc_exit" != "$ref_exit" ]; then
        return 1
    fi
    [ "$(numeric_structure "$ffc_out")" = "$(numeric_structure "$ref_out")" ]
}

# check_xfail <manifest_path> <suite_relative_path>
# Returns 0 if the path is listed in the xfail manifest, 1 otherwise.
check_xfail() {
    local manifest="$1" path="$2"
    if [ ! -f "$manifest" ]; then
        return 1
    fi
    local stripped
    stripped=$(echo "$path" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    grep -Fxq -- "$stripped" "$manifest"
}

# Escape a string for safe JSON embedding (handles backslash, quotes, newlines).
json_escape() {
    local s="$1"
    s="${s//\\/\\\\}"
    s="${s//\"/\\\"}"
    s="${s//$'\n'/\\n}"
    s="${s//$'\r'/\\r}"
    s="${s//$'\t'/\\t}"
    printf '%s' "$s"
}
