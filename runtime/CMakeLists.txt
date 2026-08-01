# Backend-qualified LIRIC runtime archives for ffc (issue #374).
#
# Produces, under a target-qualified artifact directory, one archive per
# supported LIRIC backend:
#
#   <artifact_dir>/ffc-runtime-v2-isel.lrarch
#   <artifact_dir>/ffc-runtime-v2-copy-patch.lrarch
#   <artifact_dir>/ffc-runtime-v2-llvm.lrarch   (only when LLVM is available)
#
# The LIRIC session backend `default` is an alias for copy-patch, so it has no
# archive of its own; a loader asking for `default` selects the copy-patch
# artifact.
#
# This project is deliberately standalone: ffc itself is an fpm project, and
# this only produces data artifacts, never a library ffc links against.

cmake_minimum_required(VERSION 3.16)
project(ffc_runtime C)

# Archive naming version. Bumped when the artifact layout or the archive
# payload contract changes, independent of the LIRIC archive format version
# recorded inside each file.
set(FFC_RUNTIME_ARTIFACT_VERSION "v2")

# --- Dependency discovery, each with a named diagnostic ---------------------

find_program(FFC_RUNTIME_CLANG NAMES clang)
if(NOT FFC_RUNTIME_CLANG)
    message(FATAL_ERROR
        "ffc runtime dependency missing: clang. "
        "clang is required to compile runtime/ffc_runtime.c to LLVM bitcode. "
        "Install clang or set -DFFC_RUNTIME_CLANG=<path>.")
endif()

find_program(FFC_RUNTIME_ARCHIVE_TOOL
    NAMES liric_runtime_archive
    HINTS
        ${LIRIC_BUILD_DIR}
        $ENV{LIRIC_BUILD_DIR}
        # Sibling LIRIC checkout, both from the primary checkout and from a
        # git worktree one level deeper.
        ${CMAKE_CURRENT_SOURCE_DIR}/../../liric/build
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../liric/build
    NO_DEFAULT_PATH)
if(NOT FFC_RUNTIME_ARCHIVE_TOOL)
    find_program(FFC_RUNTIME_ARCHIVE_TOOL NAMES liric_runtime_archive)
endif()
if(NOT FFC_RUNTIME_ARCHIVE_TOOL)
    message(FATAL_ERROR
        "ffc runtime dependency missing: liric_runtime_archive. "
        "Build LIRIC first, then re-configure with "
        "-DLIRIC_BUILD_DIR=<liric-build-dir>.")
endif()

# --- Target qualification ---------------------------------------------------

# The archive records the LIRIC target it was built for, and the artifact
# directory repeats it so archives for different targets never collide.
if(NOT FFC_RUNTIME_TARGET)
    set(FFC_RUNTIME_TARGET "host")
endif()
set(FFC_RUNTIME_ARTIFACT_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/artifacts/${FFC_RUNTIME_TARGET}")
file(MAKE_DIRECTORY "${FFC_RUNTIME_ARTIFACT_DIR}")

# --- Runtime bitcode --------------------------------------------------------

set(FFC_RUNTIME_BC "${CMAKE_CURRENT_BINARY_DIR}/ffc_runtime.bc")
add_custom_command(
    OUTPUT "${FFC_RUNTIME_BC}"
    # C flags only, spelled out here rather than inherited. This project
    # produces a C artifact, and a Fortran-only flag reaching this command
    # (-J, -ffree-form, -fimplicit-none) is a hard error from clang rather
    # than something to discover later. The language level matches the one
    # the runtime source states for itself.
    COMMAND "${FFC_RUNTIME_CLANG}" -x c -std=c11 -emit-llvm -c -O1
            -Wall -Wextra -Werror
            -o "${FFC_RUNTIME_BC}"
            "${CMAKE_CURRENT_SOURCE_DIR}/ffc_runtime.c"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ffc_runtime.c"
    COMMENT "Compiling ffc runtime to LLVM bitcode"
    VERBATIM)

# --- One archive per backend ------------------------------------------------

# LLVM archives can only be produced when the LIRIC build exposes the LLVM
# backend. Callers that want it opt in explicitly so a LIRIC build without
# LLVM does not silently skip a requested artifact.
set(FFC_RUNTIME_BACKENDS isel copy_patch)
if(FFC_RUNTIME_ENABLE_LLVM)
    list(APPEND FFC_RUNTIME_BACKENDS llvm)
endif()

set(FFC_RUNTIME_ARCHIVES "")
foreach(backend IN LISTS FFC_RUNTIME_BACKENDS)
    # Artifact names use a dash; the tool's backend token uses an underscore.
    string(REPLACE "_" "-" backend_name "${backend}")
    set(archive
        "${FFC_RUNTIME_ARTIFACT_DIR}/ffc-runtime-${FFC_RUNTIME_ARTIFACT_VERSION}-${backend_name}.lrarch")
    if(FFC_RUNTIME_TARGET STREQUAL "host")
        set(target_args "")
    else()
        set(target_args --target "${FFC_RUNTIME_TARGET}")
    endif()
    add_custom_command(
        OUTPUT "${archive}"
        COMMAND "${FFC_RUNTIME_ARCHIVE_TOOL}"
                --input-bc "${FFC_RUNTIME_BC}"
                --backend "${backend}"
                ${target_args}
                --output "${archive}"
        DEPENDS "${FFC_RUNTIME_BC}"
        COMMENT "Packaging ffc runtime archive for backend ${backend_name}"
        VERBATIM)
    list(APPEND FFC_RUNTIME_ARCHIVES "${archive}")
endforeach()

add_custom_target(ffc_runtime_archives ALL DEPENDS ${FFC_RUNTIME_ARCHIVES})
