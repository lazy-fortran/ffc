function report_error(message) {
    printf "ERROR: %s:%d: %s\n", source, NR, message > "/dev/stderr"
    failed = 1
    exit 1
}

function skip_space() {
    while (position <= line_length &&
            substr(json_line, position, 1) ~ /[[:space:]]/) position++
}

function parse_string(    result, character, escaped, unicode) {
    if (substr(json_line, position, 1) != "\"") {
        report_error("expected JSON string")
    }
    position++
    result = ""
    while (position <= line_length) {
        character = substr(json_line, position++, 1)
        if (character == "\"") return result
        if (character == "\\") {
            if (position > line_length) report_error("truncated JSON escape")
            escaped = substr(json_line, position++, 1)
            if (escaped == "u") {
                unicode = substr(json_line, position, 4)
                if (length(unicode) != 4 || unicode !~ /^[0-9A-Fa-f]{4}$/) {
                    report_error("malformed JSON unicode escape")
                }
                result = result "\\u" unicode
                position += 4
            } else if (escaped ~ /^["\\\/bfnrt]$/) {
                result = result "\\" escaped
            } else {
                report_error("malformed JSON escape")
            }
        } else {
            if (character ~ /[[:cntrl:]]/) {
                report_error("control character in JSON string")
            }
            result = result character
        }
    }
    report_error("unterminated JSON string")
}

function parse_integer(    start, character, result) {
    start = position
    if (substr(json_line, position, 1) == "-") position++
    character = substr(json_line, position, 1)
    if (character !~ /^[0-9]$/) report_error("malformed JSON number")
    if (character == "0" && substr(json_line, position + 1, 1) ~ /^[0-9]$/) {
        report_error("leading zero in JSON number")
    }
    while (substr(json_line, position, 1) ~ /^[0-9]$/) position++
    result = substr(json_line, start, position - start)
    if (substr(json_line, position, 1) ~ /^[.eE]$/) {
        report_error("non-integer JSON number")
    }
    return result
}

function parse_value(key,    character) {
    character = substr(json_line, position, 1)
    if (character == "\"") {
        field_type[key] = "string"
        field_value[key] = parse_string()
    } else if (substr(json_line, position, 4) == "true") {
        field_type[key] = "boolean"
        field_value[key] = 1
        position += 4
    } else if (substr(json_line, position, 5) == "false") {
        field_type[key] = "boolean"
        field_value[key] = 0
        position += 5
    } else if (character == "-" || character ~ /^[0-9]$/) {
        field_type[key] = "integer"
        field_value[key] = parse_integer()
    } else {
        report_error("unsupported JSON value")
    }
}

function parse_record(    key, separator, old_key) {
    for (old_key in field_value) {
        delete field_value[old_key]
        delete field_type[old_key]
    }
    json_line = $0
    line_length = length(json_line)
    position = 1
    skip_space()
    if (substr(json_line, position++, 1) != "{") {
        report_error("record is not a JSON object")
    }
    skip_space()
    if (substr(json_line, position, 1) == "}") position++
    else {
        while (1) {
            key = parse_string()
            if (key in field_value) report_error("duplicate JSON key: " key)
            skip_space()
            if (substr(json_line, position++, 1) != ":") {
                report_error("missing JSON colon")
            }
            skip_space()
            parse_value(key)
            skip_space()
            separator = substr(json_line, position++, 1)
            if (separator == "}") break
            if (separator != ",") report_error("missing JSON comma")
            skip_space()
        }
    }
    skip_space()
    if (position <= line_length) report_error("trailing data after JSON object")
}

function require_type(key, expected) {
    if (!(key in field_value)) report_error("missing field: " key)
    if (field_type[key] != expected) report_error("wrong field type: " key)
}

function require_string(key) {
    require_type(key, "string")
    return field_value[key]
}

function require_integer(key, nonnegative,    number) {
    require_type(key, "integer")
    number = field_value[key] + 0
    if (nonnegative && number < 0) report_error("negative counter: " key)
    return number
}

function require_boolean(key) {
    require_type(key, "boolean")
    return field_value[key]
}

function require_revision(key,    revision) {
    revision = require_string(key)
    if (length(revision) != 40 || revision !~ /^[0-9A-Fa-f]+$/ ||
            revision ~ /^0+$/) report_error("malformed revision field: " key)
    return revision
}

function require_digest(key,    digest) {
    digest = require_string(key)
    if (length(digest) != 64 || digest !~ /^[0-9A-Fa-f]+$/ ||
            digest ~ /^0+$/) report_error("malformed digest field: " key)
    return digest
}

function row_field_allowed(key) {
    return key == "suite" || key == "file" || key == "status" ||
        key == "ffc_exit" || key == "ref_exit" || key == "note" ||
        key == "noref" || key == "noref_reason" ||
        key == "noref_manifest_category" ||
        key == "warning_expectation" ||
        key == "attempts" || key == "observed" ||
        key == "observed_status" || key == "expectation"
}

function summary_field_allowed(key) {
    return key == "suite" || key == "status" || key == "pass" ||
        key == "xfail" || key == "xpass" || key == "fail" ||
        key == "noref" || key == "skip" || key == "warning_unchecked" ||
        key == "total" || key == "flaky" ||
        key == "schema_version" || key == "full_run" ||
        key == "sampled" || key == "sample_size" ||
        key == "sample_population" || key == "sample_seed" ||
        key == "sample_margin_pct" ||
        key == "provenance_verified" ||
        key == "ffc_revision" || key == "ffc_source_sha256" ||
        key == "ffc_binary_sha256" || key == "fortfront_revision" ||
        key == "fortfront_tree" || key == "liric_revision" ||
        key == "liric_tree" || key == "corpus_revision" ||
        key == "corpus_tree" || key == "corpus_files_sha256" ||
        key == "worktree" || key == "report_kind" ||
        key == "observation_schema_version" ||
        key == "reference_compiler" ||
        key == "reference_cache_enabled" ||
        key == "reference_cache_hits" || key == "timeout_seconds" ||
        key == "skip_manifest_sha256" || key == "noref_manifest_sha256" ||
        key == "classification_mode" || key == "observation_sha256" ||
        key == "classification_manifest_sha256" || key == "attempt_count"
}

function validate_summary(    key, suite, pass_count, xfail_count, xpass_count,
        fail_count, noref_count, skip_count, warning_count, total_count,
        flaky_count, ffc_revision, source_digest, binary_digest, fortfront_revision,
        fortfront_tree, liric_revision, liric_tree, corpus_revision,
        corpus_tree, corpus_files_digest) {
    for (key in field_value) {
        if (!summary_field_allowed(key)) report_error("unknown SUMMARY field: " key)
    }
    suite = require_string("suite")
    if (suite != expected_suite) report_error("mixed or unexpected suite: " suite)
    if (require_string("status") != "SUMMARY") report_error("invalid SUMMARY")
    pass_count = require_integer("pass", 1)
    xfail_count = require_integer("xfail", 1)
    xpass_count = require_integer("xpass", 1)
    fail_count = require_integer("fail", 1)
    noref_count = require_integer("noref", 1)
    skip_count = require_integer("skip", 1)
    warning_count = require_integer("warning_unchecked", 1)
    total_count = require_integer("total", 1)
    flaky_count = 0
    if ("flaky" in field_value) flaky_count = require_integer("flaky", 1)
    if (counts["FLAKY"] != flaky_count) report_error("SUMMARY flaky mismatch")
    if (require_integer("schema_version", 1) != 1) {
        report_error("unknown report schema version")
    }
    if ("sampled" in field_value && require_boolean("sampled")) {
        report_error("sampled report is an estimate, not a dashboard input")
    }
    if (!require_boolean("full_run")) report_error("report is not a full run")
    if (!require_boolean("provenance_verified")) {
        report_error("report provenance is not verified")
    }
    ffc_revision = require_revision("ffc_revision")
    source_digest = require_digest("ffc_source_sha256")
    binary_digest = require_digest("ffc_binary_sha256")
    fortfront_revision = require_revision("fortfront_revision")
    fortfront_tree = require_revision("fortfront_tree")
    liric_revision = require_revision("liric_revision")
    liric_tree = require_revision("liric_tree")
    corpus_revision = require_revision("corpus_revision")
    corpus_tree = require_revision("corpus_tree")
    corpus_files_digest = require_digest("corpus_files_sha256")
    # ffc #642: a report must name the worktree that produced it.
    require_string("worktree")
    if ("report_kind" in field_value) {
        if (require_string("report_kind") != "classification") {
            report_error("dashboard input is not a classification view")
        }
        if (require_integer("observation_schema_version", 1) != 1) {
            report_error("unknown observation schema version")
        }
        require_string("reference_compiler")
        require_boolean("reference_cache_enabled")
        require_integer("reference_cache_hits", 1)
        require_integer("timeout_seconds", 1)
        require_digest("skip_manifest_sha256")
        require_digest("noref_manifest_sha256")
        if (require_string("classification_mode") != "manifest") {
            report_error("xfail-disabled view is not a dashboard input")
        }
        require_digest("observation_sha256")
        require_digest("classification_manifest_sha256")
        if (classified_row_count != row_count) {
            report_error("classification row lacks observation fields")
        }
        if ("attempt_count" in field_value &&
                require_integer("attempt_count", 1) < 2) {
            report_error("invalid repeat attempt count")
        }
    } else if (classified_row_count != 0) {
        report_error("classification fields without classification SUMMARY")
    }
    if (row_count != total_count) report_error("SUMMARY total mismatch")
    if (counts["PASS"] != pass_count || counts["XFAIL"] != xfail_count ||
            counts["XPASS"] != xpass_count || counts["FAIL"] != fail_count ||
            counts["SKIP"] != skip_count) {
        report_error("SUMMARY status count mismatch")
    }
    if (observed_noref != noref_count) report_error("SUMMARY NOREF mismatch")
    if (observed_warning != warning_count) {
        report_error("SUMMARY warning count mismatch")
    }
    print suite, pass_count, xfail_count, xpass_count, fail_count,
        noref_count, skip_count, warning_count, total_count, ffc_revision,
        source_digest, binary_digest, fortfront_revision, fortfront_tree,
        liric_revision, liric_tree, corpus_revision, corpus_tree,
        corpus_files_digest >> summaries
}

function noref_reason_allowed(reason) {
    return reason == "reference-rejected" ||
        reason == "reference-runtime-failure" ||
        reason == "undefined-runtime-value" ||
        reason == "missing-external-definition" ||
        reason == "compile-only"
}

function validate_row(    key, suite, status, file_name, has_ffc, has_ref,
        is_noref, noref_reason, warning, observed_status, expectation,
        expected_status) {
    for (key in field_value) {
        if (!row_field_allowed(key)) report_error("unknown result field: " key)
    }
    suite = require_string("suite")
    if (suite != expected_suite) report_error("mixed or unexpected suite: " suite)
    status = require_string("status")
    if (status != "PASS" && status != "XFAIL" && status != "XPASS" &&
            status != "FAIL" && status != "SKIP" && status != "FLAKY") {
        report_error("unknown status: " status)
    }
    file_name = require_string("file")
    if (file_name == "") report_error("empty file field")
    require_string("note")
    if (seen[file_name]++) report_error("duplicate report row: " file_name)
    has_ffc = "ffc_exit" in field_value
    has_ref = "ref_exit" in field_value
    if (has_ffc != has_ref) report_error("result has only one exit field")
    if (has_ffc) {
        require_integer("ffc_exit", 0)
        require_integer("ref_exit", 0)
    } else if (status != "SKIP" && status != "FLAKY" &&
            !(status == "FAIL" && field_value["note"] ~ /directive/)) {
        report_error("result is missing exit fields")
    }
    if (status == "FLAKY") {
        if (require_integer("attempts", 1) < 2) {
            report_error("FLAKY row needs at least two attempts")
        }
        if (require_string("observed") !~ /\|/) {
            report_error("FLAKY row needs differing observed statuses")
        }
    } else if ("attempts" in field_value || "observed" in field_value) {
        report_error("attempt fields outside a FLAKY row")
    }
    if (("observed_status" in field_value) != ("expectation" in field_value)) {
        report_error("incomplete observation classification fields")
    }
    if ("observed_status" in field_value) {
        classified_row_count++
        observed_status = require_string("observed_status")
        expectation = require_string("expectation")
        if (observed_status != "PASS" && observed_status != "FAIL" &&
                observed_status != "SKIP" && observed_status != "FLAKY") {
            report_error("invalid observed status: " observed_status)
        }
        if (expectation != "none" && expectation != "xfail") {
            report_error("invalid expectation: " expectation)
        }
        expected_status = observed_status
        if (expectation == "xfail" && observed_status == "PASS")
            expected_status = "XPASS"
        else if (expectation == "xfail" && observed_status == "FAIL")
            expected_status = "XFAIL"
        if (status != expected_status) {
            report_error("classified status does not match observation")
        }
    }
    is_noref = 0
    if ("noref" in field_value) {
        is_noref = require_boolean("noref")
        if (!is_noref) report_error("noref must be true when present")
        if (status != "PASS" && status != "XPASS") {
            report_error("NOREF row has incompatible status: " status)
        }
        if (!has_ffc) report_error("NOREF row has incompatible exit fields")
        noref_reason = require_string("noref_reason")
        if (!noref_reason_allowed(noref_reason)) {
            report_error("unapproved noref reason: " noref_reason)
        }
        if (noref_reason == "reference-rejected" &&
                field_value["ref_exit"] == 0) {
            report_error("NOREF row has incompatible exit fields")
        }
        if (noref_reason == "reference-runtime-failure" &&
                (field_value["ffc_exit"] != 0 ||
                 field_value["ref_exit"] == 0)) {
            report_error("NOREF row has incompatible runtime exit fields")
        }
    } else if ("noref_reason" in field_value) {
        report_error("noref_reason without noref")
    }
    if ("noref_manifest_category" in field_value) {
        noref_reason = require_string("noref_manifest_category")
        if (noref_reason != "undefined-runtime-value" &&
                noref_reason != "missing-external-definition" &&
                noref_reason != "compile-only") {
            report_error("unapproved noref manifest category")
        }
        if (expectation == "xfail") {
            report_error("files cannot be both xfail and noref")
        }
    }
    warning = 0
    if ("warning_expectation" in field_value) {
        if (require_string("warning_expectation") != "unchecked") {
            report_error("invalid warning expectation")
        }
        if (suite != "gfortran-dg") {
            report_error("warning expectation outside gfortran-dg")
        }
        warning = 1
    }
    row_count++
    counts[status]++
    observed_noref += is_noref
    observed_warning += warning
    print suite, file_name, status, is_noref, warning >> rows
}

BEGIN {
    summary_seen = 0
    row_count = 0
    classified_row_count = 0
}

{
    if (summary_seen) report_error("SUMMARY must be the final record")
    parse_record()
    require_type("status", "string")
    if (field_value["status"] == "SUMMARY") {
        summary_seen = 1
        validate_summary()
    } else {
        validate_row()
    }
}

END {
    if (!failed && !summary_seen) {
        printf "ERROR: %s: missing SUMMARY\n", source > "/dev/stderr"
        exit 1
    }
}
