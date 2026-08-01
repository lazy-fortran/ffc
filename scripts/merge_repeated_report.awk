# merge_repeated_report.awk: merge per-attempt conformance reports.
#
# Reads the JSONL result records of N attempts over the same selection and
# writes one record per file. A file whose status is identical in every
# attempt keeps its first record verbatim. A file whose status differs
# between attempts becomes a FLAKY record listing the observed statuses, so
# an intermittent pass can never be read as a stable one.
#
# Usage: awk -v suite=SUITE -f merge_repeated_report.awk attempt_*.jsonl

function json_field(line, key,    pattern, piece) {
    pattern = "\"" key "\":\"[^\"]*\""
    if (match(line, pattern) == 0) return ""
    piece = substr(line, RSTART, RLENGTH)
    sub("^\"" key "\":\"", "", piece)
    sub("\"$", "", piece)
    return piece
}

/"status":"SUMMARY"/ { next }

{
    file_name = json_field($0, "file")
    status = json_field($0, "status")
    if (file_name == "" || status == "") next
    if (!(file_name in seen)) {
        order[++file_count] = file_name
        seen[file_name] = 1
        first_record[file_name] = $0
        first_status[file_name] = status
        observed[file_name] = status
    } else if (index("|" observed[file_name] "|", "|" status "|") == 0) {
        observed[file_name] = observed[file_name] "|" status
    }
    if (status != first_status[file_name]) unstable[file_name] = 1
    attempts[file_name]++
}

END {
    for (index_of_file = 1; index_of_file <= file_count; index_of_file++) {
        file_name = order[index_of_file]
        if (file_name in unstable) {
            printf "{\"suite\":\"%s\",\"file\":\"%s\",\"status\":\"FLAKY\",", \
                suite, file_name
            printf "\"note\":\"unstable across %d attempts\",", \
                attempts[file_name]
            printf "\"attempts\":%d,\"observed\":\"%s\"}\n", \
                attempts[file_name], observed[file_name]
        } else {
            print first_record[file_name]
        }
    }
}
