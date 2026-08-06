.PHONY: check-rejection-gate

# Run before merging a change that can reject source.  The baseline is a
# compile-only observation of the pinned FortFront examples.  Leave the corpus
# variable empty to let the script resolve sibling checkouts from either a
# normal checkout or an isolated /mnt/storage worktree.
FFC_REJECTION_GATE_CORPUS ?=
FFC_REJECTION_GATE_BIN ?= build/fo/bin/ffc
FFC_REJECTION_GATE_OUT ?= build/corpus_rejection_current.tsv
FFC_REJECTION_GATE_BASELINE ?= test/fixtures/corpus_rejection_baseline.tsv

check-rejection-gate:
	@if [ -n "$(FFC_REJECTION_GATE_CORPUS)" ]; then \
	    bash scripts/corpus_rejection_gate.sh \
	        --corpus "$(FFC_REJECTION_GATE_CORPUS)" \
	        --ffc "$(FFC_REJECTION_GATE_BIN)" \
	        --out "$(FFC_REJECTION_GATE_OUT)" \
	        --baseline "$(FFC_REJECTION_GATE_BASELINE)"; \
	else \
	    fortfront_dir=$$FFC_FORTFRONT_DIR; \
	    if [ -z "$$fortfront_dir" ]; then \
	        for candidate in "../code/lazy-fortran/fortfront" "../fortfront" \
	            "../../code/lazy-fortran/fortfront"; do \
	            if [ -d "$$candidate/examples" ]; then fortfront_dir="$$candidate"; break; fi; \
	        done; \
	    fi; \
	    test -d "$$fortfront_dir/examples" || { echo "FortFront examples not found; set FFC_REJECTION_GATE_CORPUS or FFC_FORTFRONT_DIR" >&2; exit 2; }; \
	    bash scripts/corpus_rejection_gate.sh \
	        --corpus "$$fortfront_dir/examples" \
	        --ffc "$(FFC_REJECTION_GATE_BIN)" \
	        --out "$(FFC_REJECTION_GATE_OUT)" \
	        --baseline "$(FFC_REJECTION_GATE_BASELINE)"; \
	fi
