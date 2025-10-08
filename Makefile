.PHONY: report

# Environment variables expected:
#   NPY   - path to the envelope matrix (required)
#   META  - path to metadata JSON (required)
#   OUT   - output directory (default: outputs/stats)
#   DATASETS - dataset labels (default: testing)
#   TARGET_TRIALS - target trials for Group A (default: 2,4,5)
#   TIME_HZ - sampling rate (default: 40)
#   RUN_ALL_FLAGS - additional flags passed to stats/run_all.py
#   SUMMARY_ALPHA - alpha level for summary (default: 0.05)

OUT ?= outputs/stats
DATASETS ?= testing
TARGET_TRIALS ?= 2,4,5
TIME_HZ ?= 40
RUN_ALL_FLAGS ?=
SUMMARY_ALPHA ?= 0.05

report:
	@if [ -z "$(NPY)" ] || [ -z "$(META)" ]; then \
		echo "NPY and META variables must be set"; \
		exit 1; \
	fi
	python stats/run_all.py \
		--npy "$(NPY)" \
		--meta "$(META)" \
		--out "$(OUT)" \
		--datasets "$(DATASETS)" \
		--target-trials "$(TARGET_TRIALS)" \
		--time-hz "$(TIME_HZ)" \
	$(RUN_ALL_FLAGS)
	python stats/summarize_results.py \
		--in "$(OUT)" \
		--out "$(OUT)/summary.txt" \
		--alpha "$(SUMMARY_ALPHA)"
