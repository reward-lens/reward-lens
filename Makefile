# The build machine's four verbs (PRODUCT_BUILD_PROMPT.md W0.2). PY defaults to the system
# interpreter until P-PKG lands a project environment; override with PY=.venv/bin/python.
PY ?= python3

.PHONY: demo gates test status verify-gates

demo:
	$(PY) fleet/golden/run_demo.py

gates:
	$(PY) fleet/gates/run_gates.py

test:
	$(PY) -m pytest -q

status:
	$(PY) fleet/status.py

verify-gates:
	$(PY) fleet/gates/verify_gate_list.py
