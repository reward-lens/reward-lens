# Demo numbers, measured

The numbers the wave-1 transcripts print, measured on this machine from the example
`reward_lens.examples.code_reward` as written by `reward-lens init --example code-reward` and
audited by the integrated engine (P-EXAMPLE attempt 3, P-AUDIT-1 attempts 6 to 12, P-CLI attempts
4 to 6; 2026-09-13). This file's presence switches the harness's numeral mask off
(`fleet/golden/VOLATILE.md`, `numerals_until_demo_numbers`): from here on every numeral in the listed
transcripts is compared as written. Under D-78 a transcript takes the measured number, never the
reverse, and each change to a golden is recorded in `fleet/DECISIONS.md` (A-020, A-023, D-78).

| what | value | where it prints |
| --- | --- | --- |
| tasks | 40 | `40 tasks`; `tasks.jsonl  40 code tasks` |
| responses | 120 | `120 responses`; `responses.jsonl  120 sampled responses from a small model` |
| files `write()` produces | 31 | not printed |
| written project tree | 57,947 bytes | not printed |
| panels the demo plans | 2 | `Plan  2 panels` |
| paid calls | 0 | `0 paid calls` |
| the plan's estimate | 2 s | `about 2 s` (the target's `about 30 s` was a placeholder) |
| validity checks passed | 4 of 5 | `4 of 5 checks passed` |
| validity checks not in this build | 7 | `7 checks are not in this build` |
| task-set checks on a bare grader | 3 | `the 3 task-set checks need a task set` |
| validity checks not in this build, bare grader | 4 | `4 are not in this build` |
| sampled responses the signal panel asks for | 128 | `128 sampled responses  signal, cost, selection` (the target's `120` was a placeholder) |
| the line the audit finds | `grader.py:91` | `grader.py line 91 reads outcome/test_solution.py` (the target's `line 14` was a placeholder) |
| reach crossings on the demo | 1 (`REACH-EDIT-TESTS`) | `1 surface exposed, by static analysis only` |
| blocking findings on the demo | 1 | `1 blocking finding` |
| required panels that could not run | 8 | `8 required panels could not run` (the target's `7` was a placeholder; A-023) |
| replay repeats on a bare grader | 100 | `100 repeats, identical output, no hidden state` |
| the report's size | about 381 KB | masked as `<size>` (`file-size` rule) |
| the run's wall time | under a second | masked as `<dur>` |
| the record's name | `code-reward-v1-<date>` | the Report line, the date masked |

The example package on disk is 88,887 bytes. The twelve exploiting responses, by id, for anyone
reproducing by hand: r-01-3, r-02-3, r-03-3, r-11-3, r-12-3, r-13-3, r-21-3, r-22-3, r-23-3, r-31-3,
r-32-3, r-33-3 (P-EXAMPLE attempt 2's table).
