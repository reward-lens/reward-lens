# Volatile fields

`fleet/golden/run_demo.py` masks these on both sides before it diffs a captured transcript against
the golden file of the current wave. The fenced JSON block below is the machine-readable rule set
and the only source; the prose explains it. A rule applies to the file globs it names.

Declared volatile by the commission (section 5): durations, timestamps, absolute paths, run ids,
file sizes, the machine block, any date inside a filename or path, and the `uv tool install` line
with its output, which is captured from the real tool rather than trusted. Digest values are
volatile too: their presence and shape are checked, their hex is not, because a digest depends on
bytes the golden cannot know.

Until `fleet/golden/DEMO_NUMBERS.md` exists, every numeral in the demo transcripts is masked as
well, because the numbers there are volatile until P-EXAMPLE lands and the real example's values
are captured, frozen and recorded in that file with the commit that fixed them (section 5). The
example is never tuned to hit a number written first. Once `DEMO_NUMBERS.md` exists the numeral
mask switches off and only the rules below remain.

```json
{
  "noexec_commands": ["uv tool install reward-lens"],
  "rules": [
    {"name": "duration", "files": ["*.txt"], "pattern": "\\b\\d+m \\d+s\\b|\\b\\d+(\\.\\d+)? ?(ms|s)\\b(?! per)", "replace": "<dur>"},
    {"name": "duration-json", "files": ["*.json"], "pattern": "\"duration_s\": \\d+(\\.\\d+)?", "replace": "\"duration_s\": <dur>"},
    {"name": "timestamp", "files": ["*"], "pattern": "\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}Z", "replace": "<ts>"},
    {"name": "date-in-path", "files": ["*"], "pattern": "-\\d{4}-\\d{2}-\\d{2}(?=[.\\w/])", "replace": "-<date>"},
    {"name": "abs-path", "files": ["*"], "pattern": "/(home|tmp|private|var|Users)/[^\\s\"']+", "replace": "<abspath>"},
    {"name": "run-id", "files": ["*"], "pattern": "\\brun[-:][0-9a-f]{6,}\\b", "replace": "run-<id>"},
    {"name": "digest", "files": ["*"], "pattern": "sha256:[0-9a-f.\\u2026]+", "replace": "sha256:<hex>"},
    {"name": "file-size", "files": ["*.txt"], "pattern": "\\b\\d+(\\.\\d+)? (KB|MB|B)\\b", "replace": "<size>"},
    {"name": "machine-block", "files": ["*.json"], "pattern": "\"environment_excluded_from_digest\": \\{[^}]*\\}", "replace": "\"environment_excluded_from_digest\": {}"},
    {"name": "probe-ms", "files": ["*.txt"], "pattern": "probed in \\d+ ms", "replace": "probed in <dur>"},
    {"name": "sandbox-tier-line", "files": ["doctor.txt"], "pattern": "(?m)^    [LMWT][0-9]: .*$", "replace": "    <sandbox-tier-line>"},
    {"name": "sandbox-tier-json", "files": ["*.json"], "pattern": "\"sandbox_tier\": \"[LMWT][0-9]\"", "replace": "\"sandbox_tier\": \"<tier>\""},
    {"name": "cost-wall", "files": ["*.json", "*.txt"], "pattern": "\"wall_s\": [0-9.]+", "replace": "\"wall_s\": <duration>"},
    {"name": "embedding-html-bytes", "files": ["*.json"], "pattern": "\"html_bytes\": [0-9]+", "replace": "\"html_bytes\": <size>"},
    {"name": "embedding-record-bytes", "files": ["*.json"], "pattern": "\"record_bytes\": [0-9]+", "replace": "\"record_bytes\": <size>"},
    {"name": "install-path", "files": ["*.txt"], "pattern": "python 3\\.\\d+\\.\\d+  ·  \\S+(?: \\S+)*  ·  \\S+", "replace": "python <ver>  ·  <os>  ·  <abspath>"},
    {"name": "version-line", "files": ["doctor.txt", "root-help.txt"], "pattern": "(?m)^(\\s*)reward-lens \\d+\\.\\d+\\.\\d+\\S*(.*)$", "replace": "\\1reward-lens <version>\\2"}
  ],
  "numerals_until_demo_numbers": {"files": ["first-hour.txt", "audit-unfamiliar.txt", "compare-reject.txt", "improve.txt", "audit-envelope.json", "init-detect.txt", "forecast.txt", "trace-fixture.txt"], "pattern": "(?<![A-Za-z_-])\\d+(\\.\\d+)?%?", "replace": "<n>"}
}
```

Not volatile, ever: the wording, ordering, glyphs, panel set and section names; which panels are
lit; the verdict word; the exit code; the presence of a report path and of the `Next` block.
