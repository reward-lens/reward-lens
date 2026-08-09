# Repository guidance

Use `reward-lens` in prose and commands, `reward_lens` for Python imports, and `REWARD_LENS_*` for environment variables. The current source target is 3.1.0 and remains unreleased. Do not change established public symbols, serialized keys, or configuration names without a compatibility review.

Keep reusable library code, schemas, small fixtures, tests, and user documentation in this repository. Keep private operations records, raw run output, provider credentials, account configuration, and large experimental artifacts outside it. Do not add a workflow that deploys a site, publishes a package, creates a release, provisions cloud resources, or runs paid or GPU work on a normal push.

Run supported checks in an isolated environment. `TEST_STATUS.md` separates the CPU CI contract from optional and incomplete suites. Update documentation and compatibility tests with a public behavior change.
