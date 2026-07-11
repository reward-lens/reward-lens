# Studies

**Who decides whether a result confirmed the hypothesis, the author or the spec?** In `reward_lens.studies` it is the spec, and it decides before the data exists. You write down the hypothesis, the predictions, and the kill criteria, freeze that into a content-hashed artifact, then run it. The run adjudicates the frozen predictions, not whatever the analysis wishes it had predicted. The discipline is [studies and preregistration](../discipline/studies-and-preregistration.md).

## The spec

A `StudySpec` collects a `Hypothesis`, its `Prediction`s, the `KillCriterion`s that would sink it, and a `SubjectQuery` naming what to run it on. A prediction knows how to check a value against itself; a kill criterion knows when it has fired.

::: reward_lens.studies.spec.StudySpec
    options:
      heading_level: 3

::: reward_lens.studies.spec.Hypothesis
    options:
      heading_level: 3

::: reward_lens.studies.spec.Prediction
    options:
      heading_level: 3

::: reward_lens.studies.spec.KillCriterion
    options:
      heading_level: 3

::: reward_lens.studies.spec.SubjectQuery
    options:
      heading_level: 3

## Freeze and run

`freeze` stamps the spec with its content hash and the git sha into a `FrozenStudy`, giving it an identity of the form `study:{id}@v{version}#{hash8}`. `run_study` then executes it and adjudicates the frozen predictions, returning an outcome of `confirmed`, `refuted`, or `inconclusive`. Only a study that predated its data can reach the `REGISTERED` rung of trust. The worked version is [freeze and run a study](../how-to/freeze-and-run-a-study.md).

::: reward_lens.studies.freeze.freeze
    options:
      heading_level: 3

::: reward_lens.studies.runner.run_study
    options:
      heading_level: 3

## The scoreboard

`Scoreboard` collects the standing claims, the theorems and the candidate laws, and their current status as registered evidence accrues for or against each one.

::: reward_lens.studies.scoreboard.Scoreboard
    options:
      heading_level: 3
