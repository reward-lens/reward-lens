# Error codes

Every refusal reward-lens makes carries a code that does not change between releases.
The code is on the second line of the error, and the long form of any of them prints
with the network off:

```text
reward-lens explain RL0341
```

## Every code

| Code | What it is | Exit |
| --- | --- | --- |
| [RL0001](RL0001.md) | the invocation is not one this build can carry out | 4 |
| [RL0002](RL0002.md) | an identifier you supplied is not usable | 4 |
| [RL0003](RL0003.md) | a field in the project file is not valid | 4 |
| [RL0004](RL0004.md) | the output format asked for is not one this build writes | 4 |
| [RL0120](RL0120.md) | the reach panel was given no task set | 4 |
| [RL0130](RL0130.md) | the run was interrupted | 130 |
| [RL0201](RL0201.md) | the grader reads a file the graded code can write | 2 |
| [RL0202](RL0202.md) | the answer is reachable from inside the graded process | 2 |
| [RL0203](RL0203.md) | a branch of the grader never decides anything | 2 |
| [RL0210](RL0210.md) | the grader returns nothing on malformed input | 2 |
| [RL0211](RL0211.md) | the grader raises on input it will meet | 2 |
| [RL0212](RL0212.md) | the grader returns a boolean where a score belongs | 2 |
| [RL0213](RL0213.md) | a reward function returned None and the trainer would make it NaN | 2 |
| [RL0214](RL0214.md) | a reward function raised and the framework would score it 0.0 | 2 |
| [RL0215](RL0215.md) | a reward function returned a boolean and float() accepted it | 2 |
| [RL0231](RL0231.md) | the policy's context discloses the grader | 2 |
| [RL0301](RL0301.md) | the correctness claim is not qualified by an outcome check | 2 |
| [RL0302](RL0302.md) | this candidate set has already had its one acceptance round | 4 |
| [RL0341](RL0341.md) | the outcome check collected no tests | 4 |
| [RL0401](RL0401.md) | no sandbox tier is available on this machine | 5 |
| [RL0402](RL0402.md) | the sandbox tier asked for is stronger than this machine holds | 5 |
| [RL0410](RL0410.md) | an execution limit was breached | 7 |
| [RL0501](RL0501.md) | the money cap was reached | 6 |
| [RL0601](RL0601.md) | the record was written by a later version | 4 |
| [RL0602](RL0602.md) | an entry does not carry the evidence its kind calls for | 4 |
| [RL0603](RL0603.md) | a number carries more precision than it was measured at | 4 |
| [RL0604](RL0604.md) | the record does not match the schema | 4 |
| [RL0620](RL0620.md) | something this record depends on has changed since it was written | 4 |
| [RL0621](RL0621.md) | no run by that name or id | 4 |
| [RL0701](RL0701.md) | the feature asked for is not in this build | 5 |
| [RL0702](RL0702.md) | the adapter cannot prove it holds a capability the run needs | 5 |
| [RL0703](RL0703.md) | the verb is not in this build | 5 |
| [RL0710](RL0710.md) | the reward has a shape this build does not adapt | 5 |
| [RL0801](RL0801.md) | a decision only you can make is waiting, and there is no terminal to ask on | 3 |
| [RL0900](RL0900.md) | reward-lens failed internally | 7 |

## What the exit statuses mean

| Exit | Meaning |
| --- | --- |
| 0 | the work completed, and any decision it was asked for qualified. |
| 1 | the decision asked for was rejected by the policy it was put to. |
| 2 | the decision asked for is unresolved, because the evidence is missing or inconclusive. Rejected and unresolved never collapse into each other. |
| 3 | a decision only you can make is pending, and there was no terminal to ask on. The payload names the flag and the command to run. |
| 4 | usage, configuration or input contract. Nothing was measured and no record was written. Many shells use 2 for a usage mistake; here 2 never means that. |
| 5 | a capability, service or extra the run needed is unavailable, so the measurement that rests on it cannot be made. |
| 6 | the money cap was reached. The record up to that point was written. |
| 7 | reward-lens itself failed, or an integrity check did. |
| 130 | the run was interrupted. |

## How the numbers are grouped

| Band | What it covers |
| --- | --- |
| RL00xx | usage |
| RL01xx | input contract |
| RL02xx | validity findings |
| RL03xx | outcome |
| RL04xx | execution |
| RL05xx | budget |
| RL06xx | record integrity |
| RL07xx | capability or extra unavailable |
| RL08xx | pending decision |
| RL09xx | internal |
