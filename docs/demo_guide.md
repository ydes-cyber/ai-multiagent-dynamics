# Demo Explaination guide

Start with the question: can we tell a small deterministic change in starting
conditions apart from random disturbances in a multi-agent simulation? This
demo is a first comparison, not a final chaos test.

There are 20 agents moving in a reflecting rectangle. They use nearby agent
positions for attraction and separation. They do not send messages or remember
past states. I run one fixed baseline, one run with agent 0 moved `1e-5` along
x, and five runs that all start at the baseline state but have random
acceleration. The random runs use noise strength 0.02 and seeds 2026–2030.

Show the baseline paths first, then the two separate divergence graphs. At each
time, divergence is the root-mean-square position difference between agents
with the same ID. It is zero for identical positions. Since only one of 20
agents is shifted initially, the perturbation begins near `2.236e-6` RMSE,
smaller than `1e-5`. The stochastic runs begin at zero because their starting
states are exactly the same as the baseline's. Noise enters during updates.

Finally, show the trial graph. Its line is the mean divergence of five seeds;
the shaded region is one sample standard deviation above and below the mean.
Different seeds show how much this particular noise mechanism changes the
result. This band is descriptive, not a confidence interval.

In the default run, the perturbation's final RMSE was about 2.738. Seed 2026's
stochastic final RMSE was about 3.274. Across five stochastic trials, the final
mean was about 2.626 and the sample standard deviation was about 0.421.
These values come from the default generated CSV files and may change if the
settings change.

## What each figure shows

| Figure | Explanation |
| --- | --- |
| `baseline_trajectories.png` | Where the deterministic agents move; dots are starts. |
| `perturbed_divergence.png` | Position difference caused by one tiny starting shift under deterministic rules. |
| `stochastic_divergence.png` | Position difference caused by random acceleration in seed 2026. |
| `stochastic_trials.png` | Mean difference across five seeds, with a sample standard deviation band. |

## Three limitations to acknowledge

1. Growth in trajectory difference does not by itself establish chaos. Neighbor
   thresholds and wall reflections can amplify small changes too.
2. The agents follow hand-written local rules, not trained AI policies. There is
   no communication or memory in this model.
3. Five seeds and one noise setting give only a small descriptive comparison.
   There is no significance test or validated nonlinear dynamics measure yet.

## Likely questions

**How is the random condition controlled?** It starts from the exact baseline
state and uses the same rules, environment, timestep, and duration. Only the
added acceleration draws vary. A fixed seed reproduces a trial.

**What does `noise_strength=0.02` mean?** It is the standard deviation of each
independent x or y Gaussian acceleration draw, in position units per model time
unit squared. Zero switches this disturbance off.

**Why is the initial perturbation RMSE below `1e-5`?** RMSE averages squared
distance over all 20 agents. Only one agent moved, so it is `1e-5 / sqrt(20)`.

**Does the larger final difference prove chaos?** No. It shows sensitivity in
this comparison. A chaos claim needs further analysis and validation.

**Is damping part of the model?** A damping setting exists in the config, but
the current velocity update does not apply it.
