# AI Multi-Agent Dynamics Research Demo

Research question: To what extent can statistical and nonlinear dynamics methods
distinguish deterministic sensitivity to initial conditions from explicit
stochasticity in AI-controlled multi-agent systems?

This first demo compares three runs of the same simple two-dimensional model.
The agents follow local rules; they are not learned AI models. Each agent has an
ID, position, and velocity. It responds to the positions of agents within a
radius of 3.0. Attraction points toward the average displacement to those
neighbors. Separation pushes away from neighbors closer than 0.5. There is no
message passing, shared memory, or agent memory.

The environment is a 10 by 10 rectangle with reflecting boundaries. The default
run has 20 agents on a fixed grid, 500 updates of size `dt=0.1`, attraction
strength 0.05, and separation strength 0.2. At each update, all accelerations
are calculated from the same current state. Velocity and then position are
updated, followed by boundary reflection. Timestep 0 is the starting state, so
each run records 501 states per agent. The `damping` field exists in the config
but is not applied by the current velocity update.

## Three conditions

1. **Baseline:** fixed initial state and deterministic rules, with no noise.
2. **Initial perturbation:** the same deterministic run, except agent 0 starts
   `1e-5` position units farther along x.
3. **Stochastic:** the baseline initial state and rules, plus a fresh random
   acceleration on each agent's x and y components at every update. Each draw
   is Gaussian with mean zero and standard deviation `noise_strength`.

The default `noise_strength` is 0.0, preserving the deterministic simulation.
The demo uses 0.02 acceleration units, equivalent to position units per model
time unit squared. It runs five trials with seeds 2026 through 2030. The seed
makes each trial reproducible while allowing the trials to differ.

## Run it

From the repository root, use Python with `matplotlib` installed:

```bash
python -m experiments.run_demo
python -m unittest discover -s tests -v
```

The demo also accepts `--steps`, `--trials`, `--noise-strength`,
`--perturbation-size`, and `--first-seed`. For example:

```bash
python -m experiments.run_demo --steps 300 --trials 4 --first-seed 100
```

The default run writes these files:

| File | Contents |
| --- | --- |
| `figures/baseline_trajectories.png` | Paths of all baseline agents; dots mark their starting positions. |
| `figures/perturbed_divergence.png` | Baseline versus the small initial perturbation. |
| `figures/stochastic_divergence.png` | Baseline versus the stochastic trial with seed 2026. |
| `figures/stochastic_trials.png` | Mean stochastic divergence and a band of one sample standard deviation across seeds. |
| `results/perturbed_divergence.csv` | Perturbed divergence at each timestep. |
| `results/stochastic_trials.csv` | Stochastic divergence by timestep, trial, and seed. |
| `results/stochastic_summary.csv` | Mean and sample standard deviation by timestep. |

The older `figures/initial_condition_divergence.png` is preserved. Running the
new demo does not overwrite it.

For each timestep, divergence matches positions by agent ID, finds each pair's
Euclidean distance, squares the distances, averages across agents, and takes
the square root. This is positional root-mean-square error (RMSE), in position
units. A `1e-5` displacement of one of 20 agents therefore starts at about
`1e-5 / sqrt(20) = 2.236e-6` RMSE. Both conditions use this same measure.

## What this does and does not show

The perturbation comparison checks how this deterministic model responds to a
small initial change. The stochastic comparison shows differences caused by
explicit random acceleration. The trial band describes variation among five
chosen seeds; it is not a confidence interval or a significance test.

Trajectory divergence alone does not prove deterministic chaos. Thresholds in
neighbor and boundary rules can also amplify small changes. This demo has not
estimated a Lyapunov exponent or validated a chaos test, and it does not
conclusively distinguish chaos from stochasticity. The autocorrelation, entropy,
and predictability analysis files and the older metrics stubs are unfinished
and are not used here.
