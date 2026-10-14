"""Run the baseline, initial perturbation, and stochastic comparisons."""

import argparse
import csv
import math
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.divergence import calculate_divergence, create_initial_state
from experiments.plot_simulation import plot_trajectories
from src.agents import Agent
from src.simulation import SimulationConfig, run_simulation


def save_divergence_plot(divergence, title, path):
    """Save positional RMSE over time for one comparison."""
    timesteps = [timestep for timestep, value in divergence]
    values = [value for timestep, value in divergence]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(timesteps, values)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Positional divergence (RMSE, position units)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_csv(path, fieldnames, rows):
    """Save measured divergence values for later inspection."""
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_demo(number_of_timesteps=500, number_of_trials=5,
             noise_strength=0.02, perturbation_size=1e-5, first_seed=2026):
    """Run three controlled conditions and save their plots and measurements."""
    if number_of_timesteps < 1 or number_of_trials < 2:
        raise ValueError("Use at least one timestep and two stochastic trials.")
    if not math.isfinite(perturbation_size) or perturbation_size <= 0:
        raise ValueError("perturbation_size must be finite and positive")
    if not math.isfinite(noise_strength) or noise_strength <= 0:
        raise ValueError("noise_strength must be finite and positive for this demo")

    figures_dir = Path("figures")
    results_dir = Path("results")
    figures_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    config = SimulationConfig(number_of_timesteps=number_of_timesteps)
    baseline_agents = create_initial_state(config)
    perturbed_agents = [
        Agent(agent.id,
              (agent.position[0] + (perturbation_size if agent.id == 0 else 0.0),
               agent.position[1]),
              agent.velocity)
        for agent in baseline_agents
    ]

    baseline_trajectory = run_simulation(baseline_agents, config)
    perturbed_trajectory = run_simulation(perturbed_agents, config)
    perturbed_divergence = calculate_divergence(baseline_trajectory, perturbed_trajectory)

    fig = plot_trajectories(baseline_trajectory, config)
    fig.savefig(figures_dir / "baseline_trajectories.png", dpi=200)
    plt.close(fig)
    save_divergence_plot(perturbed_divergence,
                         "Baseline vs. Small Initial Perturbation",
                         figures_dir / "perturbed_divergence.png")

    trial_rows = []
    trial_divergences = []
    for trial in range(number_of_trials):
        seed = first_seed + trial
        stochastic_config = replace(config, noise_strength=noise_strength,
                                    random_seed=seed)
        stochastic_trajectory = run_simulation(baseline_agents, stochastic_config)
        stochastic_divergence = calculate_divergence(
            baseline_trajectory, stochastic_trajectory)
        trial_divergences.append(stochastic_divergence)
        for timestep, divergence in stochastic_divergence:
            trial_rows.append({"trial": trial + 1, "seed": seed,
                               "noise_strength": noise_strength,
                               "timestep": timestep, "divergence": divergence})

    save_divergence_plot(trial_divergences[0],
                         f"Baseline vs. Stochastic Run (seed {first_seed})",
                         figures_dir / "stochastic_divergence.png")

    summary_rows = []
    for timestep in range(number_of_timesteps + 1):
        values = [trial_divergence[timestep][1]
                  for trial_divergence in trial_divergences]
        mean_divergence = sum(values) / number_of_trials
        standard_deviation = math.sqrt(
            sum((value - mean_divergence) ** 2 for value in values)
            / (number_of_trials - 1))
        summary_rows.append({"timestep": timestep,
                             "noise_strength": noise_strength,
                             "mean_divergence": mean_divergence,
                             "sample_standard_deviation": standard_deviation})

    timesteps = [row["timestep"] for row in summary_rows]
    means = [row["mean_divergence"] for row in summary_rows]
    deviations = [row["sample_standard_deviation"] for row in summary_rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(timesteps, means, label="Mean divergence")
    ax.fill_between(timesteps,
                    [max(0, mean - deviation) for mean, deviation in zip(means, deviations)],
                    [mean + deviation for mean, deviation in zip(means, deviations)],
                    alpha=0.25, label="±1 sample standard deviation")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Positional divergence (RMSE, position units)")
    ax.set_title(f"Stochastic Divergence Across {number_of_trials} Seeds")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "stochastic_trials.png", dpi=200)
    plt.close(fig)

    write_csv(results_dir / "perturbed_divergence.csv",
              ["timestep", "perturbation_size", "divergence"],
              [{"timestep": timestep, "perturbation_size": perturbation_size,
                "divergence": value}
               for timestep, value in perturbed_divergence])
    write_csv(results_dir / "stochastic_trials.csv",
              ["trial", "seed", "noise_strength", "timestep", "divergence"],
              trial_rows)
    write_csv(results_dir / "stochastic_summary.csv",
              ["timestep", "noise_strength", "mean_divergence",
               "sample_standard_deviation"],
              summary_rows)

    print(f"Baseline: {config.number_of_agents} agents, {number_of_timesteps} steps, "
          f"dt={config.timestep_size}, noise_strength=0")
    print(f"Initial perturbation: agent 0 x + {perturbation_size:g}; "
          f"initial RMSE={perturbed_divergence[0][1]:.8g}, "
          f"final RMSE={perturbed_divergence[-1][1]:.8g}")
    print(f"Stochastic: acceleration noise strength={noise_strength:g}; "
          f"{number_of_trials} trials, seeds {first_seed}–{first_seed + number_of_trials - 1}")
    print(f"Seed {first_seed}: initial RMSE={trial_divergences[0][0][1]:.8g}, "
          f"final RMSE={trial_divergences[0][-1][1]:.8g}")
    print(f"Final stochastic RMSE: mean={means[-1]:.8g}, "
          f"sample SD={deviations[-1]:.8g}")


def main():
    """Read optional settings and run the reproducible demo from the root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--noise-strength", type=float, default=0.02)
    parser.add_argument("--perturbation-size", type=float, default=1e-5)
    parser.add_argument("--first-seed", type=int, default=2026)
    args = parser.parse_args()
    run_demo(args.steps, args.trials, args.noise_strength,
             args.perturbation_size, args.first_seed)


if __name__ == "__main__":
    main()
