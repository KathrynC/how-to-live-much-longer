"""Tests for the CA visualization module."""
import os
import pytest

from ca_simulator import run_single_cell, run_tissue_grid
from ca_visualize import (
    plot_ca_trajectory, plot_rule_timeline, plot_ca_fidelity,
    plot_tissue_grid, plot_cliff_approach, generate_all_plots,
)


class TestVisualization:
    def test_plot_ca_trajectory(self, tmp_path):
        result = run_single_cell()
        path = str(tmp_path / "ca_traj.png")
        plot_ca_trajectory(result, output_path=path)
        assert os.path.exists(path)

    def test_plot_rule_timeline(self, tmp_path):
        result = run_single_cell()
        path = str(tmp_path / "rule_timeline.png")
        plot_rule_timeline(result, output_path=path)
        assert os.path.exists(path)

    def test_plot_tissue_grid(self, tmp_path):
        result = run_tissue_grid()
        path = str(tmp_path / "tissue_grid.png")
        plot_tissue_grid(result, output_path=path)
        assert os.path.exists(path)

    def test_plot_cliff_approach(self, tmp_path):
        result = run_single_cell()
        path = str(tmp_path / "cliff_approach.png")
        plot_cliff_approach(result, output_path=path)
        assert os.path.exists(path)

    def test_generate_all_plots(self, tmp_path):
        generate_all_plots(output_dir=str(tmp_path))
        files = os.listdir(str(tmp_path))
        assert len(files) >= 3

    def test_fidelity_without_ode(self, tmp_path):
        result = run_single_cell()
        path = str(tmp_path / "fidelity.png")
        plot_ca_fidelity(result, None, output_path=path)
        # Should not create file when no ODE result
        assert not os.path.exists(path)

    def test_custom_title(self, tmp_path):
        result = run_single_cell()
        path = str(tmp_path / "custom.png")
        plot_ca_trajectory(result, title="Custom Title", output_path=path)
        assert os.path.exists(path)
