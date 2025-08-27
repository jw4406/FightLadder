import wandb
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Optional
import argparse
import os


def fetch_wr_data(project_name: str, entity: str) -> Dict[str, List]:
    """
    Fetch wr (win rate) data from wandb logs.
    
    Args:
        project_name: Name of the wandb project
        entity: The wandb username/team name
    
    Returns:
        Dictionary with metric names as keys and lists of (step, value) tuples as values
    """
    api = wandb.Api()
    
    # Get ALL runs from the project
    runs = api.runs(f"{entity}/{project_name}")
    if not runs:
        raise ValueError(f"No runs found in project {entity}/{project_name}")

    # Collect data from all runs
    wr_data = {}
    for run in runs:      
        for key, value in run.summary.items():
            if 'br_win_rate' in key:
                # For runs with global_step, use it; otherwise use run creation time or step
                step = run.summary.get('global_step', run.summary.get('_step', 0))
                if key not in wr_data:
                    wr_data[key] = []
                wr_data[key].append((step, value))

    # Sort data points by global_step for proper plotting
    for metric in wr_data:
        wr_data[metric].sort(key=lambda x: x[0])
    
    return wr_data


def plot_wr_data(wr_data: Dict[str, List], save_path: Optional[str] = None) -> None:
    """
    Plot the win rate data over time.
    
    Args:
        wr_data: Dictionary with metric names as keys and (step, value) tuples as values
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(12, 6))
    
    for metric_name, data_points in wr_data.items():
        if data_points:  # Only plot if we have data
            steps, values = zip(*data_points)
            plt.plot(steps, values, label=metric_name, marker='o', markersize=3)
    
    plt.xlabel('Global Step')
    plt.ylabel('Win Rate')
    plt.title('Win Rate vs Training Steps')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def main():
    """Main function to fetch and plot wandb data."""

    parser = argparse.ArgumentParser(description='Plot win rate data from wandb logs')
    parser.add_argument('--project', type=str, help='wandb project name', default="exploiter")
    parser.add_argument('--entity', type=str, help='wandb username/entity', default="jw4406")
    parser.add_argument('--save_path', type=str, help='path to save plot', default='plots/wr_plot.png')
    
    args = parser.parse_args()
    
    try:
        print("Fetching data from wandb...")
        wr_data = fetch_wr_data(args.project, args.entity)
        
        if not wr_data:
            print("No win rate data found in the run")
            return
        
        print(f"Found {len(wr_data)} win rate metrics")
        for metric in wr_data.keys():
            print(f"  - {metric}: {len(wr_data[metric])} data points")
        
        print("Creating plot...")
        plot_wr_data(wr_data, save_path=args.save_path)
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()