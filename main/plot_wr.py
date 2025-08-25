import wandb
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Optional
import argparse


def fetch_wr_data(project_name: str, entity: str, run_id: Optional[str] = None) -> Dict[str, List]:
    """
    Fetch wr (win rate) data from wandb logs.
    
    Args:
        project_name: Name of the wandb project
        entity: The wandb username/team name
        run_id: Specific run ID (optional, will use latest run if None)
    
    Returns:
        Dictionary with metric names as keys and lists of (step, value) tuples as values
    """
    api = wandb.Api()
    
    if run_id:
        run = api.run(f"{entity}/{project_name}/{run_id}")
    else:
        # Get the most recent run
        runs = api.runs(f"{entity}/{project_name}")
        if not runs:
            raise ValueError(f"No runs found in project {entity}/{project_name}")
        run = runs[0]
    
    # Get the history (logged metrics over time)
    history = run.history()

    print(f"Available columns: {list(history.columns)}") #TODO: Remove this - debugging only
    
    # Extract wr-related metrics
    #TODO: It seems this block doesn't work - wr_data remains empty, need to figure out why.
    wr_data = {}
    for col in history.columns:
        if 'br_win_rate' in col:
            # Filter out NaN values and create (step, value) pairs
            valid_data = history[['global_step', col]].dropna()
            wr_data[col] = list(zip(valid_data['global_step'], valid_data[col]))
    
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
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def main():
    """Main function to fetch and plot wandb data."""

    parser = argparse.ArgumentParser(description='Plot win rate data from wandb logs')
    parser.add_argument('--project', type=str, help='wandb project name', default="exploiter")
    parser.add_argument('--entity', type=str, help='wandb username/entity', default="jw4406")
    parser.add_argument('--run_id', type=str, help='specific run ID (uses latest if not provided)')
    parser.add_argument('--save_path', type=str, help='path to save plot', default='wr_plot.png')
    
    args = parser.parse_args()
    
    try:
        print("Fetching data from wandb...")
        wr_data = fetch_wr_data(args.project, args.entity, args.run_id)
        
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