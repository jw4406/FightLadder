import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_csv_data(csv_path):
    """
    Reads a CSV file with two columns (Global Step and a value) and creates a plot.

    Args:
        csv_path (str): The path to the CSV file.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)

        # Get column names
        if len(df.columns) < 2:
            print(f"Error: CSV file '{csv_path}' must have at least two columns.")
            return

        x_col = df.columns[0]
        y_col = df.columns[1]

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(df[x_col], df[y_col], marker='o', linestyle='-')

        # Add titles and labels
        plt.title(f'{y_col} vs. {x_col}')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.grid(True)
        
        # Save the plot
        base_filename = os.path.splitext(os.path.basename(csv_path))[0]
        output_filename = f"{base_filename}_plot.png"
        plt.savefig(output_filename)

        print(f"Plot saved to {output_filename}")
        plt.close()

    except FileNotFoundError:
        print(f"Error: File not found at '{csv_path}'")
    except Exception as e:
        print(f"An error occurred while processing '{csv_path}': {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot data from CSV files.")
    parser.add_argument('csv_files', nargs='+', help="Paths to the CSV files to plot.")
    
    args = parser.parse_args()

    for csv_file in args.csv_files:
        plot_csv_data(csv_file)
