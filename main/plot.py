import os
import csv

def consolidate_stats(stats_dir, output_filename, value_header):
    """
    Reads all .txt files from a directory, appends their data to a CSV file,
    and then deletes the .txt files. If the CSV file exists, it's updated
    with new data; otherwise, it's created.
    """
    stats = {}
    
    # 1. Read existing data from CSV if it exists
    if os.path.exists(output_filename):
        try:
            with open(output_filename, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # Skip header
                for row in reader:
                    stats[int(row[0])] = float(row[1])
        except (IOError, StopIteration, ValueError, IndexError) as e:
            print(f"Warning: Could not read existing CSV {output_filename}. It might be empty or corrupted. Error: {e}")

    if not os.path.exists(stats_dir):
        print(f"Directory not found: {stats_dir}")
        return

    processed_files = []
    # 2. Read new data from .txt files
    for filename in os.listdir(stats_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(stats_dir, filename)
            try:
                checkpoint_num = int(os.path.splitext(filename)[0])
                with open(file_path, 'r') as f:
                    value = float(f.read().strip())
                stats[checkpoint_num] = value
                processed_files.append(file_path)
            except (ValueError, IndexError):
                print(f"Could not process file: {filename}")
    
    # 3. Write updated data back to CSV
    if stats:
        sorted_stats = sorted(stats.items())
        try:
            with open(output_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Checkpoint", value_header])
                writer.writerows(sorted_stats)
            
            # 4. Delete processed .txt files only after successful write
            for file_path in processed_files:
                os.remove(file_path)

            print(f"Successfully updated {output_filename} and removed {len(processed_files)} source file(s).")
        except IOError as e:
            print(f"Error writing to {output_filename}. Source files were not deleted. Error: {e}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    wr_stats_dir = os.path.join(current_dir, "trained_models/wr_stats")
    mean_rew_stats_dir = os.path.join(current_dir, "trained_models/mean_rew_stats")
    
    output_wr_file = os.path.join(current_dir, "win_rates.csv")
    output_mean_rew_file = os.path.join(current_dir, "mean_rewards.csv")

    consolidate_stats(wr_stats_dir, output_wr_file, "WinRate")
    consolidate_stats(mean_rew_stats_dir, output_mean_rew_file, "MeanReward")
