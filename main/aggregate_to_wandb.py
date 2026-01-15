import wandb
import pandas as pd
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--read_from_proj_name", type=str, required=True)
parser.add_argument("--upload_to_proj_name", type=str, required=True)
args = parser.parse_args()
UPLOAD_PROJECT_NAME = args.upload_to_proj_name
READ_FROM_PROJECT_NAME = args.read_from_proj_name
# 1. Setup your entity/project
ENTITY = "jw4406"
#PROJECT = "exploiter"

api = wandb.Api()
runs = api.runs(f"{ENTITY}/{READ_FROM_PROJECT_NAME}")

data = []

# 2. Iterate through runs and grab the single summary metrics
for run in runs:
    # Safely get keys, defaulting to None if missing
    epoch = run.summary.get("main_training_epoch")
    rew = run.summary.get("exploiter_rew")
    
    # Filter: Only keep runs that actually have both metrics
    if epoch is not None and rew is not None:
        data.append([epoch, rew])

# 3. Create a DataFrame and SORT IT (This fixes the "connected line" issue)
df = pd.DataFrame(data, columns=["main_training_epoch", "exploiter_rew"])
df = df.sort_values("main_training_epoch")

# 4. Initialize a specific "Summary" run to hold this plot
with wandb.init(project=UPLOAD_PROJECT_NAME, name="Analysis_Line_Plot", job_type="analysis"):
    
    # Create a native WandB table
    table = wandb.Table(dataframe=df)
    
    # Log the native Line Plot
    # This automatically handles the X/Y mapping and tooltips
    wandb.log({
        "Exploitability_Curve": wandb.plot.line(
            table, 
            "main_training_epoch", 
            "exploiter_rew", 
            title="Exploitability vs Training Epoch"
        )
    })

print("Plot uploaded successfully.")
