import wandb
import pandas as pd

# 1. Setup your entity/project
ENTITY = "jw4406"
PROJECT = "exploiter"

api = wandb.Api()
runs = api.runs(f"{ENTITY}/{PROJECT}")

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
with wandb.init(project=PROJECT, name="Analysis_Line_Plot", job_type="analysis"):
    
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