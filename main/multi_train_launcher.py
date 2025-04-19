import subprocess

# List of player characters
CHARACTERS = ["Vega", "Balrog", "Guile", "EHonda", "Blanka"]#, "Sagat", "MBison", "Dhalsim", "Zangief", "ChunLi", "Ken"]

MAX_PARALLEL = 12  # Adjust to fit your cluster's CPU/GPU availability
PROCESSES = []

for i, char in enumerate(CHARACTERS):
    print(f"Launching training for: {char}")
    cmd = ["python", "ippo.py", "--player", char]
    p = subprocess.Popen(cmd)
    PROCESSES.append(p)

    if len(PROCESSES) >= MAX_PARALLEL:
        for p in PROCESSES:
            p.wait()
        PROCESSES = []

# Final cleanup
for p in PROCESSES:
    p.wait()

