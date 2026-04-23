# Script to evaluate a trained agent on the environment and log data to .acmi files for visualization in the Daedalus Visualizer
import os

from src.envs.base_env import env_from_config
from src.configs.config import Config
from src.utils.math import enu_to_geodetic


def export_batch_acmi(filename, step_idx, dt, npos, epos, alt, roll, pitch, yaw):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # 'w' overwrites the file on the first step, 'a' appends after that
    mode = 'w' if step_idx == 0 else 'a'
    current_time = step_idx * dt

    with open(filename, mode) as f:
        # Write the Tacview Header on the first frame
        if step_idx == 0:
            f.write("FileType=text/acmi/tacview\n")
            f.write("FileVersion=2.1\n")

        f.write(f"#{current_time:.2f}\n")

        num_agents = npos.shape[0]
        for i in range(num_agents):
            # Extract scalars (.item()) to prevent the math.sqrt crash!
            n = npos[i].item()
            e = epos[i].item()
            a = alt[i].item()

            # Convert ENU to Lat/Lon safely (scalar by scalar)
            lat, lon, h = enu_to_geodetic(e, n, a, 0, 0, 0)

            # Convert radians to degrees for Tacview
            r = math.degrees(roll[i].item())
            p = math.degrees(pitch[i].item())
            y_ang = math.degrees(yaw[i].item())

            # Assign a unique Hex ID to each aircraft
            obj_id = f"A{100 + i}"

            # Write the telemetry line
            if step_idx == 0:
                f.write(f"{obj_id},T={lon}|{lat}|{h}|{r}|{p}|{y_ang},Name=F-16 #{i+1},Type=Air+FixedWing\n")
            else:
                f.write(f"{obj_id},T={lon}|{lat}|{h}|{r}|{p}|{y_ang}\n")