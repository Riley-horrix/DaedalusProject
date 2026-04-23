import sys
import os
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Approximate radius of the Earth in meters
R_EARTH = 6378137.0

def lonlat_to_meters(lon, lat, lon0, lat0):
    """Converts GPS coordinates to X/Y meters relative to an origin point."""
    dlat = math.radians(lat - lat0)
    dlon = math.radians(lon - lon0)

    # X is East/West. We multiply by cos(lat) to account for meridian convergence.
    x = R_EARTH * dlon * math.cos(math.radians(lat0))
    # Y is North/South
    y = R_EARTH * dlat

    return x, y

class MiniACMI:
    def __init__(self, filepath):
        self.objects = {}
        self.parse(filepath)

    def parse(self, filepath):
        current_time = 0.0
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                if not line: continue

                if line.startswith('#'):
                    try:
                        current_time = float(line[1:])
                    except ValueError:
                        pass
                    continue

                if ',' in line:
                    parts = line.split(',')
                    obj_id = parts[0]

                    if obj_id not in self.objects:
                        self.objects[obj_id] = {
                            'id': obj_id,
                            'name': 'Unknown',
                            'type': 'Unknown',
                            'samples': []
                        }

                    lon = lat = alt = None

                    for prop in parts[1:]:
                        if '=' in prop:
                            key, val = prop.split('=', 1)

                            if key == 'Name':
                                self.objects[obj_id]['name'] = val
                            elif key == 'Type':
                                self.objects[obj_id]['type'] = val
                            elif key == 'T':
                                coords = val.split('|')
                                try:
                                    lon = float(coords[0])
                                    lat = float(coords[1])
                                    alt = float(coords[2]) if len(coords) > 2 else 0.0
                                except (ValueError, IndexError):
                                    pass

                    if lon is not None and lat is not None:
                        self.objects[obj_id]['samples'].append({
                            't': current_time,
                            'lon': lon,
                            'lat': lat,
                            'alt': alt
                        })

def render_f16(filepaths):
    fig = plt.figure(figsize=(10, 10)) # Made square for better 1:1 viewing
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("F-16 Flight Path (True 1:1:1 Scale)")
    ax.set_xlabel("X - East/West (meters)")
    ax.set_ylabel("Y - North/South (meters)")
    ax.set_zlabel("Z - Altitude (meters)")

    colors = ['b', 'r', 'g', 'c', 'm', 'y']
    color_index = 0

    # Track all converted coordinates for global bounding box
    all_x = []
    all_y = []
    all_z = []

    origin_lon = None
    origin_lat = None

    for filepath in filepaths:
        filename = os.path.basename(filepath)
        print(f"Loading: {filepath}")

        acmi = MiniACMI(filepath)

        targets = []
        objects_iter = acmi.objects.values() if isinstance(acmi.objects, dict) else acmi.objects

        for obj in objects_iter:
            name = getattr(obj, 'name', obj.get('name', ''))
            obj_type = getattr(obj, 'type', obj.get('type', ''))

            search_str = (str(name) + " " + str(obj_type)).upper()

            if "F-16" in search_str or "F16" in search_str or "VIPER" in search_str:
                targets.append(obj)

        if not targets:
            print(f"No F-16 objects found in {filename}.")
            continue

        for i, target in enumerate(targets):
            samples = getattr(target, 'samples', target.get('samples', []))
            if not samples: continue

            x_coords, y_coords, z_coords = [], [], []

            for s in samples:
                # Extract raw values safely
                raw_lon = s['lon'] if isinstance(s, dict) else getattr(s, 'lon', 0)
                raw_lat = s['lat'] if isinstance(s, dict) else getattr(s, 'lat', 0)
                raw_alt = s['alt'] if isinstance(s, dict) else getattr(s, 'alt', 0)

                # Set the origin to the very first coordinate we process
                if origin_lon is None:
                    origin_lon = raw_lon
                    origin_lat = raw_lat

                # Convert to meters
                x, y = lonlat_to_meters(raw_lon, raw_lat, origin_lon, origin_lat)

                x_coords.append(x)
                y_coords.append(y)
                z_coords.append(raw_alt)

            if not x_coords: continue

            all_x.extend(x_coords)
            all_y.extend(y_coords)
            all_z.extend(z_coords)

            color = colors[color_index % len(colors)]
            obj_name = getattr(target, 'name', target.get('name', f"F-16 #{i+1}"))
            display_name = f"{filename} - {obj_name}"

            ax.plot(x_coords, y_coords, z_coords, label=display_name, color=color, linewidth=2)
            ax.scatter(x_coords[0], y_coords[0], z_coords[0], color='green', marker='o', s=30)
            ax.scatter(x_coords[-1], y_coords[-1], z_coords[-1], color='red', marker='^', s=50)
            ax.text(x_coords[-1], y_coords[-1], z_coords[-1], f" {display_name}", fontsize=9)

            color_index += 1

    # --- TRUE 1:1:1 EQUAL AXIS LOGIC ---
    if all_x and all_y and all_z:
        # 1. Find the center point of the 3D data block
        mid_x = (min(all_x) + max(all_x)) / 2.0
        mid_y = (min(all_y) + max(all_y)) / 2.0
        mid_z = (min(all_z) + max(all_z)) / 2.0

        # 2. Find the single largest dimension spread (X, Y, or Z)
        max_range = max(
            max(all_x) - min(all_x),
            max(all_y) - min(all_y),
            max(all_z) - min(all_z)
        ) / 2.0

        max_range *= 1.05 # Add 5% padding so nothing touches the edge

        # 3. Apply the exact same range from the center of all three axes
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # 4. Tell Matplotlib to draw the 3D box as a perfect cube
        try:
            ax.set_box_aspect((1, 1, 1))
        except AttributeError:
            # Silently pass if user is on an ancient version of Matplotlib
            pass

    ax.legend()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_acmi.py <file1.acmi> <file2.acmi> ...")
    else:
        render_f16(sys.argv[1:])