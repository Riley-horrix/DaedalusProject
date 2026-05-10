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
    x = R_EARTH * dlon * math.cos(math.radians(lat0))
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
                        self.objects[obj_id] = {'id': obj_id, 'name': 'Unknown', 'type': 'Unknown', 'samples': []}

                    lon = lat = alt = roll = pitch = yaw = None

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

                                    # If length > 5, this is a Control Task target (or a plane)
                                    if len(coords) > 5:
                                        roll = float(coords[3])
                                        pitch = float(coords[4])
                                        yaw = float(coords[5])
                                except (ValueError, IndexError):
                                    pass

                    if lon is not None and lat is not None:
                        self.objects[obj_id]['samples'].append({
                            't': current_time, 'lon': lon, 'lat': lat, 'alt': alt,
                            'roll': roll, 'pitch': pitch, 'yaw': yaw
                        })

def render_f16(filepaths):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Agent Flight Paths vs Targets")
    ax.set_xlabel("X - East/West (meters)")
    ax.set_ylabel("Y - North/South (meters)")
    ax.set_zlabel("Z - Altitude (meters)")

    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'orange', 'purple', 'cyan']
    color_index = 0

    all_x, all_y, all_z = [], [], []
    origin_lon = None
    origin_lat = None

    for filepath in filepaths:
        filename = os.path.basename(filepath)
        print(f"Loading: {filepath}")

        acmi = MiniACMI(filepath)

        f16_objs = []
        target_objs = []
        objects_iter = acmi.objects.values() if isinstance(acmi.objects, dict) else acmi.objects

        for obj in objects_iter:
            name = getattr(obj, 'name', obj.get('name', ''))
            obj_type = getattr(obj, 'type', obj.get('type', ''))
            search_str = (str(name) + " " + str(obj_type)).upper()

            if "F-16" in search_str or "F16" in search_str or "VIPER" in search_str:
                f16_objs.append(obj)
            elif "TARGET" in search_str or "WAYPOINT" in search_str:
                target_objs.append(obj)

        if not f16_objs:
            print(f"No F-16 objects found in {filename}.")
            continue

        # -------------------------------------------------------------
        # 1. Plot F-16 Flight Paths
        # -------------------------------------------------------------
        for i, plane in enumerate(f16_objs):
            samples = getattr(plane, 'samples', plane.get('samples', []))
            if not samples: continue

            x_coords, y_coords, z_coords = [], [], []
            for s in samples:
                if origin_lon is None:
                    origin_lon, origin_lat = s['lon'], s['lat']

                x, y = lonlat_to_meters(s['lon'], s['lat'], origin_lon, origin_lat)
                x_coords.append(x)
                y_coords.append(y)
                z_coords.append(s['alt'])

            if not x_coords: continue
            all_x.extend(x_coords)
            all_y.extend(y_coords)
            all_z.extend(z_coords)

            color = colors[color_index % len(colors)]
            obj_name = getattr(plane, 'name', plane.get('name', f"F-16 #{i+1}"))

            ax.plot(x_coords, y_coords, z_coords, label=obj_name, color=color, linewidth=2, alpha=0.8)
            ax.scatter(x_coords[0], y_coords[0], z_coords[0], color='green', marker='o', s=30)
            ax.scatter(x_coords[-1], y_coords[-1], z_coords[-1], color='red', marker='^', s=50)
            color_index += 1

        color_index = 0

        # -------------------------------------------------------------
        # 2. Plot Targets (Dynamic Logic for Tracking vs Control)
        # -------------------------------------------------------------
        for i, target in enumerate(target_objs):
            samples = getattr(target, 'samples', target.get('samples', []))
            if not samples: continue

            # We use the final sample for both, since tracking is static anyway
            s = samples[-1]

            if origin_lon is None:
                origin_lon, origin_lat = s['lon'], s['lat']

            x, y = lonlat_to_meters(s['lon'], s['lat'], origin_lon, origin_lat)
            color = colors[color_index % len(colors)]
            color_index += 1

            all_x.append(x)
            all_y.append(y)
            all_z.append(s['alt'])

            obj_name = getattr(target, 'name', target.get('name', f"Target #{i+1}"))

            # Check if this target contains pitch/yaw vectors (Control Task)
            is_control_task = s['pitch'] is not None and s['yaw'] is not None

            if is_control_task:
                # Convert degrees to radians for vector math
                p_rad = math.radians(s['pitch'])
                y_rad = math.radians(s['yaw'])

                vec_length = 200.0
                dx = vec_length * math.cos(p_rad) * math.sin(y_rad)
                dy = vec_length * math.cos(p_rad) * math.cos(y_rad)
                dz = vec_length * math.sin(p_rad)

                all_x.extend([x + dx])
                all_y.extend([y + dy])
                all_z.extend([s['alt'] + dz])

                # Draw the directional arrow
                ax.quiver(x, y, s['alt'], dx, dy, dz, color=color, label=obj_name, arrow_length_ratio=0.15, linewidth=2.5)
                # Add a star at the origin of the vector
                ax.scatter(x, y, s['alt'], color=color, marker='*', s=100)
            else:
                # Tracking Task (Position Only)
                ax.scatter(x, y, s['alt'], label=obj_name, color=color, marker='*', s=150)

    # -------------------------------------------------------------
    # Calculate 1:1:1 Scale boundaries
    # -------------------------------------------------------------
    if all_x and all_y and all_z:
        mid_x = (min(all_x) + max(all_x)) / 2.0
        mid_y = (min(all_y) + max(all_y)) / 2.0
        mid_z = (min(all_z) + max(all_z)) / 2.0
        max_range = max(max(all_x) - min(all_x), max(all_y) - min(all_y), max(all_z) - min(all_z)) / 2.0 * 1.05

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        try:
            ax.set_box_aspect((1, 1, 1))
        except AttributeError:
            pass

    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_acmi.py <file1.acmi> <file2.acmi> ...")
    else:
        render_f16(sys.argv[1:])