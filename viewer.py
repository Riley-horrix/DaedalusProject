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
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Agent flight paths and target waypoints")
    ax.set_xlabel("X - East/West (meters)")
    ax.set_ylabel("Y - North/South (meters)")
    ax.set_zlabel("Z - Altitude (meters)")

    colors = ['b', 'r', 'g', 'c', 'm', 'y']
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

        # 1. Sort the objects into Planes and Targets
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

        # 2. Plot F-16s (As Lines)
        for i, plane in enumerate(f16_objs):
            samples = getattr(plane, 'samples', plane.get('samples', []))
            if not samples: continue

            x_coords, y_coords, z_coords = [], [], []
            for s in samples:
                raw_lon = s['lon'] if isinstance(s, dict) else getattr(s, 'lon', 0)
                raw_lat = s['lat'] if isinstance(s, dict) else getattr(s, 'lat', 0)
                raw_alt = s['alt'] if isinstance(s, dict) else getattr(s, 'alt', 0)

                if origin_lon is None:
                    origin_lon, origin_lat = raw_lon, raw_lat

                x, y = lonlat_to_meters(raw_lon, raw_lat, origin_lon, origin_lat)
                x_coords.append(x)
                y_coords.append(y)
                z_coords.append(raw_alt)

            if not x_coords: continue
            all_x.extend(x_coords)
            all_y.extend(y_coords)
            all_z.extend(z_coords)

            color = colors[color_index % len(colors)]
            obj_name = getattr(plane, 'name', plane.get('name', f"F-16 #{i+1}"))

            ax.plot(x_coords, y_coords, z_coords, label=obj_name, color=color, linewidth=2)
            ax.scatter(x_coords[0], y_coords[0], z_coords[0], color='green', marker='o', s=30) # Start
            ax.scatter(x_coords[-1], y_coords[-1], z_coords[-1], color='red', marker='^', s=50) # End
            color_index += 1

        # 3. Plot Targets (As Static Stars)
        for i, target in enumerate(target_objs):
            samples = getattr(target, 'samples', target.get('samples', []))
            if not samples: continue

            # Just grab the first sample since the target is static
            s = samples[0]
            raw_lon = s['lon'] if isinstance(s, dict) else getattr(s, 'lon', 0)
            raw_lat = s['lat'] if isinstance(s, dict) else getattr(s, 'lat', 0)
            raw_alt = s['alt'] if isinstance(s, dict) else getattr(s, 'alt', 0)

            if origin_lon is None:
                origin_lon, origin_lat = raw_lon, raw_lat

            x, y = lonlat_to_meters(raw_lon, raw_lat, origin_lon, origin_lat)

            all_x.append(x)
            all_y.append(y)
            all_z.append(raw_alt)

            obj_name = getattr(target, 'name', target.get('name', f"Target #{i+1}"))
            ax.scatter(x, y, raw_alt, label=obj_name, color='blue', marker='*', s=150)

    if all_x and all_y and all_z:
        mid_x = (min(all_x) + max(all_x)) / 2.0
        mid_y = (min(all_y) + max(all_y)) / 2.0
        mid_z = (min(all_z) + max(all_z)) / 2.0

        max_range = max(
            max(all_x) - min(all_x),
            max(all_y) - min(all_y),
            max(all_z) - min(all_z)
        ) / 2.0
        max_range *= 1.05

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        try:
            ax.set_box_aspect((1, 1, 1))
        except AttributeError:
            pass

    # Move legend outside the plot so it doesn't block the flight path
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_acmi.py <file1.acmi> <file2.acmi> ...")
    else:
        render_f16(sys.argv[1:])