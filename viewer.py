import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("F-16 Flight Path Visualization")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("Altitude (m)")

    colors = ['b', 'r', 'g', 'c', 'm', 'y']
    color_index = 0

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

            if not samples:
                continue

            lons, lats, alts = [], [], []

            for s in samples:
                if isinstance(s, dict):
                    lons.append(s['lon'])
                    lats.append(s['lat'])
                    alts.append(s['alt'])
                else:
                    lons.append(getattr(s, 'lon', 0))
                    lats.append(getattr(s, 'lat', 0))
                    alts.append(getattr(s, 'alt', 0))

            if not lons:
                continue

            color = colors[color_index % len(colors)]
            obj_name = getattr(target, 'name', target.get('name', f"F-16 #{i+1}"))
            display_name = f"{filename} - {obj_name}"

            ax.plot(lons, lats, alts, label=display_name, color=color, linewidth=2)
            ax.scatter(lons[0], lats[0], alts[0], color='green', marker='o', s=30)
            ax.scatter(lons[-1], lats[-1], alts[-1], color='red', marker='^', s=50)
            ax.text(lons[-1], lats[-1], alts[-1], f" {display_name}", fontsize=9)

            color_index += 1

    ax.legend()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_acmi.py <file1.acmi> <file2.acmi> ...")
    else:
        render_f16(sys.argv[1:])