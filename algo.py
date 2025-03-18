import json
import csv
import numpy as np
from scipy.spatial import distance
from pykalman import KalmanFilter
from datetime import datetime
import time
import json
import csv
import re

def read_json_file(json_file):
    """Reads JSON file and returns a list of valid JSON objects."""
    valid_data = []
    with open(json_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                line = re.sub(r",\s*$", "", line)  # Trailing comma fix
                json_obj = json.loads(line)  # Parse JSON
                valid_data.append(json_obj)
            except json.JSONDecodeError:
                print("❌ Skipping Invalid Line:", line[:100])  
    return valid_data

def read_csv_to_dict(csv_file):
    """Reads CSV file and returns a dictionary indexed by timestamp."""
    imu_data = {}
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                timestamp = row.get("timestamp", "").strip()
                row['heading'] = float(row.get('heading', 0))  # Convert heading to float
                row['status'] = row.get('status', '').strip()  # Handle missing status
                imu_data[timestamp] = row
        return imu_data
    except FileNotFoundError:
        print(f"❌ Error: CSV file {csv_file} not found.")
        return {}
    except Exception as e:
        print(f"❌ CSV Read Error: {e}")
        return {}

# File Paths
json_file_path = "/home/tar-tt060-saurav/Downloads/task_cam_data.json"
csv_file_path = "/home/tar-tt060-saurav/Downloads/task_imu(in).csv"

# Load data into separate variables
camera_sensor_data = read_json_file(json_file_path)
imu_sensor_data = read_csv_to_dict(csv_file_path)






# **1. Parse Camera Data**
def parse_camera_data(camera_sensor_data):
    parsed_data = []
    for sensor_entry in camera_sensor_data:
        for cam_id, data in sensor_entry.items():
            timestamp = data['timestamp'][:19]  # Trim to seconds (HH:MM:SS)
            positions = data['object_positions_x_y']
            for pos in positions:
                parsed_data.append({
                    'timestamp': timestamp,
                    'x': pos[0],
                    'y': pos[1],
                    'sensor_id': cam_id
                })
    return parsed_data

# **2. Parse IMU Data**
def parse_imu_data(imu_sensor_data):
    parsed_imu = []
    for key, entry in imu_sensor_data.items():
        timestamp = entry['timestamp'][:19]  # Trim to seconds (HH:MM:SS)
        parsed_imu.append({
            'timestamp': timestamp,
            'heading': float(entry['heading']),
            'state': entry['state']
        })
    return parsed_imu

# **3. Apply Kalman Filter on IMU Heading**
def apply_kalman_filter(imu_data):
    headings = np.array([entry['heading'] for entry in imu_data])
    kf = KalmanFilter(initial_state_mean=headings[0], n_dim_obs=1)
    filtered_headings, _ = kf.filter(headings)
    
    for i, entry in enumerate(imu_data):
        entry['filtered_heading'] = filtered_headings[i]
    
    return imu_data

# **4. Clustering Objects Based on Distance**
def cluster_objects(camera_batch, threshold=2):
    clusters = []
    assigned = set()

    for i, obj1 in enumerate(camera_batch):
        if i in assigned:
            continue
        cluster = [[obj1['x'], obj1['y'], obj1['sensor_id']]]
        assigned.add(i)

        for j, obj2 in enumerate(camera_batch):
            if j in assigned:
                continue
            dist = distance.euclidean((obj1['x'], obj1['y']), (obj2['x'], obj2['y']))
            if dist <= threshold:
                cluster.append([obj2['x'], obj2['y'], obj2['sensor_id']])
                assigned.add(j)

        clusters.append(cluster)
    return clusters

# **5. Assign Unique f_id to Clusters**
fused_object_ids = {}
next_f_id = 1

def assign_f_id(cluster):
    global next_f_id
    key = tuple(map(tuple, cluster))  # Convert cluster to hashable key
    if key not in fused_object_ids:
        fused_object_ids[key] = next_f_id
        next_f_id += 1
    return fused_object_ids[key]


from datetime import datetime

def normalize_timestamp(timestamp):
    """ Convert timestamp to a common format for matching. """
    return timestamp.replace("T", " ")  # Remove 'T' to match formats
import os
# **6. Real-Time Simulation (Processing Frame-by-Frame)**
def simulate_real_time_processing(camera_data, imu_data):
    fused_data = []
    cam_batch = []
    imu_batch = []
    i, j = 0, 0
    prev_timestamp = None
    count = 0
    filename="fused_data_final.csv"
        # **Open CSV file only once before the loop**
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)

        # **Write header if the file is newly created**
        if not file_exists:
            writer.writerow(["f_timestamp", "f_id", "cluster_data", "heading", "state"])

        while i < len(camera_data) and j < len(imu_data):
            cam_timestamp = camera_data[i]['timestamp']
            imu_timestamp = imu_data[j]['timestamp']


            # **Set initial timestamp**
            if prev_timestamp is None:
                prev_timestamp = cam_timestamp

            # **Accumulate Camera Data with the same timestamp**
            while i < len(camera_data) and normalize_timestamp(camera_data[i]['timestamp']) == prev_timestamp:
                cam_batch.append(camera_data[i])
                i += 1

            # **Accumulate IMU Data with the same timestamp**
            
            while j < len(imu_data) and normalize_timestamp(imu_data[j]['timestamp']) == prev_timestamp:

                imu_batch.append(imu_data[j])
                j += 1
            # print(normalize_timestamp(imu_data[j-1]['timestamp']), "-----", normalize_timestamp(camera_data[i-1]['timestamp']))
            # **If timestamp changes, process the batch**
            # exit()
            if cam_timestamp != prev_timestamp:
                fused_batch = process_batch(cam_batch, imu_batch,writer)
                count+=1
                # print("Fused batch ",fused_batch)
                print("len ************************************************",len(cam_batch),len(fused_batch))
                # time.sleep(1)
                fused_data.extend(fused_batch)
                cam_batch.clear()
                imu_batch.clear()
                prev_timestamp = cam_timestamp
    print(count)
    return fused_data

# **7. Process Each Batch of Data**
def process_batch(cam_batch, imu_batch,file_writer):
    if not cam_batch or not imu_batch:
        return []

    clusters = cluster_objects(cam_batch)
    fused_batch = []

    for cluster in clusters:
        f_id = assign_f_id(cluster)

        # Match nearest IMU data
        imu_entry = imu_batch[0]  # Using the first IMU entry for this timestamp
        fused_data = [
            cam_batch[0]['timestamp'], f_id, cluster, imu_entry['filtered_heading'], imu_entry['state']
        ]
        fused_batch.append(fused_data)

        file_writer.writerow(fused_data)
        # print(f"Processed: {cam_batch[0]['timestamp']}, f_id: {f_id}, Cluster: {cluster}, Heading: {imu_entry['filtered_heading']}, State: {imu_entry['state']}")

    return fused_batch

# **8. Write Fused Data to CSV**
def write_to_csv(fused_data, filename="fused_datas.csv"):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["f_timestamp", "f_id", "cluster_data", "heading", "state"])
        writer.writerows(fused_data)

# **9. Main Function**
def main(camera_sensor_data, imu_sensor_data):
    # Parse data
    camera_data = parse_camera_data(camera_sensor_data)
    imu_data = parse_imu_data(imu_sensor_data)

    # Sort IMU Data
    imu_data = sorted(imu_data, key=lambda x: x['timestamp'])

    # Apply Kalman Filter
    imu_data = apply_kalman_filter(imu_data)

    # Simulate Real-Time Processing
    fused_data = simulate_real_time_processing(camera_data, imu_data)

    # Save results
    write_to_csv(fused_data)

# **Run the pipeline**
main(camera_sensor_data, imu_sensor_data)




