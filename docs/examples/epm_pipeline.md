End‑to‑end example computing various measures for different maze arms using folder‑based loaders, batch preprocessing, feature generation, and summary export. Paths are illustrative; adapt to your environment.

```python
# 1) Load a dataset of single‑view DLC CSVs into a TrackingCollection
from py3r.behaviour.tracking.tracking_collection import TrackingCollection
from py3r.behaviour.features.features_collection import FeaturesCollection
from py3r.behaviour.summary.summary_collection import SummaryCollection

DATA_DIR = "/data/recordings"            # e.g. contains EPM_id1.csv, EPM_id2.csv, ...
TAGS_CSV = "/data/tags.csv"              # optional, with columns: handle, treatment, genotype, ...
OUT_DIR  = "/outputs"                    # where to save summary outputs

tc = TrackingCollection.from_dlc_folder(folder_path=DATA_DIR, fps=30)

# 2) (Optional) Add tags from a CSV for grouping/analysis
# CSV must contain a 'handle' column matching filenames (without extension)
# other column names are the tag names, and those column values are the tag values
# e.g. handle, sex, treatment
#      filename1, m, control
#      filename2, f, crs
#      ...etc
try:
    tc.add_tags_from_csv(path_tag_file=TAGS_CSV)
except FileNotFoundError:
    pass

# 3) Batch preprocessing of tracking files
# Remove low-confidence detections (method/thresholds depend on your DLC export)
tc.filter_likelihood(threshold=0.95)

# Smooth all points with mean centre window 3, with exception for environment points
environment_points = ["tl", "tr","ctr","rt","rb","cbr", "br", "bl","cbl","lb","lt","ctl"]
tc.smooth_all(window=3, method="mean", overrides=[(environment_points, "median", 30)])

# Rescale distance to metres according to two corners of the EPM, here named 'tl' and 'br'
tc.rescale_by_known_distance(point1="tl", point2="br", distance_in_metres=0.655)

# Trim ends of recordings if needed
tc.trim(endframe=-10*30)  # drop 10s from end at 30 fps

# 4) Basic QA such as checking length of recordings and ploting tracking trajectories
# Length check (per recording, assuming 5 min, time in seconds)
timecheck = tc.time_as_expected(mintime=300-(0.1*300) ,maxtime=300+(0.1*300))
for key, val in timecheck.items():
    if not val:
        raise Exception(f"file {key} failed timecheck")

# Plot trajectories (per recording, using 'bodycentre' for trajectory of mouse and corners of EPM as static frame)
tc.plot(trajectories=["bodycentre"], static=environment_points, 
        lines=[("tl", "tr"), ("tr", "ctr"), ("ctr", "rt"), ("rt", "rb"),
               ("rb", "cbr"), ("cbr", "br"), ("br", "bl"), ("bl", "cbl"),
               ("cbl", "lb"), ("lb", "lt"), ("lt", "ctl"), ("ctl", "tl")])

# 5) Create FeaturesCollection object
fc = FeaturesCollection.from_tracking_collection(tc)

# 6) Compute features necessary to get different EPM measures 
# Define different boundaries (open arms, closed arms) and check if mouse (defined by 'bodycentre') is inside defined boundary
# Adjust boundaries so they match orientation of your EPM.
# Open arms
for handle in fc.keys():
    _on_open_arm = []
    _oa_boundary = fc[handle].define_boundary(['tl', 'tr', 'ctr', 'ctl'], scaling=1.1, centre = ["ctr", "ctl"])
    _on_open_arm.append(fc[handle].within_boundary_static(point="bodycentre", boundary=_oa_boundary))
    _oa_boundary = fc[handle].define_boundary(['cbl', 'cbr', 'br', 'bl'], scaling=1.1, centre = ["cbr", "cbl"])
    _on_open_arm.append(fc[handle].within_boundary_static(point="bodycentre", boundary=_oa_boundary))
    fc[handle].data["bodycentre_on_open_arms"] = (_on_open_arm[0] + _on_open_arm[1]).astype("boolean")

# Closed arms
for handle in fc.keys():
    _on_closed_arm = []
    _ca_boundary = fc[handle].define_boundary(["ctr", "rt", "rb", "cbr"], scaling=1.1, centre = ["ctr", "cbr"])
    _on_closed_arm.append(fc[handle].within_boundary_static(point="bodycentre", boundary=_ca_boundary))
    _ca_boundary = fc[handle].define_boundary(["lt", "ctl", "cbl", "lb"], scaling=1.1, centre = ["ctl", "cbl"])
    _on_closed_arm.append(fc[handle].within_boundary_static(point="bodycentre", boundary=_ca_boundary))
    fc[handle].data["bodycentre_on_closed_arms"] = (_on_closed_arm[0] + _on_closed_arm[1]).astype("boolean")

# Distance change between one frame and the next for "bodycentre"
fc.distance_change(point="bodycentre").store()

# 7) (Optional) Save features to csv
fc.save(f"{OUT_DIR}/features", data_format="csv", overwrite=True)

# 8) Create SummaryCollection object
sc = SummaryCollection.from_features_collection(fc)

# 9) Compute summary measures per recording
# Total distance moved
sc.total_distance("bodycentre").store()

# Time on open arms
sc.time_true("bodycentre_on_open_arms").store("time_on_open_arms")

# Distance moved on open arms
for handle in sc.keys():
    _temp_feature = sc[handle].features.data["bodycentre_on_open_arms"] * sc[handle].features.data["distance_change_bodycentre"]
    sc[handle].data["distance_moved_on_open_arms"] = _temp_feature.sum()

# Time on closed arms
sc.time_true("bodycentre_on_closed_arms").store("time_on_closed_arms")

# Distance moved on closed arms
for handle in sc.keys():
    _temp_feature = sc[handle].features.data["bodycentre_on_closed_arms"] * sc[handle].features.data["distance_change_bodycentre"]
    sc[handle].data["distance_moved_on_closed_arms"] = _temp_feature.sum()

# 10) Collate scalar outputs into DataFrame and save results in CSV
summary_df = sc.to_df(include_tags=True)
summary_df.to_csv(f"{OUT_DIR}/EPM_results.csv")
```


