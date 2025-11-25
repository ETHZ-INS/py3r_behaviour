End‑to‑end example computing "distance moved" and "time in center" using folder‑based loaders, batch preprocessing, feature generation, and summary export. Paths are illustrative; adapt to your environment.

```python
# 1) Load a dataset of single‑view DLC CSVs into a TrackingCollection
from py3r.behaviour.tracking.tracking_collection import TrackingCollection
from py3r.behaviour.features.features_collection import FeaturesCollection
from py3r.behaviour.summary.summary_collection import SummaryCollection

DATA_DIR = "/data/recordings"            # e.g. contains OFT_id1.csv, OFT_id2.csv, ...
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
    tc.add_tags_from_csv(csv_path=TAGS_CSV)
except FileNotFoundError:
    pass

# 3) Batch preprocessing of tracking files
# Remove low-confidence detections (method/thresholds depend on your DLC export)
tc.filter_likelihood(threshold=0.95)

# Smooth all points with mean centre window 3, with exception for environment points
tc.smooth_all(window=3, method='mean', overrides=[(["tr", "tl", "bl", "br"], "median", 30)])

# Rescale distance to metres according to corners of the OFT, here named 'tl' and 'br'
tc.rescale_by_known_distance(point1='tl', point2='br', distance_in_metres=0.64)

# Trim ends of recordings if needed
tc.trim(endframe=-10*30)  # drop 10s from end at 30 fps

# 4) Basic QA such as checking length of recordings and ploting tracking trajectories
# Length check (per recording, assuming 10 min, time in seconds)
timecheck = tc.time_as_expected(mintime=600-(0.1*600) ,maxtime=600+(0.1*600))
for key, val in timecheck.items():
    if not val:
        raise Exception(f"file {key} failed timecheck")

# Plot trajectories (per recording, using 'bodycentre' for trajectory of mouse and corners of OFT as static frame)
tc.plot(trajectories=["bodycentre"], static=["tr", "tl", "bl", "br"], 
        lines=[("tr","tl"), ("tl","bl"), ("bl","br"), ("br","tr")])

# 5) Create FeaturesCollection object
fc = FeaturesCollection.from_tracking_collection(tc)

# 6) Compute features necessary to compute time in center 
# Define boundary of center area and check if mouse (defined by 'bodycentre') is inside defined boundary
center_boundary = fc.define_boundary(["tl", "tr", "bl", "br"], scaling=0.5)
fc.within_boundary_static(point="bodycentre", boundary=center_boundary, boundary_name="center").store()

# 7) (Optional) Save features to csv
fc.save(f"{OUT_DIR}/features", data_format="csv", overwrite=True)

# 8) Create SummaryCollection object
sc = SummaryCollection.from_features_collection(fc)

# 9) Compute summary measures per recording
# Total distance moved
sc.total_distance("bodycentre").store()

# Time in center
sc.time_true("within_boundary_static_bodycentre_in_center").store("time_in_center")

# 10) Collate scalar outputs into DataFrame and save results in CSV
summary_df = sc.to_df(include_tags=True)
summary_df.to_csv(f"{OUT_DIR}/OFT_results.csv")
```


[oft_pipeline.md](oft_pipeline.md)