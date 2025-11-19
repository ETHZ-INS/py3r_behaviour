End‑to‑end example using folder‑based loaders, batch preprocessing, feature generation, and summary export. Paths are illustrative; adapt to your environment.

```python
# 1) Load a dataset of single‑view DLC CSVs into a TrackingCollection
from py3r.behaviour.tracking.tracking_collection import TrackingCollection
from py3r.behaviour.features.features_collection import FeaturesCollection
from py3r.behaviour.summary.summary_collection import SummaryCollection

DATA_DIR = "/data/recordings"            # e.g. contains S1.csv, S2.csv, ...
TAGS_CSV = "/data/tags.csv"              # optional, with columns: handle, treatment, genotype, ...
OUT_DIR  = "/outputs"                    # where to save artefacts

tc = TrackingCollection.from_dlc_folder(DATA_DIR, fps=30)

# 2) (Optional) Add tags from a CSV for grouping/analysis
# CSV must contain a 'handle' column matching filenames (without extension)
# other column names are the tag names, and those column values are the tag values
# e.g. handle, sex, treatment
#      filename1, m, control
#      filename2, f, crs
#      ...etc
try:
    tc.add_tags_from_csv(TAGS_CSV, handle_col="handle")
except FileNotFoundError:
    pass

# 3) Basic QA (examples)
# - Length check (per recording, assuming 10 min, time in seconds)
timecheck = tc.time_as_expected(mintime=570, maxtime=630)
for key, val in timecheck.items():
    if not val:
        raise Exception(f"file {key} failed timecheck")


# 4) Batch preprocessing (adapt as needed)

# Remove low-confidence detections (method/thresholds depend on your DLC export)
tc.filter_likelihood(threshold=0.95, inplace=True)

# rescale distance to metres according to corners of the OFT, here named 'tl' and 'br'
tc.rescale_by_known_distance(point1='tl', point2='br', distance_in_metres=0.67, inplace=True)

# Smooth all points with mean centre window 3, with exception for environment points
tc.smooth_all(3, 'mean', (['tl','tr','bl','br'],30,'median'), inplace=True)

# Interpolate missing values
tc.interpolate(threshold=3, method='linear', inplace=True)

#Trim ends if needed
tc.trim(start=15*30, end=-15*30, inplace=True)  # drop 15s from start/end at 30 fps

# 5) Build FeaturesCollection
fc = FeaturesCollection.from_tracking_collection(tc)

# 6) Compute and store common features (distance/speed/etc.)
# Distance between two tracked points (e.g., 'p1' and 'p2')
d12 = {h: f.distance_between('p1','p2') for h, f in fc.items()}
fc.store(d12, name="d12")

# Per‑frame speed of a point
spd = {h: f.speed('p1') for h, f in fc.items()}
fc.store(spd, name="speed_p1")

# Boundary example (median static boundary from three points) and membership
inside = {}
for h, f in fc.items():
    boundary = f.define_boundary(['p1', 'p2', 'p3'], scaling=1.0)
    inside[h] = f.within_boundary_static('p1', boundary)
fc.store(inside, name="inside_region")

# 7) Save features (round‑trip format)
fc.save(f"{OUT_DIR}/features", data_format="csv", overwrite=True)

# 8) Summaries
sc = SummaryCollection.from_features_collection(fc)

# Per‑recording: total distance and time inside region
totdist = {h: s.total_distance('p1') for h, s in sc.items()}
time_in = {h: s.time_true('inside_region') for h, s in sc.items()}
# Store all computed SummaryResult objects
sc.store(totdist, name="total_distance_p1")
sc.store(time_in, name="time_inside_region_s")

# Collate scalar outputs into a tidy DataFrame (optionally include tags)
summary_df = sc.to_df(include_tags=True)
print(summary_df.head())

# 9) Save summaries (round‑trip format)
sc.save(f"{OUT_DIR}/summaries", data_format="csv", overwrite=True)
```


