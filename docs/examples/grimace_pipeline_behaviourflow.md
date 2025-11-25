```python
import json
import py3r.behaviour as p3b
import numpy as np

DATA_DIR = "/data/recordings"  # e.g. contains OFT_id1.csv, OFT_id2.csv, ...
TAGS_CSV = "/data/tags.csv"  # optional, with columns: handle, treatment, genotype, ...
OUT_DIR = "/outputs"  # where to save summary outputs
RECORDING_LENGTH = 300  # seconds

# Load the data into a TrackingCollection object
tc = p3b.TrackingCollection.from_yolo3r_folder(folder_path=DATA_DIR, fps=30)

# Strip the long prefixes from the column names in the data
tc.strip_column_names()

# Add tags from a CSV for grouping/analysis
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
tc.filter_likelihood(threshold=0.6)

# Smooth all points with mean centre window 3, with exception for environment points
tc.smooth_all(
    window=3, method="mean", overrides=[(["tr", "tl", "bl", "br"], "median", 30)]
)

# Rescale distance to metres according to corners of the GrimACE arena, here named 'tl' and 'br'
tc.rescale_by_known_distance(point1="tl", point2="br", distance_in_metres=0.13)

# 4) Basic QA such as checking length of recordings and ploting tracking trajectories
# Length check (per recording, assuming 10 min, time in seconds)
timecheck = tc.time_as_expected(
    mintime=RECORDING_LENGTH - (0.1 * RECORDING_LENGTH),
    maxtime=RECORDING_LENGTH + (0.1 * RECORDING_LENGTH),
)
for key, val in timecheck.items():
    if not val:
        raise ValueError(f"file {key} failed timecheck")

# Plot trajectories (per recording, using 'bodycentre' for trajectory of mouse and corners of OFT as static frame)
tc.plot(
    trajectories=["bodycentre"],
    static=["tr", "tl", "bl", "br"],
    lines=[("tr", "tl"), ("tl", "bl"), ("bl", "br"), ("br", "tr")],
)

# 5) Create FeaturesCollection object
fc = p3b.FeaturesCollection.from_tracking_collection(tc)

# 6) Compute features to be used for BehaviourFlow analysis
# (uncomment features to add to computation)
# Note: adding features increases memory requirements

# Accelerations
fc.acceleration("nose").store()
# fc.acceleration("headcentre").store()
fc.acceleration("neck").store()
# fc.acceleration("earr").store()
# fc.acceleration("earl").store()
fc.acceleration("bodycentre").store()
# fc.acceleration("bcl").store()
# fc.acceleration("bcr").store()
# fc.acceleration("hipl").store()
# fc.acceleration("hipr").store()
fc.acceleration("tailbase").store()

# Angular deviations
# fc.azimuth_deviation("tailbase", "hipr", "hipl").store()
fc.azimuth_deviation("bodycentre", "tailbase", "neck").store()
fc.azimuth_deviation("bodycentre", "bcr", "bcl").store()
fc.azimuth_deviation("neck", "bodycentre", "headcentre").store()
# fc.azimuth_deviation("bodycentre", "tailbase", "headcentre").store()
# fc.azimuth_deviation("bcl", "hipl", "earl").store()
# fc.azimuth_deviation("bcr", "hipr", "earr").store()
# fc.azimuth_deviation("nose", "earr", "earl").store()

# Distances
# fc.distance_between("nose", "headcentre").store()
fc.distance_between("neck", "headcentre").store()
fc.distance_between("neck", "bodycentre").store()
# fc.distance_between("bcr", "bodycentre").store()
# fc.distance_between("bcl", "bodycentre").store()
fc.distance_between("tailbase", "bodycentre").store()
# fc.distance_between("tailbase", "hipr").store()
# fc.distance_between("tailbase", "hipl").store()
# fc.distance_between("bcr", "hipr").store()
# fc.distance_between("bcl", "hipl").store()
# fc.distance_between("bcl", "earl").store()
# fc.distance_between("bcr", "earr").store()
# fc.distance_between("nose", "earr").store()
# fc.distance_between("nose", "earl").store()

# Areas
# fc.area_of_boundary(["tailbase", "hipr", "hipl"], median=False).store()
fc.area_of_boundary(["hipr", "hipl", "bcl", "bcr"], median=False).store()
fc.area_of_boundary(["bcr", "earr", "earl", "bcl"], median=False).store()
# fc.area_of_boundary(["earr", "nose", "earl"], median=False).store()

# Distance to GrimACE arena boundary
bdry = fc.define_boundary(["tl", "tr", "br", "bl"], scaling=1.0)
# fc.distance_to_boundary_static("nose", bdry, boundary_name="grimacebox").store()
fc.distance_to_boundary_static("neck", bdry, boundary_name="grimacebox").store()
fc.distance_to_boundary_static("bodycentre", bdry, boundary_name="grimacebox").store()
fc.distance_to_boundary_static("tailbase", bdry, boundary_name="grimacebox").store()

# Embed and cluster the features for BehaviourFlow analysis
embedding_dict = {f: np.arange(-15, 16, 1) for f in fc[0].data.columns}
labels, centroids, norm = fc.cluster_embedding(
    embedding_dict, n_clusters=25, lowmem=False, auto_normalize=True
)
labels.store(name="km25_standard_norm")


# 7) (Optional) Save features to disk (parquet format)
fc.save(f"{OUT_DIR}/features", data_format="parquet", overwrite=True)

# 8) Create SummaryCollection object
sc = p3b.SummaryCollection.from_features_collection(fc)

# 9) Compute summary measures per recording
# Total distance moved
sc.total_distance("bodycentre").store()

# 10) Collate scalar outputs into DataFrame and save results in CSV
summary_df = sc.to_df(include_tags=True)
summary_df.to_csv(f"{OUT_DIR}/OFT_results.csv")

# Group the summary collection by tags for BehaviourFlow analysis
sc_grouped = sc.groupby(["treatment", "timepoint"])

# Perform BehaviourFlow analysis
bfa_results = sc_grouped.bfa(
    "km25_standard_norm", all_states=np.arange(0, 25, 1).astype(int), numshuffles=1000
)

# Save the BehaviourFlow analysis results
with open(f"{OUT_DIR}/bfa_results.json", "w") as f:
    json.dump(bfa_results, f, indent=4)

# Compute the statistics for the BehaviourFlow analysis
bfa_stats = sc_grouped.bfa_stats(bfa_results)

# Save the BehaviourFlow analysis statistics
with open(f"{OUT_DIR}/bfa_stats.json", "w") as f:
    json.dump(bfa_stats, f, indent=4)
    
```