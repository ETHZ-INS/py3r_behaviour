End‑to‑end example performing behavior segmentation and running behavior flow analysis (BFA) on k-means clustering results. The pipeline introduces folder‑based loaders, batch preprocessing, feature generation and embedding, k-means clustering, and performing BFA. Paths are illustrative; adapt to your environment.

```python
# 1) Load a dataset of single‑view DLC CSVs into a TrackingCollection
from py3r.behaviour.tracking.tracking_collection import TrackingCollection
from py3r.behaviour.features.features_collection import FeaturesCollection
from py3r.behaviour.summary.summary_collection import SummaryCollection

DATA_DIR = "/data/recordings"            # e.g. contains OFT_id1.csv, OFT_id2.csv, ...
TAGS_CSV = "/data/tags.csv"              # optional, with columns: handle, treatment, genotype, ...
OUT_DIR  = "/outputs"                    # where to save summary outputs

tc = TrackingCollection.from_dlc_folder(folder_path=DATA_DIR, fps=25)

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

# 6) Compute features which will used for clustering
# The following features are exemplary, adjust accordingly.
# Speed of different keypoints
fc.speed("nose").store()
fc.speed("neck").store()
fc.speed("earr").store()
fc.speed("earl").store()
fc.speed("bodycentre").store()
fc.speed("hipl").store()
fc.speed("hipr").store()
fc.speed("tailbase").store()
# Angle between two lines, crossing in one specified keypoint
fc.azimuth_deviation("tailbase", "hipr", "hipl").store()
fc.azimuth_deviation("bodycentre", "tailbase", "neck").store()
fc.azimuth_deviation("neck", "bodycentre", "headcentre").store()
fc.azimuth_deviation("headcentre", "earr", "earl").store()
# Distance between two keypoints
fc.distance_between("nose", "headcentre").store()
fc.distance_between("neck", "headcentre").store()
fc.distance_between("neck", "bodycentre").store()
fc.distance_between("bcr", "bodycentre").store()
fc.distance_between("bcl", "bodycentre").store()
fc.distance_between("tailbase", "bodycentre").store()
fc.distance_between("tailbase", "hipr").store()
fc.distance_between("tailbase", "hipl").store()
fc.distance_between("bcr", "hipr").store()
fc.distance_between("bcl", "hipl").store()
fc.distance_between("bcl", "earl").store()
fc.distance_between("bcr", "earr").store()
fc.distance_between("nose", "earr").store()
fc.distance_between("nose", "earl").store()
# Area spanned by three or four keypoints
fc.area_of_boundary(["tailbase", "hipr", "hipl"], median=False).store()
fc.area_of_boundary(["hipr", "hipl", "bcl", "bcr"], median=False).store()
fc.area_of_boundary(["bcr", "earr", "earl", "bcl"], median=False).store()
fc.area_of_boundary(["earr", "nose", "earl"], median=False).store()

# 7) Create dictionary that defines the time-embedding of the different features (
features = fc[1].data.columns
offset = list(np.arange(-15, 16, 1)) # features of each frame will be embedded for 15 frames before and after the current frame
embedding_dict = {f: offset for f in features}

# 8) Cluster the embedded feature space using k-means clustering
# The keyword n_clusters defines the number of clusters used.
cluster_labels, centroids, _ = fc.cluster_embedding(embedding_dict=embedding_dict, n_clusters = 25)
cluster_labels.store("kmeans_25", overwrite=True)

# 9) (Optional) Save features to csv
fc.save(f"{OUT_DIR}/features", data_format="csv", overwrite=True)

# 10) Create SummaryCollection object and group it by one or more pre-defined tags
sc = SummaryCollection.from_features_collection(fc)
sc = sc.groupby(tags="group")

# 11) Perform behavior flow analysis on clustering results and print result
distances = sc.bfa(column = "kmeans_25", all_states = np.arange(0,25))
bfa_stats = SummaryCollection.bfa_stats(distances)
print(bfa_stats)
```


