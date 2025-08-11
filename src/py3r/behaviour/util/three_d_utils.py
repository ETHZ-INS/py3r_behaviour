import cv2
import numpy as np
import os
import glob
import json


def find_chessboard_corners(image_paths, chessboard_size):
    """Find chessboard corners in a list of images. Returns (objpoints, imgpoints, valid_indices)."""
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2)
    objpoints = []
    imgpoints = []
    valid_indices = []
    for idx, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            valid_indices.append(idx)
    return objpoints, imgpoints, valid_indices


def calibrate_stereo_system(
    view1_folder, view2_folder, chessboard_size, square_size, output_json
):
    """
    Calibrate a stereo camera system using chessboard images from two folders.
    """
    # Validate input folders
    if not os.path.isdir(view1_folder) or not os.path.isdir(view2_folder):
        raise FileNotFoundError("One or both view folders do not exist.")

    view1_name = os.path.basename(os.path.normpath(view1_folder))
    view2_name = os.path.basename(os.path.normpath(view2_folder))

    images1 = sorted(glob.glob(os.path.join(view1_folder, "*.png")))
    images2 = sorted(glob.glob(os.path.join(view2_folder, "*.png")))
    if not images1 or not images2:
        raise FileNotFoundError("No PNG images found in one or both folders.")
    if len(images1) != len(images2):
        raise ValueError("Image count mismatch between views.")

    # Find chessboard corners in both views
    objpoints1, imgpoints1, valid1 = find_chessboard_corners(images1, chessboard_size)
    objpoints2, imgpoints2, valid2 = find_chessboard_corners(images2, chessboard_size)

    # Use only pairs where both views found corners
    valid_pairs = set(valid1) & set(valid2)
    if not valid_pairs:
        raise RuntimeError("No valid image pairs with detected chessboard corners.")

    objpoints = []
    imgpoints1_final = []
    imgpoints2_final = []
    for idx in sorted(valid_pairs):
        objpoints.append(objpoints1[valid1.index(idx)])
        imgpoints1_final.append(imgpoints1[valid1.index(idx)])
        imgpoints2_final.append(imgpoints2[valid2.index(idx)])

    # Calibrate one camera
    gray = cv2.cvtColor(cv2.imread(images1[0]), cv2.COLOR_BGR2GRAY)
    ret, K, dist, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints1_final, gray.shape[::-1], None, None
    )
    if not ret:
        raise RuntimeError("Single camera calibration failed.")

    # Stereo calibration
    ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints1_final,
        imgpoints2_final,
        K,
        dist,
        K,
        dist,
        gray.shape[::-1],
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5),
    )
    if not ret:
        raise RuntimeError("Stereo calibration failed.")

    calib = {
        "views": {
            view1_name: {"K": K.tolist(), "dist": dist.tolist()},
            view2_name: {"K": K.tolist(), "dist": dist.tolist()},
        },
        "relative_pose": {"R": R.tolist(), "T": T.tolist()},
        "image_size": gray.shape[::-1],
        "view_order": [view1_name, view2_name],
    }

    with open(output_json, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"Calibration saved to {output_json} with views: {view1_name}, {view2_name}")


def extract_calibration_images(
    video1_path,
    video2_path,
    out_dir1,
    out_dir2,
    num_images=200,
    chessboard_size=(9, 6),
    min_sharpness=80.0,
    max_anisotropy=20.0,
    min_edge_density=0.005,
):
    """
    Extracts num_images frames from two videos for calibration.
    Only saves frames where a sharp chessboard is detected.
    """
    os.makedirs(out_dir1, exist_ok=True)
    os.makedirs(out_dir2, exist_ok=True)
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    n_frames = int(
        min(cap1.get(cv2.CAP_PROP_FRAME_COUNT), cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    )
    if n_frames < num_images:
        raise ValueError(
            f"Not enough frames in the videos to extract {num_images} images."
        )
    indices = np.linspace(0, n_frames - 1, num_images, dtype=int)
    saved = 0
    for idx in indices:
        cap1.set(cv2.CAP_PROP_POS_FRAMES, idx)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not (ret1 and ret2):
            continue
        print(f"Read frame {idx} from {video1_path} or {video2_path}")
        # convert to grayscale if needed
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = frame1.copy()
        if len(frame2.shape) == 3:
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = frame2.copy()

        # find chessboard corners
        chessboard_found1, _ = cv2.findChessboardCorners(gray1, chessboard_size, None)
        chessboard_found2, _ = cv2.findChessboardCorners(gray2, chessboard_size, None)

        if chessboard_found1 and chessboard_found2:
            print(f"Found chessboard in frame {idx} of {video1_path} and {video2_path}")
            # check not blurred
            if not is_blurred(
                gray1, min_sharpness, max_anisotropy, min_edge_density
            ) and not is_blurred(
                gray2, min_sharpness, max_anisotropy, min_edge_density
            ):
                print(f"Not blurred in frame {idx} of {video1_path} and {video2_path}")
                out1 = os.path.join(out_dir1, f"calib_{saved:03d}.png")
                out2 = os.path.join(out_dir2, f"calib_{saved:03d}.png")
                cv2.imwrite(out1, frame1)
                cv2.imwrite(out2, frame2)
                saved += 1
        if saved >= num_images:
            break
    cap1.release()
    cap2.release()
    print(f"Saved {saved} calibration image pairs to {out_dir1} and {out_dir2}")


def is_blurred(gray, min_lap_var=100.0, max_anisotropy=5.0, min_edge_density=0.01):
    """
    Returns True if the image is likely motion blurred.
    - min_lap_var: minimum Laplacian variance for sharpness
    - max_anisotropy: maximum allowed ratio of dominant to orthogonal gradient energy
    - min_edge_density: minimum fraction of edge pixels (Canny) required
    """
    # 1. Laplacian variance (focus)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < min_lap_var:
        return True  # blurry (could be defocus or motion)

    # 2. Directional gradient ratio (anisotropy)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    energy_x = np.sum(np.abs(sobelx))
    energy_y = np.sum(np.abs(sobely))
    if energy_x > energy_y:
        anisotropy = energy_x / (energy_y + 1e-6)
    else:
        anisotropy = energy_y / (energy_x + 1e-6)
    if anisotropy > max_anisotropy:
        return True  # strong directional blur

    # 3. Edge density (optional, to avoid blank/low-contrast images)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.mean(edges > 0)
    if edge_density < min_edge_density:
        return True  # not enough edges

    return False  # not blurred
