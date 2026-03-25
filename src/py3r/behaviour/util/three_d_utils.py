import glob
import json
import os

import cv2
import numpy as np


def _make_circular_mask(shape, radius):
    """Return a binary mask with a filled circle of given radius centred in the image."""
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), int(radius), 255, -1)
    return mask


def _canonical_corners(corners):
    """
    Ensure corner[0] is always at the minimum (x+y) position.

    For even×even chessboards (e.g. 8×6) findChessboardCorners can return
    corners in two opposite orderings depending on board orientation in the
    image.  This causes silently wrong stereo correspondences while monocular
    calibration remains unaffected.  Canonicalising before calibration fixes
    the ambiguity.
    """
    first, last = corners[0, 0], corners[-1, 0]
    if float(last[0] + last[1]) < float(first[0] + first[1]):
        return corners[::-1]
    return corners


def find_chessboard_corners(image_paths, chessboard_size, mask_radius=None):
    """Find chessboard corners in a list of images.

    Parameters
    ----------
    image_paths : list[str]
    chessboard_size : tuple[int, int]
        Inner corners (cols, rows).
    mask_radius : int or None
        If set, apply a circular mask of this pixel radius (centred on the
        image) before detection.  Useful for fisheye lenses where the valid
        image area is a circle; prevents corners near the distorted boundary
        from being included.  Detection is performed on the masked image;
        subpixel refinement is performed on the original unmasked image so
        that accuracy in the valid region is preserved.

    Returns
    -------
    objpoints, imgpoints, valid_indices
    """
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : chessboard_size[0], 0 : chessboard_size[1]].T.reshape(-1, 2)

    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    det_flags = cv2.CALIB_CB_FAST_CHECK

    objpoints, imgpoints, valid_indices = [], [], []
    mask = None

    for idx, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if mask_radius is not None:
            if mask is None:
                mask = _make_circular_mask(gray.shape, mask_radius)
            det_gray = cv2.bitwise_and(gray, gray, mask=mask)
        else:
            det_gray = gray

        ret, corners = cv2.findChessboardCorners(det_gray, chessboard_size, det_flags)
        if ret:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), subpix_criteria)
            corners = _canonical_corners(corners)
            objpoints.append(objp.copy())
            imgpoints.append(corners)
            valid_indices.append(idx)

    return objpoints, imgpoints, valid_indices


def calibrate_stereo_system(
    view1_folder,
    view2_folder,
    chessboard_size,
    square_size,
    output_json,
    shared_intrinsics=False,
    mask_radius=None,
):
    """Calibrate a stereo camera system from chessboard image pairs.

    Parameters
    ----------
    view1_folder, view2_folder : str
        Folders containing paired PNG calibration images (sorted order matched).
    chessboard_size : tuple[int, int]
        Number of inner corners (cols, rows).
    square_size : float
        Physical size of one chessboard square in metres.
    output_json : str
        Path to write the calibration JSON.
    shared_intrinsics : bool
        If True, fit a single intrinsic model from observations pooled across
        both views before stereo calibration.  Appropriate when both cameras
        use the same physical lens (e.g. a split-sensor rig).  Default False.
    mask_radius : int or None
        If set, restrict corner detection to a circle of this radius (pixels)
        centred on the image.  Useful for circular fisheye images to avoid
        corners near the distorted boundary.  Default None (no mask).

    Notes
    -----
    Corner ordering is always canonicalised so that corner[0] maps to the
    minimum (x+y) position.  This is necessary for even×even boards (e.g.
    8×6) which have 180° rotational symmetry and can otherwise produce
    silently reversed correspondences between views.
    """
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

    _, imgpoints1, valid1 = find_chessboard_corners(images1, chessboard_size, mask_radius)
    _, imgpoints2, valid2 = find_chessboard_corners(images2, chessboard_size, mask_radius)

    valid_pairs = sorted(set(valid1) & set(valid2))
    if not valid_pairs:
        raise RuntimeError("No valid image pairs with detected chessboard corners.")

    valid1_lookup = {v: i for i, v in enumerate(valid1)}
    valid2_lookup = {v: i for i, v in enumerate(valid2)}

    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : chessboard_size[0], 0 : chessboard_size[1]].T.reshape(-1, 2)
    objp *= float(square_size)

    objpoints, ip1, ip2 = [], [], []
    for idx in valid_pairs:
        objpoints.append(objp.copy())
        ip1.append(imgpoints1[valid1_lookup[idx]])
        ip2.append(imgpoints2[valid2_lookup[idx]])

    if len(objpoints) < 3:
        raise RuntimeError(f"Only {len(objpoints)} valid pair(s) found; need at least 3.")

    sample = cv2.imread(images1[0])
    image_size = (sample.shape[1], sample.shape[0])

    stereo_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    if shared_intrinsics:
        rms_mono, K, dist, _, _ = cv2.calibrateCamera(
            objpoints + objpoints, ip1 + ip2, image_size, None, None
        )
        K1, d1, K2, d2 = K, dist, K.copy(), dist.copy()
        print(f"Shared monocular RMS: {rms_mono:.3f}px")
        rms_stereo, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            objpoints,
            ip1,
            ip2,
            K1,
            d1,
            K2,
            d2,
            image_size,
            flags=cv2.CALIB_FIX_INTRINSIC,
            criteria=stereo_criteria,
        )
    else:
        rms1, K1, d1, _, _ = cv2.calibrateCamera(objpoints, ip1, image_size, None, None)
        rms2, K2, d2, _, _ = cv2.calibrateCamera(objpoints, ip2, image_size, None, None)
        print(f"Monocular RMS: view1={rms1:.3f}px  view2={rms2:.3f}px")
        rms_stereo, K1, d1, K2, d2, R, T, E, F = cv2.stereoCalibrate(
            objpoints,
            ip1,
            ip2,
            K1,
            d1,
            K2,
            d2,
            image_size,
            flags=cv2.CALIB_USE_INTRINSIC_GUESS,
            criteria=stereo_criteria,
        )

    baseline = float(np.linalg.norm(T))
    angle_deg = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    print(f"Stereo RMS: {rms_stereo:.3f}px  baseline: {baseline:.4f}m  rotation: {angle_deg:.2f}°")

    calib = {
        "views": {
            view1_name: {"K": K1.tolist(), "dist": d1.tolist()},
            view2_name: {"K": K2.tolist(), "dist": d2.tolist()},
        },
        "relative_pose": {"R": R.tolist(), "T": T.tolist()},
        "image_size": list(image_size),
        "view_order": [view1_name, view2_name],
        "shared_intrinsics": shared_intrinsics,
        "mask_radius": mask_radius,
        "num_pairs": len(objpoints),
        "rms": {"stereo": float(rms_stereo)},
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
    n_frames = int(min(cap1.get(cv2.CAP_PROP_FRAME_COUNT), cap2.get(cv2.CAP_PROP_FRAME_COUNT)))
    if n_frames < num_images:
        raise ValueError(f"Not enough frames in the videos to extract {num_images} images.")
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
            ) and not is_blurred(gray2, min_sharpness, max_anisotropy, min_edge_density):
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
