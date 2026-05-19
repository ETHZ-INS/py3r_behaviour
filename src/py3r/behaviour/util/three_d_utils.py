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
    (to avoid silent checkerboard orientation flips between views).
    """
    first, last = corners[0, 0], corners[-1, 0]
    if float(last[0] + last[1]) < float(first[0] + first[1]):
        return corners[::-1]
    return corners


def find_chessboard_corners(image_paths, chessboard_dims, mask_radius=None, canonical_corners=True):
    """Find chessboard corners in a list of images.

    Parameters
    ----------
    image_paths : list[str]
    chessboard_dims : tuple[int, int]
        Inner corners (cols, rows).
    mask_radius : int or None
        If set, apply a circular mask of this pixel radius (centred on the
        image) before detection.  Useful for fisheye lenses where the valid
        image area is a circle; prevents corners near the distorted boundary
        from being included.  Detection is performed on the masked image;
        subpixel refinement is performed on the original unmasked image so
        that accuracy in the valid region is preserved.
    canonical_corners : bool
        If True, enforce a canonical ordering so that corner[0] is always at
        the minimum (x+y) pixel position.  This eliminates the 180° ordering
        ambiguity that OpenCV has with even×even boards **within a single
        view** — but it only produces correct *cross-view* correspondences
        when both cameras point in the same direction (e.g. a split-chip rig).
        Set False for standard multi-angle stereo rigs where the min(x+y)
        pixel corner is a different physical board corner in each view.
        Default True.

    Returns
    -------
    objpoints, imgpoints, valid_indices
    """
    objp = np.zeros((chessboard_dims[0] * chessboard_dims[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : chessboard_dims[0], 0 : chessboard_dims[1]].T.reshape(-1, 2)

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

        ret, corners = cv2.findChessboardCorners(det_gray, chessboard_dims, det_flags)
        if ret:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), subpix_criteria)
            if canonical_corners:
                corners = _canonical_corners(corners)
            objpoints.append(objp.copy())
            imgpoints.append(corners)
            valid_indices.append(idx)

    return objpoints, imgpoints, valid_indices


def calibrate_stereo_system(
    view1_folder,
    view2_folder,
    chessboard_dims,
    square_size,
    output_json,
    shared_intrinsics=True,
    mask_radius=None,
    canonical_corners=None,
):
    """Calibrate a stereo camera system from chessboard image pairs.

    Parameters
    ----------
    view1_folder, view2_folder : str
        Folders containing paired PNG calibration images (sorted order matched).
    chessboard_dims : tuple[int, int]
        Number of inner corners (cols, rows).
    square_size : float
        Physical size of one chessboard square in metres.
    output_json : str
        Path to write the calibration JSON.
    shared_intrinsics : bool
        If True, fit a single intrinsic model from observations pooled across
        both views before stereo calibration.  Appropriate when both cameras
        use the same physical lens (e.g. a split-sensor rig).  Default True.
    mask_radius : int or None
        If set, restrict corner detection to a circle of this radius (pixels)
        centred on the image.  Useful for circular fisheye images to avoid
        corners near the distorted boundary.  Default None (no mask).
    canonical_corners : bool or None
        Controls the 180° corner-ordering fix for even×even boards.
        ``None`` (default) auto-selects: True when ``shared_intrinsics=True``
        (co-directional / split-chip cameras), False otherwise.
        See :func:`find_chessboard_corners` for a full explanation.

    """
    if canonical_corners is None:
        canonical_corners = shared_intrinsics
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

    _, imgpoints1, valid1 = find_chessboard_corners(
        images1, chessboard_dims, mask_radius, canonical_corners
    )
    _, imgpoints2, valid2 = find_chessboard_corners(
        images2, chessboard_dims, mask_radius, canonical_corners
    )

    valid_pairs = sorted(set(valid1) & set(valid2))
    if not valid_pairs:
        raise RuntimeError("No valid image pairs with detected chessboard corners.")

    valid1_lookup = {v: i for i, v in enumerate(valid1)}
    valid2_lookup = {v: i for i, v in enumerate(valid2)}

    objp = np.zeros((chessboard_dims[0] * chessboard_dims[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : chessboard_dims[0], 0 : chessboard_dims[1]].T.reshape(-1, 2)
    objp *= float(square_size)

    objpoints, ip1, ip2 = [], [], []
    for idx in valid_pairs:
        objpoints.append(objp.copy())
        ip1.append(imgpoints1[valid1_lookup[idx]])
        ip2.append(imgpoints2[valid2_lookup[idx]])

    if len(objpoints) < 3:
        raise RuntimeError(f"Only {len(objpoints)} valid pair(s) found; need at least 3.")

    print(f"using {len(objpoints)} image pairs for calibration")

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


def calibrate_stereo_from_videos(
    videos,
    output_folder,
    chessboard_dims,
    square_size,
    *,
    num_images=50,
    mask_radius=None,
    shared_intrinsics=True,
    frame_offset=0,
    min_step=3,
    min_pass1_yield=0.1,
    canonical_corners=None,
    min_board_displacement=20.0,
):
    """Extract calibration frames and calibrate a stereo rig in a single pass.

    Scans two videos simultaneously, detects the chessboard in both views for
    each candidate frame, saves accepted frames for QC, and then runs
    stereo calibration.

    A fail-fast check after pass 1 prevents wasted infill work: if fewer than
    ``min_pass1_yield`` of the pass 1 candidates yielded a valid detection,
    the video is considered insufficiently covered and scanning stops.

    Parameters
    ----------
    videos : dict[str, str]
        Exactly two entries mapping view name to video path, e.g.
        ``{"top": "/data/top.mp4", "offset": "/data/offset.mp4"}``.
        Insertion order determines view1 / view2.
    output_folder : str
        Root output directory (created if absent).  Structure::

            output_folder/
                calibration.json
                <view1_name>/calib_0000.png ...
                <view2_name>/calib_0000.png ...

    chessboard_dims : tuple[int, int]
        Inner corners (cols, rows), e.g. ``(8, 6)``.
    square_size : float
        Physical side length of one chessboard square in metres.
    num_images : int
        Target number of valid pairs to collect.  Calibration proceeds with
        however many are found if the video ends first (as long as ≥ 3).
        Default 50.
    mask_radius : int or None
        If set, restrict corner detection to a centred circle of this radius
        (pixels).  Subpixel refinement uses the unmasked image.  Default None.
    shared_intrinsics : bool
        Pool observations from both views to fit a single K/dist model before
        stereo calibration.  Appropriate for split-sensor rigs with identical
        optics.  Default True.
    frame_offset : int
        Frame index offset applied to the second video relative to the first
        (positive = second video starts later).  Default 0.
    min_step : int
        Minimum frame step for infill passes.  Scanning will not go denser
        than one check every ``min_step`` frames.  Default 3.
    min_pass1_yield : float
        Minimum fraction of pass 1 candidates that must yield a valid
        detection to proceed with infill.  If the detection rate is below
        this threshold the video is considered unsuitable and scanning stops
        early.  Default 0.1 (10 %).
    canonical_corners : bool or None
        Controls the 180° corner-ordering fix for even×even boards.
        ``None`` (default) auto-selects: True when ``shared_intrinsics=True``
        (co-directional / split-chip cameras), False otherwise.
        For multi-angle rigs (cameras pointing in different directions), pass
        False explicitly — the canonical fix is only reliable when both cameras
        see the board from the same direction.  See :func:`find_chessboard_corners`
        for a full explanation.
    min_board_displacement : float
        Minimum mean corner displacement (pixels) required between a new pair
        and the most recently accepted pair.  Pairs where the board has barely
        moved are rejected, preventing clusters of near-duplicate poses that
        give the stereo optimiser a flat landscape and allow it to drift to a
        wrong local minimum.  Default 20 px.  Set to 0 to disable.

    Returns
    -------
    dict
        The calibration dictionary (same content as calibration.json).
    """
    if len(videos) != 2:
        raise ValueError("Exactly two views are required for stereo calibration.")

    if canonical_corners is None:
        canonical_corners = shared_intrinsics

    view_names = list(videos.keys())
    v1, v2 = view_names
    paths = list(videos.values())

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, v1), exist_ok=True)
    os.makedirs(os.path.join(output_folder, v2), exist_ok=True)

    cap1 = cv2.VideoCapture(paths[0])
    cap2 = cv2.VideoCapture(paths[1])
    n1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    n2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))

    # Effective range: frames where both streams have valid data.
    n_eff = max(0, min(n1, n2 - frame_offset) if frame_offset >= 0 else min(n1 + frame_offset, n2))

    objp = np.zeros((chessboard_dims[0] * chessboard_dims[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : chessboard_dims[0], 0 : chessboard_dims[1]].T.reshape(-1, 2)
    objp *= float(square_size)

    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    det_flags = cv2.CALIB_CB_FAST_CHECK

    objpoints, ip1, ip2 = [], [], []
    circ_mask = None
    saved = 0
    last_frame1 = None

    def _check_frame(frame_idx):
        nonlocal saved, last_frame1, circ_mask
        idx2 = frame_idx + frame_offset
        if frame_idx >= n1 or idx2 < 0 or idx2 >= n2:
            return
        cap1.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, idx2)
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not (ret1 and ret2):
            return
        last_frame1 = frame1
        g1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        if mask_radius is not None:
            if circ_mask is None:
                circ_mask = _make_circular_mask(g1.shape, mask_radius)
            det1 = cv2.bitwise_and(g1, g1, mask=circ_mask)
            det2 = cv2.bitwise_and(g2, g2, mask=circ_mask)
        else:
            det1, det2 = g1, g2
        r1, c1 = cv2.findChessboardCorners(det1, chessboard_dims, det_flags)
        r2, c2 = cv2.findChessboardCorners(det2, chessboard_dims, det_flags)
        if r1 and r2:
            c1 = cv2.cornerSubPix(g1, c1, (11, 11), (-1, -1), subpix_criteria)
            c2 = cv2.cornerSubPix(g2, c2, (11, 11), (-1, -1), subpix_criteria)
            if canonical_corners:
                c1 = _canonical_corners(c1)
                c2 = _canonical_corners(c2)
            if min_board_displacement > 0 and ip1:
                disp = float(
                    np.mean(np.linalg.norm(c1.reshape(-1, 2) - ip1[-1].reshape(-1, 2), axis=1))
                )
                if disp < min_board_displacement:
                    return
            tag = f"calib_{saved:04d}"
            cv2.imwrite(os.path.join(output_folder, v1, f"{tag}.png"), frame1)
            cv2.imwrite(os.path.join(output_folder, v2, f"{tag}.png"), frame2)
            objpoints.append(objp.copy())
            ip1.append(c1)
            ip2.append(c2)
            saved += 1
            print(f"  pair {saved:>4}/{num_images}  (frame {frame_idx})")

    # Pass 1: exactly num_images evenly-spaced candidates across the full video.
    # This guarantees temporal coverage regardless of where the board appears.
    step = max(min_step, n_eff // max(1, num_images))
    checked: set = set()
    pass_num = 0

    while step >= min_step and saved < num_images:
        pass_num += 1
        pass_candidates = [f for f in range(0, n_eff, step) if f not in checked]
        before = saved
        for fi in pass_candidates:
            if saved >= num_images:
                break
            checked.add(fi)
            _check_frame(fi)
        found = saved - before
        print(
            f"Pass {pass_num}: {saved}/{num_images} pairs  "
            f"(+{found} from {len(pass_candidates)} new frames, step={step})"
        )

        # Fail-fast after pass 1: if detection rate is too low, infill won't help.
        if pass_num == 1:
            rate = found / max(1, len(pass_candidates))
            if rate < min_pass1_yield:
                print(
                    f"Pass 1 detection rate {rate:.1%} < min_pass1_yield "
                    f"{min_pass1_yield:.0%}; video lacks sufficient board "
                    "coverage — stopping early."
                )
                break

        step //= 2

    cap1.release()
    cap2.release()
    print(f"Total: {saved} valid pairs collected across {pass_num} pass(es)")

    if saved < 3:
        raise RuntimeError(f"Only {saved} valid pair(s) found; need at least 3.")

    image_size = (last_frame1.shape[1], last_frame1.shape[0])
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
        rms_record = {"shared_mono": float(rms_mono), "stereo": float(rms_stereo)}
    else:
        rms1, K1, d1, _, _ = cv2.calibrateCamera(objpoints, ip1, image_size, None, None)
        rms2, K2, d2, _, _ = cv2.calibrateCamera(objpoints, ip2, image_size, None, None)
        print(f"Monocular RMS: {v1}={rms1:.3f}px  {v2}={rms2:.3f}px")
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
        rms_record = {
            f"mono_{v1}": float(rms1),
            f"mono_{v2}": float(rms2),
            "stereo": float(rms_stereo),
        }

    baseline = float(np.linalg.norm(T))
    angle_deg = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    print(f"Stereo RMS: {rms_stereo:.3f}px  baseline: {baseline:.4f}m  rotation: {angle_deg:.2f}°")

    calib = {
        "views": {
            v1: {"K": K1.tolist(), "dist": d1.tolist()},
            v2: {"K": K2.tolist(), "dist": d2.tolist()},
        },
        "relative_pose": {"R": R.tolist(), "T": T.tolist()},
        "image_size": list(image_size),
        "view_order": [v1, v2],
        "shared_intrinsics": shared_intrinsics,
        "mask_radius": mask_radius,
        "num_pairs": saved,
        "rms": rms_record,
    }

    output_json = os.path.join(output_folder, "calibration.json")
    with open(output_json, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"Saved → {output_json}")

    return calib


def extract_calibration_images(
    video1_path,
    video2_path,
    out_dir1,
    out_dir2,
    num_images=200,
    chessboard_dims=(9, 6),
):
    """Extract up to ``num_images`` matched frame pairs where a chessboard is
    detected in both views.  Frames are sampled evenly across the video.

    .. note::
        For a single-step extract-and-calibrate workflow, prefer
        :func:`calibrate_stereo_from_videos` instead.
    """
    os.makedirs(out_dir1, exist_ok=True)
    os.makedirs(out_dir2, exist_ok=True)
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    n_frames = int(min(cap1.get(cv2.CAP_PROP_FRAME_COUNT), cap2.get(cv2.CAP_PROP_FRAME_COUNT)))
    indices = np.linspace(0, n_frames - 1, min(num_images * 5, n_frames), dtype=int)
    saved = 0
    for idx in indices:
        if saved >= num_images:
            break
        cap1.set(cv2.CAP_PROP_POS_FRAMES, idx)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not (ret1 and ret2):
            continue
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) if len(frame1.shape) == 3 else frame1
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY) if len(frame2.shape) == 3 else frame2
        found1, _ = cv2.findChessboardCorners(gray1, chessboard_dims, cv2.CALIB_CB_FAST_CHECK)
        found2, _ = cv2.findChessboardCorners(gray2, chessboard_dims, cv2.CALIB_CB_FAST_CHECK)
        if found1 and found2:
            cv2.imwrite(os.path.join(out_dir1, f"calib_{saved:03d}.png"), frame1)
            cv2.imwrite(os.path.join(out_dir2, f"calib_{saved:03d}.png"), frame2)
            saved += 1
    cap1.release()
    cap2.release()
    print(f"Saved {saved} calibration image pairs to {out_dir1} and {out_dir2}")


def is_blurred(gray, min_lap_var=100.0):
    """Return True if the image is likely blurred.

    Uses Laplacian variance as the focus measure: a sharp image has high
    variance because edges create large positive and negative responses;
    a blurry image suppresses those responses.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image.
    min_lap_var : float
        Minimum acceptable Laplacian variance.  Values below this threshold
        are considered blurred.  Default 100.
    """
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()) < min_lap_var
