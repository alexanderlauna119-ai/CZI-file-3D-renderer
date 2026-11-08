import os
import argparse
import sys
if sys.platform.startswith("win"):
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

import numpy as np
import pyvista as pv
import vtk
vtk.vtkObject.GlobalWarningDisplayOff()
from aicspylibczi import CziFile
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox

try:
    import scipy.ndimage as ndi
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from tkinter import Tk, filedialog, colorchooser
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

# ====================================
# SETTINGS
# ====================================

Z_SCALE = 2
DOWNSAMPLE_XY = 2

# Algorithm for initial threshold on launch: "triangle" (ZEN Best Fit / Fiji) or "isodata" (ImageJ)
INITIAL_METHOD = "triangle"

# Rich default palette for multi-channel microscopy
DEFAULT_PALETTE = [
    "#0080FF",  # Channel 0: Blue (DAPI / Hoechst)
    "#00FF66",  # Channel 1: Green (GFP / Alexa 488)
    "#FF3333",  # Channel 2: Red (mCherry / Alexa 568)
    "#FFCC00",  # Channel 3: Yellow (YFP / Alexa 532)
    "#FF00FF",  # Channel 4: Magenta (Cy5 / Alexa 647)
    "#00FFFF",  # Channel 5: Cyan (CFP)
    "#FF8800",  # Channel 6: Orange
    "#FFFFFF",  # Channel 7: White
    "#AA00FF",  # Channel 8: Deep Purple
    "#00FFAA"   # Channel 9: Mint
]

def is_light_color(hex_str):
    """Determines whether a color is bright to select contrasting label text."""
    try:
        hex_clean = hex_str.lstrip("#")
        r, g, b = [int(hex_clean[i:i+2], 16) for i in (0, 2, 4)]
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return luminance > 160
    except Exception:
        return False

# ====================================
# FILE PICKER
# ====================================

def select_file():
    if TKINTER_AVAILABLE:
        try:
            root = Tk()
            root.withdraw()
            root.attributes("-topmost", True)

            filename = filedialog.askopenfilename(
                title="Select CZI File",
                filetypes=[
                    ("CZI Files", "*.czi"),
                    ("All Files", "*.*")
                ]
            )

            root.destroy()
            if filename:
                return filename
        except Exception as e:
            print("Tkinter file dialog failed, falling back to console prompt:", e)

    fn = input("Please enter the path to the CZI file: ").strip().strip('"').strip("'")
    return fn if fn else None

# ====================================
# CZI COLOR & METADATA HELPERS (ZEN / FIJI COMPATIBLE)
# ====================================

def parse_czi_color_value(raw):
    """
    Decodes color values stored in Zeiss CZI metadata XML into standard #RRGGBB hex.
    Supports:
    - 32-bit signed or unsigned ARGB integers (e.g., -16776961 -> #0000FF, -65536 -> #FF0000)
    - 9-char hex with alpha (#AARRGGBB -> #RRGGBB)
    - 7-char standard hex (#RRGGBB)
    - 8-char and 6-char hex without hash
    - Named CSS colors
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None

    # 1. Decimal integer (signed or unsigned 32-bit ARGB used by ZEN .NET framework)
    if s.startswith("-") or s.isdigit():
        try:
            val = int(s) & 0xFFFFFFFF
            r = (val >> 16) & 0xFF
            g = (val >> 8) & 0xFF
            b = val & 0xFF
            if r == 0 and g == 0 and b == 0:
                # Black is usually an uninitialized or empty placeholder in ZEN display settings
                return None
            return f"#{r:02X}{g:02X}{b:02X}"
        except Exception:
            pass

    # 2. Hex strings (#AARRGGBB, #RRGGBB, AARRGGBB, RRGGBB)
    clean = s.lstrip("#").strip()
    if all(c in "0123456789abcdefABCDEF" for c in clean):
        if len(clean) == 8:  # AARRGGBB: strip alpha (first 2 chars)
            rgb = clean[2:8].upper()
            if rgb == "000000":
                return None
            return f"#{rgb}"
        elif len(clean) == 6:  # RRGGBB
            rgb = clean.upper()
            if rgb == "000000":
                return None
            return f"#{rgb}"

    # 3. Named color fallback
    named = {
        "blue": "#0080FF",
        "green": "#00FF66",
        "red": "#FF3333",
        "yellow": "#FFCC00",
        "magenta": "#FF00FF",
        "cyan": "#00FFFF",
        "white": "#FFFFFF",
        "orange": "#FF8800",
        "purple": "#AA00FF",
        "gray": "#888888",
        "grey": "#888888"
    }
    return named.get(s.lower())

def color_from_fluorophore_name(raw_name):
    """
    Infers biological fluorophore color when no explicit DisplaySetting color was saved.
    """
    if raw_name is None:
        return None
    t = str(raw_name).strip().lower()

    if any(k in t for k in ["dapi", "hoechst", "blue", "cfp", "405", "violet", "bv421"]):
        return "#0080FF"
    if any(k in t for k in ["gfp", "green", "fitc", "alexa 488", "alexa488", "488", "egfp", "sybr"]):
        return "#00FF66"
    if any(k in t for k in ["mcherry", "rfp", "red", "alexa 568", "alexa 594", "568", "594", "texas red", "cy3", "tdtomato"]):
        return "#FF3333"
    if any(k in t for k in ["yfp", "yellow", "venus", "alexa 532", "532"]):
        return "#FFCC00"
    if any(k in t for k in ["cy5", "alexa 647", "alexa647", "far red", "far-red", "647", "draq5", "magenta", "deep red"]):
        return "#FF00FF"
    if any(k in t for k in ["cy7", "alexa 700", "alexa 750", "700", "750", "nir"]):
        return "#AA00FF"
    if any(k in t for k in ["cyan", "cerulean"]):
        return "#00FFFF"
    if any(k in t for k in ["orange", "cy3.5"]):
        return "#FF8800"
    if any(k in t for k in ["brightfield", "bf", "dic", "phase", "transmitted", "tl"]):
        return "#CCCCCC"

    return None

def color_from_emission_wavelength(raw_wl):
    """
    Maps emission or detection wavelength (in nm or meters) to standard spectrum colors.
    """
    if raw_wl is None:
        return None
    try:
        w = float(raw_wl)
        if w <= 0:
            return None
        # Convert meters to nm if given in meters (e.g., 5.25e-7 -> 525 nm)
        if w < 1e-3:
            w = w * 1e9

        if w < 470:
            return "#0080FF"  # Blue
        elif w < 495:
            return "#00FFFF"  # Cyan
        elif w < 555:
            return "#00FF66"  # Green
        elif w < 585:
            return "#FFCC00"  # Yellow
        elif w < 645:
            return "#FF3333"  # Red
        else:
            return "#FF00FF"  # Far-Red / Magenta
    except Exception:
        return None

# ====================================
# LOAD CZI + METADATA (VOXELS, CHANNELS, COLORS)
# ====================================

def load_czi(filename):
    print("\nOpening:", filename)
    czi = CziFile(filename)

    print("\nDimensions:")
    print(czi.dims)

    voxel_x = 1.0
    voxel_y = 1.0
    voxel_z = 1.0

    # Read image array and explicit dimension metadata from CziFile
    img, shp = czi.read_image()
    dim_names = [d[0] for d in shp]
    print(f"\nRaw image shape: {img.shape} with dimensions: {dim_names}")

    # Build canonical 4D array of shape (C, Z, Y, X)
    arr = img
    # If file contains scenes (S), timepoints (T), blocks (B), or mosaic tiles (M), select first index [0]
    for extra_dim in ['S', 'T', 'B', 'M', 'H', 'V', 'I']:
        if extra_dim in dim_names:
            idx = dim_names.index(extra_dim)
            if arr.shape[idx] > 1:
                print(f"Note: Dimension '{extra_dim}' has size {arr.shape[idx]}, selecting primary index [0].")
            arr = np.take(arr, indices=0, axis=idx)
            dim_names.pop(idx)

    # Resolve channel dimension ('C' or 'A' for RGB)
    if 'C' in dim_names:
        pass
    elif 'A' in dim_names:
        dim_names[dim_names.index('A')] = 'C'
    else:
        # Single channel image with no explicit C axis: prepend C=1
        arr = np.expand_dims(arr, 0)
        dim_names.insert(0, 'C')

    # Resolve Z dimension
    if 'Z' not in dim_names:
        # 2D image: insert Z=1 immediately before Y
        y_pos = dim_names.index('Y') if 'Y' in dim_names else 0
        arr = np.expand_dims(arr, y_pos)
        dim_names.insert(y_pos, 'Z')

    # Reorder remaining axes to standard (C, Z, Y, X)
    target_order = [dim_names.index('C'), dim_names.index('Z'), dim_names.index('Y'), dim_names.index('X')]
    raw_data = np.transpose(arr, target_order)

    num_channels = raw_data.shape[0]
    num_z = raw_data.shape[1]
    print(f"\nParsed volume shape: {raw_data.shape} -> {num_channels} channel(s), {num_z} Z-slice(s)")

    channel_names = [f"Channel {i}" for i in range(num_channels)]
    channel_colors = [DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)] for i in range(num_channels)]
    color_sources = ["Default palette fallback"] * num_channels

    # Parse metadata XML for voxel sizes, channel names and original Zeiss microscope display colors
    try:
        meta = czi.meta
        if isinstance(meta, ET.Element):
            root = meta
        elif isinstance(meta, (str, bytes)):
            root = ET.fromstring(meta)
        elif callable(meta):
            r = meta()
            root = r if isinstance(r, ET.Element) else ET.fromstring(r)
        elif hasattr(czi, "read_meta"):
            r = czi.read_meta()
            root = r if isinstance(r, ET.Element) else ET.fromstring(r)
        else:
            root = ET.fromstring(str(meta))

        # In-place strip XML namespaces so tag matching is direct and reliable across all Zeiss versions
        for elem in root.iter():
            if "}" in elem.tag:
                elem.tag = elem.tag.split("}", 1)[1]

        # 1. Voxel sizes (µm)
        distances = root.findall(".//Distance")
        for distance in distances:
            axis = distance.attrib.get("Id", "")
            value = distance.findtext("Value")
            if value is None:
                continue
            try:
                value_um = float(value) * 1e6
                if axis == "X":
                    voxel_x = value_um
                elif axis == "Y":
                    voxel_y = value_um
                elif axis == "Z":
                    voxel_z = value_um
            except Exception:
                pass

        print("\nVoxel size:")
        print(f"X = {voxel_x:.4f} µm")
        print(f"Y = {voxel_y:.4f} µm")
        print(f"Z = {voxel_z:.4f} µm")

        # 2. Extract Channel Names and Microscope Display Colors
        ds_nodes = (
            root.findall(".//DisplaySetting/Channels/Channel") or
            root.findall(".//DisplaySettings/Channels/Channel") or
            root.findall(".//DisplaySetting//Channel") or
            root.findall(".//DisplaySettings//Channel")
        )
        dim_nodes = (
            root.findall(".//Information/Image/Dimensions/Channels/Channel") or
            root.findall(".//Dimensions/Channels/Channel") or
            root.findall(".//Image/Dimensions/Channels/Channel")
        )
        info_ch_nodes = root.findall(".//Information/Image/Channels/Channel")

        def get_channel_node(node_list, ch_idx):
            if not node_list:
                return None
            target_ids = {f"Channel:{ch_idx}", str(ch_idx), f"Channel:{ch_idx}:0", f"Channel:{ch_idx+1}"}
            for n in node_list:
                nid = str(n.attrib.get("Id", "")).strip()
                if nid in target_ids:
                    return n
            if ch_idx < len(node_list):
                return node_list[ch_idx]
            return None

        print(f"\nMetadata scan: {len(ds_nodes)} display setting channel(s), {len(dim_nodes)} dimension channel(s) detected.")

        for i in range(num_channels):
            node_ds = get_channel_node(ds_nodes, i)
            node_dim = get_channel_node(dim_nodes, i)
            node_info = get_channel_node(info_ch_nodes, i)

            display_candidates = [n for n in [node_ds, node_dim, node_info] if n is not None]

            # Extract Channel Name
            found_name = None
            for n in display_candidates:
                for tag in ("Name", "DyeName", "Fluor", "ShortName"):
                    cand = n.attrib.get(tag) or n.findtext(tag)
                    if cand and cand.strip() and not cand.strip().lower().startswith("channel:"):
                        found_name = cand.strip()
                        break
                if found_name:
                    break

            if found_name:
                channel_names[i] = found_name

            # Extract Microscope Display Color
            found_color = None
            found_source = None

            for n, label in [(node_ds, "Zeiss ZEN DisplaySetting"), (node_dim, "CZI Dimensions"), (node_info, "CZI Info")]:
                if n is not None:
                    for tag in ("Color", "OriginalColor", "DisplayColor", "DyeColor"):
                        raw_c = n.findtext(tag) or n.attrib.get(tag)
                        parsed_c = parse_czi_color_value(raw_c)
                        if parsed_c:
                            found_color = parsed_c
                            found_source = f"{label} <{tag}>: {raw_c}"
                            break
                if found_color:
                    break

            if not found_color:
                for n in display_candidates:
                    for tag in ("DyeName", "Fluor", "Name", "ShortName"):
                        val = n.findtext(tag) or n.attrib.get(tag)
                        parsed_c = color_from_fluorophore_name(val)
                        if parsed_c:
                            found_color = parsed_c
                            found_source = f"Dye lookup '{val}'"
                            break
                    if found_color:
                        break

            if not found_color:
                for n in display_candidates:
                    for tag in ("DyeMaxEmission", "EmissionWavelength", "DetectionWavelength", "DyeEmissionWavelength"):
                        val = n.findtext(tag)
                        parsed_c = color_from_emission_wavelength(val)
                        if parsed_c:
                            found_color = parsed_c
                            found_source = f"Emission wavelength ({val} nm)"
                            break
                    if found_color:
                        break

            if not found_color:
                parsed_c = color_from_fluorophore_name(channel_names[i])
                if parsed_c:
                    found_color = parsed_c
                    found_source = f"Channel label '{channel_names[i]}'"

            if found_color:
                channel_colors[i] = found_color
                color_sources[i] = found_source

    except Exception as e:
        print("\nCould not read complete CZI metadata:", e)

    print(f"\nChannels loaded ({num_channels} total):")
    for idx in range(num_channels):
        print(f"  • Ch {idx} [{channel_names[idx]}]: color {channel_colors[idx]} ({color_sources[idx]})")

    return (
        raw_data,
        voxel_x,
        voxel_y,
        voxel_z,
        channel_names,
        channel_colors
    )

# ====================================
# ROBUST HISTOGRAM & THRESHOLD ALGORITHMS (ZEN & FIJI)
# ====================================

def compute_robust_histogram(vol, bins=256):
    """
    Computes a representative sample histogram of raw voxel intensities.
    Excludes black/zero padding artifacts (common in 3D confocal stacks and mosaic borders)
    so that background mode detection and auto-threshold algorithms are not skewed.
    """
    flat = vol.ravel()
    sample = np.random.choice(flat, 250000, replace=False) if len(flat) > 250000 else flat
    v_min = float(np.min(sample))
    v_max = float(np.max(sample))
    if v_max <= v_min:
        v_max = v_min + 1.0
        return np.ones(bins, dtype=int), np.linspace(v_min, v_max, bins + 1), v_min, v_max, sample

    counts, edges = np.histogram(sample, bins=bins, range=(v_min, v_max))
    counts_clean = counts.copy()

    # If the first bin has a huge disproportionate spike compared to neighbors (border padding),
    # suppress it just like ImageJ's AutoThresholder does ("data[0] = 0")
    if len(counts_clean) > 2 and counts_clean[0] > 2 * counts_clean[1]:
        counts_clean[0] = 0
    if len(counts_clean) > 2 and counts_clean[-1] > 2 * counts_clean[-2]:
        counts_clean[-1] = 0

    return counts_clean, edges, v_min, v_max, sample

def compute_triangle_threshold(counts, edges):
    """
    Zack's Triangle algorithm (Fiji / Zeiss ZEN Best Fit):
    Finds the inflection point / elbow where the steep background noise
    transitions into genuine fluorescence signal.
    Direct robust implementation of ImageJ AutoThresholder.java Triangle().
    """
    data = list(counts)
    n = len(data)

    # Find lowest bin with signal
    min_idx = 0
    for i in range(n):
        if data[i] > 0:
            min_idx = i
            break
    if min_idx > 0:
        min_idx -= 1

    # Find highest bin with signal
    min2 = 0
    for i in range(n - 1, 0, -1):
        if data[i] > 0:
            min2 = i
            break
    if min2 < n - 1:
        min2 += 1

    # Find peak (background mode)
    dmax = 0
    max_idx = 0
    for i in range(n):
        if data[i] > dmax:
            max_idx = i
            dmax = data[i]

    # Check skewness and reverse if needed
    inverted = False
    if (max_idx - min_idx) < (min2 - max_idx):
        inverted = True
        data = data[::-1]
        min_idx = n - 1 - min2
        max_idx = n - 1 - max_idx

    if min_idx == max_idx:
        return float((edges[min_idx] + edges[min_idx + 1]) / 2.0)

    # Line equation between peak and extreme
    nx = float(data[max_idx])
    ny = float(min_idx - max_idx)
    d = np.sqrt(nx * nx + ny * ny)
    if d < 1e-9:
        return float((edges[max_idx] + edges[max_idx + 1]) / 2.0)
    nx /= d
    ny /= d
    d = nx * min_idx + ny * data[min_idx]

    # Find split point with maximum distance to chord
    split = min_idx
    split_dist = 0.0
    for i in range(min_idx + 1, max_idx + 1):
        new_dist = nx * i + ny * data[i] - d
        if new_dist > split_dist:
            split = i
            split_dist = new_dist
    split -= 1

    res_idx = (n - 1 - split) if inverted else split
    res_idx = max(0, min(n - 1, res_idx))
    return float((edges[res_idx] + edges[res_idx + 1]) / 2.0)

def compute_isodata_threshold(counts, edges):
    """
    ImageJ 'Default' algorithm (IsoData / Ridler-Calvard iterative intermeans):
    Iteratively finds the threshold dividing the volume into foreground
    and background such that the threshold sits halfway between their means.
    Includes mode damping to protect against extreme background spikes.
    """
    data = list(counts)
    n = len(data)

    # Mode damping from ImageJ defaultIsoData
    mode = int(np.argmax(data))
    max_count = data[mode]
    data_no_mode = data.copy()
    data_no_mode[mode] = 0
    max_count2 = max(data_no_mode) if any(data_no_mode) else 0
    if max_count > 2 * max_count2 and max_count2 > 0:
        data[mode] = int(max_count2 * 1.5)

    data[0] = 0
    data[-1] = 0
    min_idx = 0
    while min_idx < n - 1 and data[min_idx] == 0:
        min_idx += 1
    max_idx = n - 1
    while max_idx > 0 and data[max_idx] == 0:
        max_idx -= 1

    if min_idx >= max_idx:
        return float((edges[0] + edges[-1]) / 2.0)

    moving_index = min_idx
    centers = (edges[:-1] + edges[1:]) / 2.0
    threshold_val = float(centers[moving_index])

    for _ in range(50):
        low_counts = data[min_idx : moving_index + 1]
        low_vals = centers[min_idx : moving_index + 1]
        high_counts = data[moving_index + 1 : max_idx + 1]
        high_vals = centers[moving_index + 1 : max_idx + 1]

        sum1 = float(np.sum(low_counts * low_vals))
        sum2 = float(np.sum(low_counts))
        sum3 = float(np.sum(high_counts * high_vals))
        sum4 = float(np.sum(high_counts))

        if sum2 == 0 or sum4 == 0:
            break

        threshold_val = ((sum1 / sum2) + (sum3 / sum4)) / 2.0
        next_idx = int(np.searchsorted(centers, threshold_val))
        if next_idx == moving_index or moving_index >= max_idx - 1:
            break
        moving_index = next_idx

    return float(threshold_val)

# ====================================
# FAST ZEN-STYLE HISTOGRAM & INTENSITY CONTROL
# ====================================

class ZenHistogramPopup:
    """Focused per-channel histogram controls with low-latency threshold editing.

    The threshold is moved directly on the histogram (no Matplotlib slider). Scene
    geometry is refreshed at a bounded rate during a drag and once on release.
    Supports seamless pull-up tab toggling [H], pin-on-top, and non-overlapping layout.
    """
    def __init__(
        self, volumes, channel_names, initial_colors, initial_thresholds,
        on_update_callback=None, on_color_callback=None, on_visibility_callback=None,
        on_filter_callback=None, initial_filter_state=False
    ):
        import time
        self._time = time
        self.volumes = volumes
        self.channel_names = list(channel_names)
        self.colors = list(initial_colors)
        self.num_channels = len(volumes)
        self.on_update = on_update_callback
        self.on_color_change = on_color_callback
        self.on_visibility = on_visibility_callback
        self.on_filter = on_filter_callback
        self.filter_enabled = bool(initial_filter_state)
        self.channel_vis = [True] * self.num_channels
        self.threshold_vals = list(initial_thresholds)
        self.active_channel = 0
        self.dragging = False
        self._updating = False
        self._is_visible = True
        self._is_pinned = False
        self.on_visibility_listener = None
        self.refocus_viewer_callback = None
        self._last_scene_update = 0.0
        self._scene_update_interval = 0.10
        self._range_zoom = [1.0] * self.num_channels
        self._range_center = []
        self.min_vals, self.max_vals, self.is_ints = [], [], []
        self.samples, self.hist_counts, self.hist_edges = [], [], []
        self.triangle_vals, self.isodata_vals = [], []

        # Precompute once; chart navigation never touches the voxel arrays again.
        for i, vol in enumerate(volumes):
            counts, edges, vmin, vmax, sample = compute_robust_histogram(vol)
            self.min_vals.append(vmin)
            self.max_vals.append(vmax)
            self.is_ints.append(np.issubdtype(vol.dtype, np.integer))
            self.samples.append(sample)
            self.hist_counts.append(counts)
            self.hist_edges.append(edges)
            self.triangle_vals.append(compute_triangle_threshold(counts, edges))
            self.isodata_vals.append(compute_isodata_threshold(counts, edges))
            self._range_center.append((vmin + vmax) / 2.0)

        plt.ion()
        self.fig = plt.figure(figsize=(10.6, 6.8), facecolor="#14181f")
        self.fig.canvas.manager.set_window_title("ZEN Histogram — Channel Intensity")
        
        # Generous bottom margin (0.285) guarantees ample vertical space:
        # X-axis numbers (~0.250) -> 'Raw intensity' label (~0.210) -> safety gap -> buttons (0.088 to 0.148) -> footer (0.026)
        self.fig.subplots_adjust(left=0.085, right=0.965, top=0.82, bottom=0.285)
        self.ax_hist = self.fig.add_subplot(111, facecolor="#1b212a")
        self.ax_hist.set_yscale("log")
        self.ax_hist.set_ylabel("Voxel count · log scale", color="#9fb0c3", fontsize=9.0)
        self.ax_hist.set_xlabel("Raw intensity", color="#9fb0c3", fontsize=9.0, labelpad=5)
        self.ax_hist.tick_params(colors="#8898aa", labelsize=8.5)
        for spine in self.ax_hist.spines.values():
            spine.set_color("#2e3947")
        self.ax_hist.grid(axis="y", color="#425164", alpha=0.25, linewidth=0.7)

        self.hist_bars = None
        self.hist_curve = None
        self.threshold_line = None
        self.threshold_span = None
        self.marker_lines = []
        self.info_text = None
        self.title_text = None
        self.channel_buttons = []
        self.visibility_button = None
        self.color_button = None
        self.value_box = None
        self.btn_tri = self.btn_iso = None
        self.btn_filter = None
        self.btn_pin = self.btn_hide = None

        # Top Header: Channel Selector Tabs
        top_y = 0.865
        gap = 0.008
        left = 0.085
        right_utility_w = 0.32  # Space reserved for Pin and Hide buttons
        avail_w = 0.965 - left - right_utility_w - 0.02
        btn_w = min(0.16, (avail_w - max(0, self.num_channels - 1) * gap) / max(1, self.num_channels))
        
        for i in range(self.num_channels):
            x = left + i * (btn_w + gap)
            ax = self.fig.add_axes([x, top_y, btn_w, 0.058])
            label = f"{i + 1}  {self.channel_names[i][:11]}"
            button = Button(ax, label, color="#202632", hovercolor="#323e50")
            button.label.set_color(self.colors[i])
            button.label.set_fontsize(8.5)
            button.label.set_fontweight("bold" if i == 0 else "normal")
            button.on_clicked(lambda event, idx=i: self.select_channel(idx))
            self.channel_buttons.append(button)

        # Top Header Utility Controls: Pin on Top & Hide back to 3D Viewer
        ax_pin = self.fig.add_axes([0.67, top_y, 0.135, 0.058])
        self.btn_pin = Button(ax_pin, "Float: OFF", color="#202632", hovercolor="#323e50")
        self.btn_pin.label.set_color("#88a0b5")
        self.btn_pin.label.set_fontsize(8.5)
        self.btn_pin.on_clicked(lambda event: self.toggle_pin())

        ax_hide = self.fig.add_axes([0.815, top_y, 0.150, 0.058])
        self.btn_hide = Button(ax_hide, "▼ Hide (Esc / H)", color="#263342", hovercolor="#394b60")
        self.btn_hide.label.set_color("#58d7ff")
        self.btn_hide.label.set_fontsize(8.5)
        self.btn_hide.label.set_fontweight("bold")
        self.btn_hide.on_clicked(lambda event: self.hide())

        # Clean Bottom Action Bar (y = 0.088, h = 0.060): Strictly partitioned non-overlapping layout
        y_ctrl = 0.088
        h_ctrl = 0.060

        # 1. Color button
        self.ax_color = self.fig.add_axes([0.085, y_ctrl, 0.062, h_ctrl])
        self.color_button = Button(self.ax_color, "Color", color=self.colors[0], hovercolor="#eeeeee")
        self.color_button.label.set_color("black" if is_light_color(self.colors[0]) else "white")
        self.color_button.label.set_fontsize(8.5)
        self.color_button.label.set_fontweight("bold")
        self.color_button.on_clicked(lambda event: self.open_color_dialog(self.active_channel))

        # 2. Visibility button
        self.ax_vis = self.fig.add_axes([0.153, y_ctrl, 0.078, h_ctrl])
        self.visibility_button = Button(self.ax_vis, "● Visible", color="#202b38", hovercolor="#324152")
        self.visibility_button.label.set_color("#60e6a0")
        self.visibility_button.label.set_fontsize(8.5)
        self.visibility_button.on_clicked(lambda event: self._toggle_active_visibility())

        # 3. Cutoff Segmented Control [ Cutoff | Value ]: Label axis + input box (NO external text overlap!)
        self.ax_cut_lbl = self.fig.add_axes([0.237, y_ctrl, 0.048, h_ctrl], facecolor="#18202b")
        self.ax_cut_lbl.set_xticks([])
        self.ax_cut_lbl.set_yticks([])
        for s in self.ax_cut_lbl.spines.values():
            s.set_color("#2d3b4e")
        self.cut_lbl_text = self.ax_cut_lbl.text(0.5, 0.5, "Cutoff", color="#9fb0c3", fontsize=8.5, ha="center", va="center")

        self.ax_value = self.fig.add_axes([0.285, y_ctrl, 0.075, h_ctrl], facecolor="#202b38")
        self.value_box = TextBox(self.ax_value, "", initial="", color="#202b38", hovercolor="#2f3e50")
        self.value_box.text_disp.set_color("white")
        self.value_box.on_submit(self._submit_active_value)

        # 4. Triangle auto-threshold
        self.ax_tri = self.fig.add_axes([0.368, y_ctrl, 0.170, h_ctrl])
        self.btn_tri = Button(self.ax_tri, "Triangle · active", color="#252a32", hovercolor="#3b4654")
        self.btn_tri.label.set_color("#ffd84d")
        self.btn_tri.label.set_fontsize(8.5)
        self.btn_tri.on_clicked(lambda event: self.apply_triangle_values(active_only=True))

        # 5. IsoData auto-threshold
        self.ax_iso = self.fig.add_axes([0.546, y_ctrl, 0.170, h_ctrl])
        self.btn_iso = Button(self.ax_iso, "IsoData · active", color="#252a32", hovercolor="#3b4654")
        self.btn_iso.label.set_color("#4ae2ff")
        self.btn_iso.label.set_fontsize(8.5)
        self.btn_iso.on_clicked(lambda event: self.apply_isodata_values(active_only=True))

        # 6. Despeckle / Isolated 1-5px Dots Filter Button
        self.ax_filter = self.fig.add_axes([0.724, y_ctrl, 0.241, h_ctrl])
        btn_filter_text = "● Filter Dots: ON (1-5px)" if self.filter_enabled else "Filter Dots: OFF (1-5px)"
        btn_filter_col = "#163828" if self.filter_enabled else "#202632"
        btn_filter_hover = "#22543d" if self.filter_enabled else "#323e50"
        self.btn_filter = Button(self.ax_filter, btn_filter_text, color=btn_filter_col, hovercolor=btn_filter_hover)
        self.btn_filter.label.set_color("#4be38a" if self.filter_enabled else "#8898aa")
        self.btn_filter.label.set_fontsize(8.5)
        self.btn_filter.label.set_fontweight("bold" if self.filter_enabled else "normal")
        self.btn_filter.on_clicked(lambda event: self.toggle_filter())

        self.footer = self.fig.text(
            0.5, 0.026,
            "Drag line · [← / →] adjust · [D] filter 1-5px dots · [↑ / ↓] switch ch · [H] hide to 3D",
            ha="center", color="#75869a", fontsize=8.2
        )

        # Event connections
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("close_event", self.on_window_close)

        # Intercept Tk window manager close so it hides instead of destroying
        try:
            self.fig.canvas.manager.window.protocol("WM_DELETE_WINDOW", self.hide)
        except Exception:
            pass

        self.select_channel(0, redraw=False)
        plt.show(block=False)

    def _format(self, idx, value):
        return f"{int(round(value)):,}" if self.is_ints[idx] else f"{value:,.3f}"

    def _display_range(self, idx):
        lo, hi = self.min_vals[idx], self.max_vals[idx]
        full = max(hi - lo, 1.0)
        width = full / self._range_zoom[idx]
        center = self._range_center[idx]
        left = max(lo, center - width / 2)
        right = min(hi, center + width / 2)
        if right - left < min(full, 1.0):
            return lo, hi
        return left, right

    def select_channel(self, ch_idx, redraw=True):
        if not (0 <= ch_idx < self.num_channels):
            return
        self.active_channel = ch_idx
        for i, button in enumerate(self.channel_buttons):
            if i == ch_idx:
                button.ax.set_facecolor("#324154")
                button.label.set_color(self.colors[i])
                button.label.set_fontweight("bold")
            else:
                button.ax.set_facecolor("#202632")
                button.label.set_color("#8da0b5")
                button.label.set_fontweight("normal")
        idx = ch_idx

        self.ax_hist.clear()
        self.ax_hist.set_yscale("log")
        self.ax_hist.set_ylabel("Voxel count · log scale", color="#9fb0c3", fontsize=9.5)
        self.ax_hist.set_xlabel("Raw intensity", color="#9fb0c3", fontsize=9.0, labelpad=5)
        self.ax_hist.tick_params(colors="#8898aa", labelsize=8.5)
        for spine in self.ax_hist.spines.values():
            spine.set_color("#2e3947")
        self.ax_hist.grid(axis="y", color="#425164", alpha=0.25, linewidth=0.7)

        counts = self.hist_counts[idx]
        edges = self.hist_edges[idx]
        centers = 0.5 * (edges[:-1] + edges[1:])
        col = self.colors[idx]

        self.hist_curve, = self.ax_hist.plot(centers, np.maximum(counts, 0.5), color=col, linewidth=1.8, zorder=2)
        self.hist_bars = self.ax_hist.fill_between(centers, 0.5, np.maximum(counts, 0.5), color=col, alpha=0.22, zorder=1)

        val = self.threshold_vals[idx]
        self.threshold_span = self.ax_hist.axvspan(val, self.max_vals[idx], color=col, alpha=0.15, zorder=0)
        self.threshold_line = self.ax_hist.axvline(val, color="white", linestyle="--", linewidth=1.9, zorder=4)

        tri = self.triangle_vals[idx]
        iso = self.isodata_vals[idx]
        self.marker_lines = [
            self.ax_hist.axvline(tri, color="#ffd84d", linestyle=":", linewidth=1.4, alpha=0.85, zorder=3),
            self.ax_hist.axvline(iso, color="#4ae2ff", linestyle=":", linewidth=1.4, alpha=0.85, zorder=3)
        ]

        left, right = self._display_range(idx)
        self.ax_hist.set_xlim(left, right)
        visible_counts = counts[(centers >= left) & (centers <= right)]
        max_c = np.max(visible_counts) if len(visible_counts) > 0 and np.max(visible_counts) > 0 else np.max(counts)
        self.ax_hist.set_ylim(bottom=0.8, top=max(10.0, max_c * 1.5))

        # Unified, collision-free status badge (incorporates channel name directly, zero overlapping text)
        self.info_text = self.ax_hist.text(
            0.985, 0.94, "",
            transform=self.ax_hist.transAxes,
            ha="right", va="top",
            color="#e2ebf5", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.38", facecolor="#141922", edgecolor=col, alpha=0.92, linewidth=0.9)
        )

        self._set_value_box(val)
        self.color_button.ax.set_facecolor(col)
        self.color_button.label.set_color("black" if is_light_color(col) else "white")
        self.set_channel_visibility(idx, self.channel_vis[idx])
        self._update_info()

        if redraw:
            self.fig.canvas.draw_idle()

    def _set_value_box(self, val):
        if self.value_box is not None:
            formatted = self._format(self.active_channel, val)
            self.value_box.set_val(formatted)

    def _update_info(self):
        idx = self.active_channel
        val = self.threshold_vals[idx]
        sample = self.samples[idx]
        pct = (sample < val).mean() * 100.0
        val_str = self._format(idx, val)
        tri_str = self._format(idx, self.triangle_vals[idx])
        iso_str = self._format(idx, self.isodata_vals[idx])
        summary = f"{self.channel_names[idx]} (Ch {idx + 1})   |   Cutoff: {val_str} ({pct:.1f}% filtered)   |   Triangle: {tri_str}   |   IsoData: {iso_str}"
        if self.info_text is not None:
            self.info_text.set_text(summary)

    def _submit_active_value(self, text):
        idx = self.active_channel
        try:
            clean = str(text).strip().replace(",", "")
            if clean.endswith("%"):
                val = self.min_vals[idx] + float(clean[:-1]) / 100.0 * (self.max_vals[idx] - self.min_vals[idx])
            else:
                val = float(clean)
        except (TypeError, ValueError):
            self._set_value_box(self.threshold_vals[idx])
            return
        self.set_threshold_internal(idx, val, origin="text", final=True)

    def on_press(self, event):
        if event.inaxes is self.ax_hist and event.xdata is not None and event.button == 1:
            self.dragging = True
            self._set_threshold_from_x(event.xdata, final=False)

    def on_motion(self, event):
        if self.dragging and event.inaxes is self.ax_hist and event.xdata is not None:
            self._set_threshold_from_x(event.xdata, final=False)

    def on_release(self, event):
        if self.dragging:
            self.dragging = False
            self._notify_scene(self.active_channel, final=True)

    def _set_threshold_from_x(self, value, final=False):
        self.set_threshold_internal(self.active_channel, value, origin="histogram", final=final)

    def on_key(self, event):
        key = (event.key or "").lower()
        idx = self.active_channel
        if key in ("h", "escape"):
            self.hide()
        elif key in ("up", "pageup", "tab"):
            self.select_channel((idx + 1) % self.num_channels)
        elif key in ("down", "pagedown", "shift+tab"):
            self.select_channel((idx - 1) % self.num_channels)
        elif key in ("1", "2", "3", "4", "5", "6", "7", "8", "9"):
            ch = int(key) - 1
            if ch < self.num_channels:
                self.select_channel(ch)
        elif key == "home":
            self._fit_active()
        elif key in ("d", "f"):
            self.toggle_filter()
        elif key == "v":
            self._toggle_active_visibility()
        elif key == "t":
            self.apply_triangle_values(active_only=True)
        elif key == "i":
            self.apply_isodata_values(active_only=True)
        elif key in ("left", "right"):
            span = self.max_vals[idx] - self.min_vals[idx]
            step = max(span / 1000.0, 1.0 if self.is_ints[idx] else span / 1000.0)
            self.set_threshold_internal(idx, self.threshold_vals[idx] + (-step if key == "left" else step), origin="key", final=True)

    def set_threshold_internal(self, ch_idx, val, origin="direct", final=False):
        if self._updating or val is None:
            return
        self._updating = True
        try:
            val = max(self.min_vals[ch_idx], min(self.max_vals[ch_idx], float(val)))
            self.threshold_vals[ch_idx] = val
            if ch_idx == self.active_channel:
                self.threshold_line.set_xdata([val, val])
                if self.threshold_span is not None:
                    try:
                        self.threshold_span.remove()
                    except Exception:
                        pass
                self.threshold_span = self.ax_hist.axvspan(val, self.max_vals[ch_idx], color=self.colors[ch_idx], alpha=0.15, zorder=0)
                self._set_value_box(val) if origin != "text" else None
                self._update_info()
                self.fig.canvas.draw_idle()
            self._notify_scene(ch_idx, final=final)
        finally:
            self._updating = False

    def _notify_scene(self, ch_idx, final=False):
        if not self.on_update:
            return
        now = self._time.time()
        if final or (now - self._last_scene_update >= self._scene_update_interval):
            self._last_scene_update = now
            self.on_update(ch_idx, self.threshold_vals[ch_idx])

    def toggle_filter(self):
        """Toggles the 1-5px isolated speckle/dot background filter."""
        self.filter_enabled = not self.filter_enabled
        self.update_filter_button_ui()
        if self.on_filter:
            self.on_filter(self.filter_enabled)

    def set_filter_state(self, is_enabled):
        """Synchronizes button state when toggled externally from 3D viewer or hotkey."""
        if self.filter_enabled != is_enabled:
            self.filter_enabled = is_enabled
            self.update_filter_button_ui()

    def update_filter_button_ui(self):
        if self.btn_filter is None:
            return
        try:
            if self.filter_enabled:
                self.btn_filter.label.set_text("● Filter Dots: ON (1-5px)")
                self.btn_filter.label.set_color("#4be38a")
                self.btn_filter.label.set_fontweight("bold")
                self.ax_filter.set_facecolor("#163828")
            else:
                self.btn_filter.label.set_text("Filter Dots: OFF (1-5px)")
                self.btn_filter.label.set_color("#8898aa")
                self.btn_filter.label.set_fontweight("normal")
                self.ax_filter.set_facecolor("#202632")
            self.fig.canvas.draw_idle()
        except Exception:
            pass

    def _zoom_active(self, factor):
        idx = self.active_channel
        self._range_zoom[idx] = max(1.0, min(64.0, self._range_zoom[idx] * factor))
        self._range_center[idx] = self.threshold_vals[idx]
        left, right = self._display_range(idx)
        self.ax_hist.set_xlim(left, right)
        self.fig.canvas.draw_idle()

    def _fit_active(self):
        idx = self.active_channel
        self._range_zoom[idx] = 1.0
        self._range_center[idx] = (self.min_vals[idx] + self.max_vals[idx]) / 2.0
        self.ax_hist.set_xlim(self.min_vals[idx], self.max_vals[idx])
        self.fig.canvas.draw_idle()

    def set_channel_visibility(self, ch_idx, is_visible):
        self.channel_vis[ch_idx] = is_visible
        if ch_idx == self.active_channel and self.visibility_button:
            if is_visible:
                self.visibility_button.label.set_text("● Visible")
                self.visibility_button.label.set_color("#60e6a0")
                self.visibility_button.ax.set_facecolor("#202b38")
            else:
                self.visibility_button.label.set_text("○ Hidden")
                self.visibility_button.label.set_color("#8898aa")
                self.visibility_button.ax.set_facecolor("#18202a")
            self.fig.canvas.draw_idle()

    def _toggle_active_visibility(self):
        idx = self.active_channel
        new_vis = not self.channel_vis[idx]
        self.set_channel_visibility(idx, new_vis)
        if self.on_visibility:
            self.on_visibility(idx, new_vis)

    def open_color_dialog(self, ch_idx):
        if TKINTER_AVAILABLE:
            try:
                root = Tk()
                root.withdraw()
                root.attributes("-topmost", True)
                result = colorchooser.askcolor(color=self.colors[ch_idx], title=f"Choose color for {self.channel_names[ch_idx]}")
                root.destroy()
                if result and result[1]:
                    self.set_channel_color(ch_idx, result[1])
                    return
            except Exception:
                pass
        current = DEFAULT_PALETTE.index(self.colors[ch_idx]) if self.colors[ch_idx] in DEFAULT_PALETTE else -1
        self.set_channel_color(ch_idx, DEFAULT_PALETTE[(current + 1) % len(DEFAULT_PALETTE)])

    def set_channel_color(self, ch_idx, new_color):
        self.colors[ch_idx] = new_color
        if ch_idx == self.active_channel:
            self.select_channel(ch_idx)
        else:
            self.channel_buttons[ch_idx].label.set_color(new_color)
            self.fig.canvas.draw_idle()
        if self.on_color_change:
            self.on_color_change(ch_idx, new_color)

    def apply_triangle_values(self, active_only=False):
        targets = [self.active_channel] if active_only else range(self.num_channels)
        for i in targets:
            self.set_threshold_internal(i, self.triangle_vals[i], origin="button", final=True)

    def apply_isodata_values(self, active_only=False):
        targets = [self.active_channel] if active_only else range(self.num_channels)
        for i in targets:
            self.set_threshold_internal(i, self.isodata_vals[i], origin="button", final=True)

    def apply_min_values(self, active_only=False):
        targets = [self.active_channel] if active_only else range(self.num_channels)
        for i in targets:
            self.set_threshold_internal(i, self.min_vals[i], origin="button", final=True)

    def toggle_visibility(self):
        if self._is_visible:
            self.hide()
        else:
            self.show()

    def hide(self):
        self._is_visible = False
        try:
            m = self.fig.canvas.manager
            if hasattr(m, "window"):
                win = m.window
                if hasattr(win, "withdraw"):
                    win.withdraw()
                elif hasattr(win, "hide"):
                    win.hide()
                elif hasattr(win, "Show"):
                    win.Show(False)
            elif hasattr(m, "frame") and hasattr(m.frame, "Show"):
                m.frame.Show(False)
        except Exception:
            pass
        if sys.platform == "win32":
            self._win32_show(False)
        if self.on_visibility_listener:
            try:
                self.on_visibility_listener(False)
            except Exception:
                pass
        if self.refocus_viewer_callback:
            try:
                self.refocus_viewer_callback()
            except Exception:
                pass

    def show(self):
        self._is_visible = True
        try:
            m = self.fig.canvas.manager
            if hasattr(m, "window"):
                win = m.window
                if hasattr(win, "deiconify"):
                    win.deiconify()
                    win.lift()
                elif hasattr(win, "show"):
                    win.show()
                    if hasattr(win, "raise_"):
                        win.raise_()
                elif hasattr(win, "Show"):
                    win.Show(True)
                    if hasattr(win, "Raise"):
                        win.Raise()
            elif hasattr(m, "frame") and hasattr(m.frame, "Show"):
                m.frame.Show(True)
                if hasattr(m.frame, "Raise"):
                    m.frame.Raise()
        except Exception:
            pass
        if sys.platform == "win32":
            self._win32_show(True)
            self._win32_bring_to_front()
            if self._is_pinned:
                self._win32_set_topmost(True)
        try:
            self.fig.canvas.draw_idle()
        except Exception:
            pass
        if self.on_visibility_listener:
            try:
                self.on_visibility_listener(True)
            except Exception:
                pass

    def toggle_pin(self):
        self._is_pinned = not self._is_pinned
        if self.btn_pin:
            if self._is_pinned:
                self.btn_pin.label.set_text("Float: ON")
                self.btn_pin.label.set_color("#5af0a5")
                self.btn_pin.ax.set_facecolor("#1e3b2e")
            else:
                self.btn_pin.label.set_text("Float: OFF")
                self.btn_pin.label.set_color("#88a0b5")
                self.btn_pin.ax.set_facecolor("#202632")
            self.fig.canvas.draw_idle()
        try:
            m = self.fig.canvas.manager
            if hasattr(m, "window"):
                win = m.window
                if hasattr(win, "attributes"):
                    win.attributes("-topmost", self._is_pinned)
        except Exception:
            pass
        if sys.platform == "win32":
            self._win32_set_topmost(self._is_pinned)

    def on_window_close(self, event):
        self.hide()

    def _get_hwnd(self):
        if sys.platform != "win32":
            return None
        import ctypes
        user32 = ctypes.windll.user32
        target_hwnd = None
        def enum_cb(hwnd, _):
            nonlocal target_hwnd
            length = user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buff = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buff, length + 1)
                if "ZEN Histogram" in buff.value:
                    target_hwnd = hwnd
                    return False
            return True
        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
        try:
            user32.EnumWindows(WNDENUMPROC(enum_cb), 0)
        except Exception:
            pass
        return target_hwnd

    def _win32_show(self, show=True):
        hwnd = self._get_hwnd()
        if hwnd:
            import ctypes
            ctypes.windll.user32.ShowWindow(hwnd, 5 if show else 0)

    def _win32_set_topmost(self, topmost=True):
        hwnd = self._get_hwnd()
        if hwnd:
            import ctypes
            ctypes.windll.user32.SetWindowPos(hwnd, -1 if topmost else -2, 0, 0, 0, 0, 0x0003)

    def _win32_bring_to_front(self):
        hwnd = self._get_hwnd()
        if hwnd:
            import ctypes
            ctypes.windll.user32.SetForegroundWindow(hwnd)

    def bring_to_front(self):
        self.show()


# ====================================
# BACKGROUND NOISE & ISOLATED SPECKLE FILTER
# ====================================

def remove_isolated_speckles(binary_mask, min_size=1, max_size=5):
    """
    Background noise filter for confocal microscopy.
    Removes isolated foreground clusters ('dots out in the middle of nowhere')
    whose connected component size is between min_size and max_size pixels/voxels.
    Preserves continuous biological structures (filaments, cell bodies, membranes).
    """
    if not SCIPY_AVAILABLE or not np.any(binary_mask):
        return binary_mask

    try:
        # Full connectivity structure (8-conn for 2D, 26-conn for 3D)
        struct = ndi.generate_binary_structure(binary_mask.ndim, binary_mask.ndim)
        labeled, num_features = ndi.label(binary_mask, structure=struct)
        if num_features == 0:
            return binary_mask

        counts = np.bincount(labeled.ravel())
        is_noise = np.zeros(len(counts), dtype=bool)
        is_noise[1:] = (counts[1:] >= min_size) & (counts[1:] <= max_size)

        return binary_mask & (~is_noise[labeled])
    except Exception as e:
        print(f"[Warning] Speckle filter encountered an issue: {e}")
        return binary_mask


# ====================================
# CREATE POINT CLOUD
# ====================================

def create_point_cloud(
    raw_volume,
    threshold_val,
    voxel_x,
    voxel_y,
    voxel_z,
    z_min_slice=1,
    z_max_slice=None,
    filter_speckles=False,
    speckle_min_size=1,
    speckle_max_size=5
):
    """
    Filters volume using true microscope intensity:
    Displays sections from z_min_slice to z_max_slice with intensity >= threshold_val.
    Retains true physical 3D elevation for every slice.
    """
    total_z = raw_volume.shape[0]
    if z_max_slice is None:
        z_max_slice = total_z

    z_start = max(0, min(total_z - 1, int(round(z_min_slice)) - 1))
    z_end = max(z_start + 1, min(total_z, int(round(z_max_slice))))

    # Sub-volume within the selected slice range
    sub_volume = raw_volume[z_start:z_end]

    binary_mask = (sub_volume >= threshold_val)
    if not np.any(binary_mask):
        return np.empty((0, 3)), np.empty(0)

    if filter_speckles:
        binary_mask = remove_isolated_speckles(
            binary_mask,
            min_size=speckle_min_size,
            max_size=speckle_max_size
        )

    local_z, y, x = np.where(binary_mask)

    if len(local_z) == 0:
        return np.empty((0, 3)), np.empty(0)

    intensity = sub_volume[local_z, y, x]
    actual_z = local_z + z_start

    points = np.column_stack(
        (
            x * voxel_x * DOWNSAMPLE_XY,
            y * voxel_y * DOWNSAMPLE_XY,
            actual_z * voxel_z * Z_SCALE
        )
    )

    return points, intensity

# ====================================
# RENDER
# ====================================

# ====================================
# ====================================
# SMOOTH CAMERA NAVIGATION & FILE EXPORT
# ====================================

WHEEL_ZOOM_STEP = 0.16  # Fractional camera dolly step per wheel notch

def install_smooth_camera_controls(plotter, focal_point=None, zoom_step=WHEEL_ZOOM_STEP):
    """
    Installs smooth, jitter-free camera navigation centered strictly on the specimen.
    Keeps WindowCenter firmly at (0, 0) to eliminate eccentric wobbling, drifting,
    and corkscrew distortion when rotating with standard trackball mouse controls.
    """
    iren = plotter.iren.interactor
    ren = plotter.renderer
    fixed_pivot = np.asarray(focal_point, dtype=float) if focal_point is not None else None

    if fixed_pivot is not None:
        try:
            plotter.camera.focal_point = tuple(fixed_pivot)
        except Exception:
            pass

    def on_wheel_forward(obj, event):
        try:
            cam = ren.GetActiveCamera()
            cam.SetWindowCenter(0.0, 0.0)
            # Smooth dolly toward the specimen focal point
            cam.Dolly(1.0 + zoom_step)
            ren.ResetCameraClippingRange()
            iren.Render()
        except Exception:
            pass

    def on_wheel_backward(obj, event):
        try:
            cam = ren.GetActiveCamera()
            cam.SetWindowCenter(0.0, 0.0)
            # Smooth dolly away from the specimen focal point
            cam.Dolly(1.0 / (1.0 + zoom_step))
            ren.ResetCameraClippingRange()
            iren.Render()
        except Exception:
            pass

    iren.AddObserver("MouseWheelForwardEvent", on_wheel_forward)
    iren.AddObserver("MouseWheelBackwardEvent", on_wheel_backward)

# Backwards compatibility alias
install_cursor_zoom = install_smooth_camera_controls


def build_combined_polydata(clouds, channel_actors, channel_colors, processed_volumes, state):
    """Combines active channels into a single unified PolyData with vertex RGB colors."""
    import matplotlib.colors as mcolors
    all_points = []
    all_colors = []

    for ch, cloud in enumerate(clouds):
        if ch < len(channel_actors):
            try:
                if not channel_actors[ch].GetVisibility():
                    continue
            except Exception:
                pass
        if cloud.n_points == 0:
            continue

        pts = cloud.points
        scalars = cloud["intensity"] if "intensity" in cloud.point_data else None

        base_hex = state["colors"][ch]
        r_base, g_base, b_base = mcolors.to_rgb(base_hex)

        if scalars is not None and len(scalars) == len(pts):
            vmin = float(state["thresholds"][ch])
            vmax = float(processed_volumes[ch].max()) if processed_volumes[ch].size > 0 else vmin + 1.0
            if vmax <= vmin:
                vmax = vmin + 1.0
            norm = np.clip((scalars - vmin) / (vmax - vmin), 0.18, 1.0)
            r = np.clip(r_base * norm * 255.0, 0, 255).astype(np.uint8)
            g = np.clip(g_base * norm * 255.0, 0, 255).astype(np.uint8)
            b = np.clip(b_base * norm * 255.0, 0, 255).astype(np.uint8)
            rgb = np.column_stack([r, g, b])
        else:
            r = int(r_base * 255)
            g = int(g_base * 255)
            b = int(b_base * 255)
            rgb = np.tile(np.array([r, g, b], dtype=np.uint8), (len(pts), 1))

        all_points.append(pts)
        all_colors.append(rgb)

    if not all_points:
        return None

    merged_points = np.vstack(all_points)
    merged_colors = np.vstack(all_colors)
    poly = pv.PolyData(merged_points)
    poly["RGB"] = merged_colors
    return poly


def build_fluorescence_cmap(hex_color, n_colors=256):
    """Build a perceptually graded microscopy fluorescence colormap from a base color.

    Ramps from a deep saturated tone near threshold (preserving faint structure without
    washing out) through vibrant fluorophore hue up to an intense glowing highlight peak.
    """
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(hex_color)
    h, s, v = mcolors.rgb_to_hsv([r, g, b])

    # Base: deep saturated tone at threshold
    c_low = mcolors.hsv_to_rgb([h, min(1.0, s * 1.15), max(0.24, v * 0.44)])
    # Mid: true fluorophore color
    c_mid = [r, g, b]
    # High: bright glowing fluorophore
    c_high = [min(1.0, c * 1.25 + 0.12) for c in (r, g, b)]
    # Peak: intense radiant bloom
    c_peak = [min(1.0, c * 0.82 + 0.35) for c in c_high]

    return mcolors.LinearSegmentedColormap.from_list(
        f"fluor_{hex_color}",
        [(0.0, c_low), (0.42, c_mid), (0.82, c_high), (1.0, c_peak)],
        N=n_colors
    )

def render(
    channels,
    voxel_x,
    voxel_y,
    voxel_z,
    channel_names,
    channel_colors,
    source_filename=None,
    auto_export=None
):
    # Dynamic window size matching monitor aspect ratio
    win_w, win_h = 1920, 1080
    if TKINTER_AVAILABLE:
        try:
            root = Tk()
            screen_w = root.winfo_screenwidth()
            screen_h = root.winfo_screenheight()
            root.destroy()
            win_w = max(1200, int(screen_w * 0.85))
            win_h = max(800, int(screen_h * 0.85))
        except Exception:
            pass

    plotter = pv.Plotter(window_size=(win_w, win_h))
    plotter.set_background(color="#05070d", top="#000000")

    # High-fidelity scientific 3D graphics pipeline
    try:
        plotter.enable_anti_aliasing("fxaa")
    except Exception:
        pass

    # High-performance 60 FPS rendering pipeline (avoids laggy 8-pass depth peeling on points)
    try:
        plotter.enable_point_smoothing()
    except Exception:
        pass

    try:
        plotter.enable_lightkit()
    except Exception:
        pass

    win_w, win_h = plotter.window_size

    num_channels = channels.shape[0]
    num_z = channels.shape[1] if channels.ndim == 4 else channels.shape[0]

    # Downsample XY once while preserving unclipped raw microscope intensities
    print("\nPreparing volumes (downsampling XY while preserving true raw intensities)...")
    processed_volumes = []
    initial_thresholds = []
    is_ints = []

    for ch in range(num_channels):
        raw_ch = channels[ch]
        ds_vol = raw_ch[:, ::DOWNSAMPLE_XY, ::DOWNSAMPLE_XY]
        processed_volumes.append(ds_vol)
        is_int = np.issubdtype(ds_vol.dtype, np.integer)
        is_ints.append(is_int)

        # Automatic intelligent threshold on launch using robust histogram
        counts, edges, vmin, vmax, samp = compute_robust_histogram(ds_vol)
        if INITIAL_METHOD == "isodata":
            init_val = compute_isodata_threshold(counts, edges)
            method_name = "ImageJ IsoData"
        else:
            init_val = compute_triangle_threshold(counts, edges)
            method_name = "ZEN/Fiji Triangle"

        initial_thresholds.append(init_val)
        cut_pct = float((ds_vol < init_val).mean() * 100.0)
        val_str = f"{int(round(init_val))}" if is_int else f"{init_val:.2f}"
        print(f"[{ch}] {channel_names[ch]} ({method_name}): threshold = {val_str} (filters {cut_pct:.1f}% background)")

    # Physical dimensions in um and slice elevations
    vol_shape = processed_volumes[0].shape
    full_z = vol_shape[0] * voxel_z * Z_SCALE
    full_y = vol_shape[1] * voxel_y * DOWNSAMPLE_XY
    full_x = vol_shape[2] * voxel_x * DOWNSAMPLE_XY
    z_coords = [z_i * voxel_z * Z_SCALE for z_i in range(num_z)]

    # Forward references for early callback safety during widget instantiation
    rep_z = None
    zen_popup = None
    slice_cage_actor = None
    checker_actor = None

    state = {
        "thresholds": list(initial_thresholds),
        "colors": list(channel_colors),
        "z_min": 1,
        "z_max": num_z,
        "filter_speckles": bool(auto_export.get("filter_speckles", False)) if auto_export else False,
        "speckle_min": 1,
        "speckle_max": 5,
        "ruler_mode": False,
        "ruler_points": [],
        "ruler_measurements": [],
        "ruler_active_marker": None
    }

    clouds = []
    channel_actors = []

    for ch in range(num_channels):
        print(f"Processing {channel_names[ch]} (Ch {ch})...")
        points, intensity = create_point_cloud(
            processed_volumes[ch],
            state["thresholds"][ch],
            voxel_x,
            voxel_y,
            voxel_z,
            z_min_slice=state["z_min"],
            z_max_slice=state["z_max"],
            filter_speckles=state.get("filter_speckles", False),
            speckle_min_size=state.get("speckle_min", 1),
            speckle_max_size=state.get("speckle_max", 5)
        )
        print(f"Points: {len(points):,}")

        cloud = pv.PolyData(points)
        if len(points) > 0:
            cloud["intensity"] = intensity

        vmin = float(state["thresholds"][ch])
        vmax = float(processed_volumes[ch].max()) if processed_volumes[ch].size > 0 else vmin + 1.0
        if vmax <= vmin:
            vmax = vmin + 1.0

        cmap = build_fluorescence_cmap(state["colors"][ch])

        actor = plotter.add_mesh(
            cloud,
            scalars="intensity" if len(points) > 0 else None,
            cmap=cmap,
            clim=[vmin, vmax],
            point_size=3.6,
            opacity=1.0,
            render_points_as_spheres=True,
            specular=0.55,
            specular_power=35.0,
            ambient=0.32,
            diffuse=0.78,
            lighting=True,
            show_scalar_bar=False,
            name=f"channel_{ch}"
        )

        clouds.append(cloud)
        channel_actors.append(actor)

    # =========================================================================
    # TOP-RIGHT HUD: PIXEL-PERFECT HIGH-RESOLUTION CARD OVERLAY
    # =========================================================================

    from PIL import Image, ImageDraw, ImageFont
    import colorsys

    HUD_W = 460
    HUD_MARGIN_TOP = 14
    HUD_MARGIN_RIGHT = 16
    HUD_HEADER_H = 66
    HUD_ROW_PITCH = 28
    HUD_H = HUD_HEADER_H + num_channels * HUD_ROW_PITCH + 36

    def boost_color_for_ui(color_hex, min_lightness=0.62):
        """Lifts dark microscope colors so blues, purples, and deep reds glow brightly on screen."""
        try:
            rgb = pv.Color(color_hex).float_rgb
            h, l, s = colorsys.rgb_to_hls(*rgb)
            l_boosted = max(l, min_lightness)
            s_boosted = min(1.0, max(s, 0.85))
            r_new, g_new, b_new = colorsys.hls_to_rgb(h, l_boosted, s_boosted)
            return pv.Color((r_new, g_new, b_new)).hex_rgb
        except Exception:
            return color_hex

    def pil_to_pyvista_imagedata(pil_img):
        """Convert PIL RGBA image to PyVista ImageData for GPU texture display with persistent VTK scalars."""
        from vtk.util import numpy_support
        arr = np.array(pil_img.convert("RGBA"), dtype=np.uint8)
        # VTK ImageData origin is bottom-left, flip vertically for direct rendering
        arr_flipped = np.ascontiguousarray(np.flipud(arr))
        h, w, _ = arr_flipped.shape

        vtk_img = vtk.vtkImageData()
        vtk_img.SetDimensions(w, h, 1)

        flat = np.ascontiguousarray(arr_flipped.reshape(-1, 4))
        vtk_arr = numpy_support.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        vtk_arr.SetName("RGBA")
        vtk_arr.SetNumberOfComponents(4)
        vtk_img.GetPointData().SetScalars(vtk_arr)
        vtk_img.GetPointData().SetActiveScalars("RGBA")

        pv_img = pv.wrap(vtk_img)
        pv_img.active_scalars_name = "RGBA"
        return pv_img

    def render_hud_card():
        """Generate the complete high-resolution HUD card matching the preview image exactly."""
        img = Image.new("RGBA", (HUD_W, HUD_H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Translucent glass card panel
        draw.rectangle(
            [2, 2, HUD_W - 3, HUD_H - 3],
            fill=(15, 20, 30, 235),
            outline=(52, 66, 88, 255),
            width=1
        )

        # Fonts with graceful fallbacks
        try:
            font_title = ImageFont.truetype("arialbd.ttf", 13)
            font_head = ImageFont.truetype("arialbd.ttf", 11)
            font_row = ImageFont.truetype("arial.ttf", 12)
            font_mono = ImageFont.truetype("courbd.ttf", 12)
            font_hint = ImageFont.truetype("arial.ttf", 11)
        except Exception:
            try:
                font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 13)
                font_head = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 11)
                font_row = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12)
                font_mono = font_row
                font_hint = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 11)
            except Exception:
                font_title = font_head = font_row = font_mono = font_hint = ImageFont.load_default()

        # Title
        title = "CHANNELS & THRESHOLDS"
        draw.text((HUD_W // 2 - 85, 14), title, fill="#FFFFFF", font=font_title)
        draw.line([(16, 36), (HUD_W - 16, 36)], fill=(45, 58, 76), width=1)

        # Column Headers
        y_hdr = 42
        draw.text((22, y_hdr), "SW", fill="#7E90A5", font=font_head)
        draw.text((56, y_hdr), "KEY", fill="#7E90A5", font=font_head)
        draw.text((100, y_hdr), "CHANNEL", fill="#7E90A5", font=font_head)
        draw.text((260, y_hdr), "THRESHOLD", fill="#7E90A5", font=font_head)
        draw.text((375, y_hdr), "STATUS", fill="#7E90A5", font=font_head)
        draw.line([(16, 58), (HUD_W - 16, 58)], fill=(45, 58, 76), width=1)

        # Channel Rows
        y_start = HUD_HEADER_H
        btn_sz = 16

        for i in range(num_channels):
            y = y_start + i * HUD_ROW_PITCH
            vis = channel_actors[i].GetVisibility()
            raw_c = state["colors"][i]
            bright_c = boost_color_for_ui(raw_c) if vis else raw_c

            # Solid glowing swatch with crisp 1px white highlight rim
            bx, by = 22, y + 1
            if vis:
                draw.rectangle([bx, by, bx + btn_sz, by + btn_sz], fill=bright_c, outline="#FFFFFF", width=1)
            else:
                draw.rectangle([bx, by, bx + btn_sz, by + btn_sz], fill=(22, 28, 38), outline="#556578", width=1)

            # Labels and values
            txt_c = "#EBF0F5" if vis else "#606E7F"
            k_str = f"[{i + 1}]"
            n_str = channel_names[i] if len(channel_names[i]) <= 14 else channel_names[i][:12] + ".."
            curr_t = state["thresholds"][i]
            t_val = f"{int(round(curr_t)):,}" if is_ints[i] else f"{curr_t:,.1f}"

            draw.text((58, y + 1), k_str, fill=txt_c, font=font_mono)
            draw.text((100, y + 1), n_str, fill=txt_c, font=font_row)
            draw.text((325, y + 1), t_val, fill=txt_c, font=font_row, anchor="ra")

            if vis:
                draw.text((380, y + 1), "ACTIVE", fill="#00FFAA", font=font_head)
            else:
                draw.text((385, y + 1), "OFF", fill="#556475", font=font_head)

        # Footer
        y_foot = y_start + num_channels * HUD_ROW_PITCH + 4
        draw.line([(16, y_foot), (HUD_W - 16, y_foot)], fill=(45, 58, 76), width=1)
        hint = "Toggle: [1]-[9]  |  Export: [E] HTML · [G] GLB · [P] PLY  |  [R] Reset"
        draw.text((20, y_foot + 6), hint, fill="#7E90A5", font=font_hint)

        return img

    # Central visibility toggle synchronizer (keeps 3D viewer, PyVista HUD, and ZEN popup unified)
    def set_channel_visibility(ch_idx, is_vis):
        channel_actors[ch_idx].SetVisibility(bool(is_vis))
        try:
            zen_popup.set_channel_visibility(ch_idx, bool(is_vis))
        except Exception:
            pass
        update_channel_hud()
        plotter.render()

    # Create the initial high-res HUD card
    init_card_img = render_hud_card()
    init_pv_img = pil_to_pyvista_imagedata(init_card_img)

    cur_w, cur_h = plotter.window_size
    pos_x = max(0.01, (cur_w - HUD_W - HUD_MARGIN_RIGHT) / max(1, cur_w))
    pos_y = max(0.01, (cur_h - HUD_H - HUD_MARGIN_TOP) / max(1, cur_h))
    sz_w = min(0.98, HUD_W / max(1, cur_w))
    sz_h = min(0.98, HUD_H / max(1, cur_h))

    hud_widget = plotter.add_logo_widget(
        init_pv_img,
        position=(pos_x, pos_y),
        size=(sz_w, sz_h),
        opacity=1.0
    )
    # Turn off default logo event processing so mouse rotates 3D scene smoothly
    try:
        hud_widget.ProcessEventsOff()
    except Exception:
        pass

    def update_channel_hud():
        try:
            new_img = render_hud_card()
            new_pv_img = pil_to_pyvista_imagedata(new_img)
            rep = hud_widget.GetRepresentation()
            rep.SetImage(new_pv_img)
            rep.Modified()
        except Exception as e:
            pass

    # Helper: recompute point clouds for all channels based on active thresholds and slice range
    def update_all_clouds():
        for ch_idx in range(len(clouds)):
            pts, ints = create_point_cloud(
                processed_volumes[ch_idx],
                state["thresholds"][ch_idx],
                voxel_x,
                voxel_y,
                voxel_z,
                z_min_slice=1,
                z_max_slice=state["z_max"],
                filter_speckles=state.get("filter_speckles", False),
                speckle_min_size=state.get("speckle_min", 1),
                speckle_max_size=state.get("speckle_max", 5)
            )
            if len(pts) > 0:
                new_cloud = pv.PolyData(pts)
                new_cloud["intensity"] = ints
                clouds[ch_idx].shallow_copy(new_cloud)
                vmin = float(state["thresholds"][ch_idx])
                vmax = float(processed_volumes[ch_idx].max()) if processed_volumes[ch_idx].size > 0 else vmin + 1.0
                if vmax <= vmin:
                    vmax = vmin + 1.0
                try:
                    channel_actors[ch_idx].mapper.SetScalarRange(vmin, vmax)
                except Exception:
                    pass
            else:
                clouds[ch_idx].shallow_copy(pv.PolyData())

        update_channel_hud()
        if "update_acquisition_hud" in locals() or "update_acquisition_hud" in globals():
            try:
                update_acquisition_hud()
            except Exception:
                pass
        plotter.reset_camera_clipping_range()
        plotter.render()

    def update_channel_mesh(ch_idx, new_threshold=None):
        if new_threshold is not None:
            state["thresholds"][ch_idx] = new_threshold

        pts, ints = create_point_cloud(
            processed_volumes[ch_idx],
            state["thresholds"][ch_idx],
            voxel_x,
            voxel_y,
            voxel_z,
            z_min_slice=1,
            z_max_slice=state["z_max"],
            filter_speckles=state.get("filter_speckles", False),
            speckle_min_size=state.get("speckle_min", 1),
            speckle_max_size=state.get("speckle_max", 5)
        )

        if len(pts) > 0:
            new_cloud = pv.PolyData(pts)
            new_cloud["intensity"] = ints
            clouds[ch_idx].shallow_copy(new_cloud)
            vmin = float(state["thresholds"][ch_idx])
            vmax = float(processed_volumes[ch_idx].max()) if processed_volumes[ch_idx].size > 0 else vmin + 1.0
            if vmax <= vmin:
                vmax = vmin + 1.0
            try:
                channel_actors[ch_idx].mapper.SetScalarRange(vmin, vmax)
            except Exception:
                pass
        else:
            clouds[ch_idx].shallow_copy(pv.PolyData())

        update_channel_hud()
        plotter.reset_camera_clipping_range()
        plotter.render()

    # Keyboard shortcuts [1], [2], ... to toggle channel visibility cleanly
    def make_channel_toggle(ch_idx):
        return lambda: set_channel_visibility(ch_idx, not channel_actors[ch_idx].GetVisibility())

    for i in range(min(9, num_channels)):
        plotter.add_key_event(str(i + 1), make_channel_toggle(i))

    # Interactive Click Handling: clicking anywhere on a channel row toggles its visibility
    def on_viewport_click(caller, event):
        try:
            iren = plotter.iren.interactor
            x, y = iren.GetEventPosition()
            rw = plotter.render_window
            if not rw:
                return
            w, h = rw.GetSize()

            # Check if click is inside the HUD card
            x_min = w - HUD_W - HUD_MARGIN_RIGHT
            x_max = w - HUD_MARGIN_RIGHT
            y_max = h - HUD_MARGIN_TOP
            y_min = y_max - HUD_H

            if x_min <= x <= x_max and y_min <= y <= y_max:
                from_top = y_max - y
                row_idx = int((from_top - HUD_HEADER_H) // HUD_ROW_PITCH)
                if 0 <= row_idx < num_channels:
                    set_channel_visibility(row_idx, not channel_actors[row_idx].GetVisibility())
                return

            # Check if click is inside the Bottom Filter Button
            fb_x_min = int(((w - TAB_W) // 2) - FILTER_BTN_W - 14)
            fb_x_max = fb_x_min + FILTER_BTN_W
            fb_y_min = TAB_MARGIN_BOTTOM
            fb_y_max = TAB_MARGIN_BOTTOM + FILTER_BTN_H
            if fb_x_min <= x <= fb_x_max and fb_y_min <= y <= fb_y_max:
                if 'toggle_speckle_filter' in locals() or 'toggle_speckle_filter' in globals():
                    toggle_speckle_filter()
                return

            # Check if click is inside the Bottom Pull-Up Tab
            tab_x_min = (w - TAB_W) // 2
            tab_x_max = tab_x_min + TAB_W
            tab_y_min = TAB_MARGIN_BOTTOM
            tab_y_max = TAB_MARGIN_BOTTOM + TAB_H
            if tab_x_min <= x <= tab_x_max and tab_y_min <= y <= tab_y_max:
                if 'zen_popup' in locals() or 'zen_popup' in globals():
                    zen_popup.toggle_visibility()
                return

            # Check if click is inside the Bottom Ruler Button
            rb_x_min = int(((w - TAB_W) // 2) + TAB_W + 14)
            rb_x_max = rb_x_min + RULER_BTN_W
            rb_y_min = TAB_MARGIN_BOTTOM
            rb_y_max = TAB_MARGIN_BOTTOM + RULER_BTN_H
            if rb_x_min <= x <= rb_x_max and rb_y_min <= y <= rb_y_max:
                if 'toggle_ruler_mode' in locals():
                    toggle_ruler_mode()
                return

            # Check if Ruler Measurement is active or Shift+Click was pressed
            is_shift = False
            try:
                is_shift = bool(iren.GetShiftKey())
            except Exception:
                pass

            if state.get("ruler_mode", False) or is_shift:
                if 'handle_ruler_click' in locals():
                    handled = handle_ruler_click(x, y)
                    if handled:
                        try:
                            caller.SetAbortEvent(1)
                        except Exception:
                            pass
                        return
        except Exception:
            pass

    try:
        plotter.iren.add_observer('LeftButtonPressEvent', on_viewport_click)
    except Exception:
        try:
            plotter.interactor.AddObserver('LeftButtonPressEvent', on_viewport_click)
        except Exception:
            pass

    # Interactive Graphics & Navigation Controls
    current_point_size = [3.8]

    def adjust_point_size(delta):
        current_point_size[0] = max(1.0, min(14.0, current_point_size[0] + delta))
        for act in channel_actors:
            try:
                act.GetProperty().SetPointSize(current_point_size[0])
            except Exception:
                pass
        plotter.render()
        print(f"3D Point Size: {current_point_size[0]:.1f} px")

    plotter.add_key_event("plus", lambda: adjust_point_size(+0.4))
    plotter.add_key_event("equal", lambda: adjust_point_size(+0.4))
    plotter.add_key_event("minus", lambda: adjust_point_size(-0.4))
    plotter.add_key_event("bracketright", lambda: adjust_point_size(+0.4))
    plotter.add_key_event("bracketleft", lambda: adjust_point_size(-0.4))

    light_mode = ["studio"]

    def toggle_lighting():
        if light_mode[0] == "studio":
            try:
                plotter.renderer.remove_all_lights()
                plotter.enable_headlights()
                light_mode[0] = "headlight"
                print("Lighting: Headlight mode")
            except Exception:
                pass
        else:
            try:
                plotter.enable_lightkit()
                light_mode[0] = "studio"
                print("Lighting: Studio Light-Kit (3-point depth & specular highlights)")
            except Exception:
                pass
        plotter.render()

    plotter.add_key_event("l", toggle_lighting)
    plotter.add_key_event("L", toggle_lighting)

    def center_on_image():
        cam = plotter.renderer.GetActiveCamera()
        cam_pos = np.array(cam.GetPosition(), dtype=float)
        old_focal = np.array(cam.GetFocalPoint(), dtype=float)
        shift = np.array(image_center) - old_focal
        cam.SetPosition(*(cam_pos + shift))
        cam.SetFocalPoint(*image_center)
        cam.SetWindowCenter(0.0, 0.0)
        plotter.reset_camera_clipping_range()
        plotter.render()
        print("Camera rotation pivot centered on volume center:", image_center)

    def reset_view():
        cam = plotter.renderer.GetActiveCamera()
        cam.SetPosition(full_x * 1.5, -full_y * 1.5, full_z * 2.0)
        cam.SetFocalPoint(*image_center)
        cam.SetViewUp(0, 0, 1)
        cam.SetWindowCenter(0.0, 0.0)
        plotter.reset_camera_clipping_range()
        plotter.render()
        print("Reset camera to default view centered on image.")

    plotter.add_key_event("c", center_on_image)
    plotter.add_key_event("C", center_on_image)
    plotter.add_key_event("r", reset_view)
    plotter.add_key_event("R", reset_view)

    # Window resize handler: keeps HUD cards and pull-up tab locked with exact pixel dimensions
    TL_HUD_W = 440
    TL_HUD_H = 196
    TL_HUD_MARGIN_TOP = 14
    TL_HUD_MARGIN_LEFT = 16
    tl_hud_widget = None

    # Bottom Pull-up Tab for Histogram & Thresholds
    TAB_W = 340
    TAB_H = 34
    TAB_MARGIN_BOTTOM = 12
    tab_widget = None

    # Bottom Background Filter Button (1-5px isolated dots)
    FILTER_BTN_W = 270
    FILTER_BTN_H = 34
    filter_btn_widget = None

    # Bottom 3D Nano-Ruler Button (Nanometer measurement)
    RULER_BTN_W = 240
    RULER_BTN_H = 34
    ruler_btn_widget = None

    def reposition_hud(w, h):
        try:
            if w < 50 or h < 50:
                return
            # Top-Right Channel HUD
            if hud_widget is not None:
                rep_tr = hud_widget.GetRepresentation()
                nx = max(0.001, (w - HUD_W - HUD_MARGIN_RIGHT) / float(w))
                ny = max(0.001, (h - HUD_H - HUD_MARGIN_TOP) / float(h))
                nw = min(0.99, HUD_W / float(w))
                nh = min(0.99, HUD_H / float(h))
                rep_tr.SetPosition(nx, ny)
                rep_tr.SetPosition2(nw, nh)
                rep_tr.Modified()

            # Top-Left Acquisition HUD
            if tl_hud_widget is not None:
                rep_tl = tl_hud_widget.GetRepresentation()
                tl_nx = max(0.001, TL_HUD_MARGIN_LEFT / float(w))
                tl_ny = max(0.001, (h - TL_HUD_H - TL_HUD_MARGIN_TOP) / float(h))
                tl_nw = min(0.99, TL_HUD_W / float(w))
                tl_nh = min(0.99, TL_HUD_H / float(h))
                rep_tl.SetPosition(tl_nx, tl_ny)
                rep_tl.SetPosition2(tl_nw, tl_nh)
                rep_tl.Modified()

            # Bottom Center Pull-up Tab
            if tab_widget is not None:
                rep_tab = tab_widget.GetRepresentation()
                tab_nx = max(0.001, ((w - TAB_W) / 2.0) / float(w))
                tab_ny = max(0.001, float(TAB_MARGIN_BOTTOM) / float(h))
                tab_nw = min(0.99, TAB_W / float(w))
                tab_nh = min(0.99, TAB_H / float(h))
                rep_tab.SetPosition(tab_nx, tab_ny)
                rep_tab.SetPosition2(tab_nw, tab_nh)
                rep_tab.Modified()

            # Bottom Filter Button (1-5px isolated dots)
            if filter_btn_widget is not None:
                rep_fb = filter_btn_widget.GetRepresentation()
                fb_x_pix = ((w - TAB_W) / 2.0) - FILTER_BTN_W - 14
                fb_nx = max(0.001, fb_x_pix / float(w))
                fb_ny = max(0.001, float(TAB_MARGIN_BOTTOM) / float(h))
                fb_nw = min(0.99, FILTER_BTN_W / float(w))
                fb_nh = min(0.99, FILTER_BTN_H / float(h))
                rep_fb.SetPosition(fb_nx, fb_ny)
                rep_fb.SetPosition2(fb_nw, fb_nh)
                rep_fb.Modified()

            # Bottom Ruler Button (Measure in nM)
            if ruler_btn_widget is not None:
                rep_rb = ruler_btn_widget.GetRepresentation()
                rb_x_pix = ((w - TAB_W) / 2.0) + TAB_W + 14
                rb_nx = max(0.001, rb_x_pix / float(w))
                rb_ny = max(0.001, float(TAB_MARGIN_BOTTOM) / float(h))
                rb_nw = min(0.99, RULER_BTN_W / float(w))
                rb_nh = min(0.99, RULER_BTN_H / float(h))
                rep_rb.SetPosition(rb_nx, rb_ny)
                rep_rb.SetPosition2(rb_nw, rb_nh)
                rep_rb.Modified()
        except Exception:
            pass

    last_known_size = [cur_w, cur_h]

    def on_window_configure(caller, event):
        try:
            rw = plotter.render_window
            if rw:
                w, h = rw.GetSize()
                if w > 50 and h > 50:
                    last_known_size[0], last_known_size[1] = w, h
                    reposition_hud(w, h)
                    plotter.render()
        except Exception:
            pass

    try:
        plotter.iren.add_observer('ConfigureEvent', on_window_configure)
    except Exception:
        try:
            plotter.interactor.AddObserver('ConfigureEvent', on_window_configure)
        except Exception:
            pass

    # Callback when user picks a new color in the popup
    def on_channel_color_change(ch_idx, new_color):
        state["colors"][ch_idx] = new_color
        new_cmap = build_fluorescence_cmap(new_color)
        try:
            lut = pv.LookupTable(cmap=new_cmap)
            channel_actors[ch_idx].mapper.SetLookupTable(lut)
        except Exception:
            try:
                rgb = pv.Color(new_color).float_rgb
                channel_actors[ch_idx].GetProperty().SetColor(rgb)
            except Exception:
                pass
        update_channel_hud()
        plotter.render()

    # Launch ZEN & Fiji Histogram, Slider & Intensity Input Popup
    zen_popup = ZenHistogramPopup(
        processed_volumes,
        channel_names,
        state["colors"],
        state["thresholds"],
        on_update_callback=update_channel_mesh,
        on_color_callback=on_channel_color_change,
        on_visibility_callback=set_channel_visibility,
        on_filter_callback=lambda enabled: toggle_speckle_filter(target_state=enabled),
        initial_filter_state=state["filter_speckles"]
    )

    # Periodic event pump to keep matplotlib responsive
    def pump_events(step):
        try:
            rw = plotter.render_window
            if rw:
                w, h = rw.GetSize()
                if [w, h] != last_known_size and w > 50 and h > 50:
                    last_known_size[0], last_known_size[1] = w, h
                    reposition_hud(w, h)
                    plotter.render()
            if plt.fignum_exists(zen_popup.fig.number):
                zen_popup.fig.canvas.flush_events()
        except Exception:
            pass

    plotter.add_timer_event(max_steps=10000000, duration=35, callback=pump_events)

    # Initial display
    update_channel_hud()

    # Z Slice Selection Slider: anchored cleanly in the bottom-right corner
    if num_z > 1:
        slider_y = 0.065
        hint_y = 0.105

        def on_z_end_change(value):
            val = max(1, min(num_z, int(round(value))))
            state["z_max"] = val
            state["z_min"] = 1
            if rep_z is not None:
                try:
                    rep_z.SetTitleText(f"Z End: Slice {state['z_max']} / {num_z}")
                except Exception:
                    pass
            update_all_clouds()

        z_widget = plotter.add_slider_widget(
            callback=on_z_end_change,
            rng=[1, num_z],
            value=state["z_max"],
            title=f"Z End: Slice {state['z_max']} / {num_z}",
            pointa=(0.72, slider_y),
            pointb=(0.97, slider_y),
            style="modern"
        )
        rep_z = z_widget.GetRepresentation()
        rep_z.ShowSliderLabelOff()
        rep_z.SetSliderLength(0.010)
        rep_z.SetSliderWidth(0.014)
        rep_z.SetTubeWidth(0.003)
        rep_z.SetEndCapLength(0.003)
        rep_z.SetEndCapWidth(0.006)
        rep_z.SetTitleHeight(0.016)

        slider_knob_rgb = pv.Color("#c8d6e5").float_rgb
        slider_tube_rgb = pv.Color("#576574").float_rgb
        slider_text_rgb = pv.Color("#c8d6e5").float_rgb

        rep_z.GetSliderProperty().SetColor(slider_knob_rgb)
        rep_z.GetTubeProperty().SetColor(slider_tube_rgb)
        rep_z.GetCapProperty().SetColor(slider_tube_rgb)
        rep_z.GetTitleProperty().SetColor(slider_text_rgb)
        rep_z.GetTitleProperty().ShadowOff()
        rep_z.GetTitleProperty().SetFontFamilyToArial()

        # Keyboard shortcuts for slice stepping
        def step_slice(delta):
            target = max(1, min(num_z, state["z_max"] + delta))
            state["z_max"] = target
            state["z_min"] = 1
            if rep_z is not None:
                try:
                    rep_z.SetValue(target)
                    rep_z.SetTitleText(f"Z End: Slice {target} / {num_z}")
                except Exception:
                    pass
            update_all_clouds()

        def show_all_slices():
            state["z_max"] = num_z
            state["z_min"] = 1
            if rep_z is not None:
                try:
                    rep_z.SetValue(num_z)
                    rep_z.SetTitleText(f"Z End: Slice {num_z} / {num_z}")
                except Exception:
                    pass
            update_all_clouds()

        plotter.add_key_event("Up", lambda: step_slice(1))
        plotter.add_key_event("Down", lambda: step_slice(-1))
        plotter.add_key_event("w", lambda: step_slice(1))
        plotter.add_key_event("W", lambda: step_slice(1))
        plotter.add_key_event("s", lambda: step_slice(-1))
        plotter.add_key_event("S", lambda: step_slice(-1))
        plotter.add_key_event("a", show_all_slices)

    
        # Determine checkerboard floor tile resolution (calibrated to round um step)
    target_cells = 8
    raw_step = max(full_x, full_y) / target_cells if max(full_x, full_y) > 0 else 10.0
    candidates = [1.0, 2.0, 5.0, 10.0, 20.0, 25.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    step_um = min(candidates, key=lambda c: abs(c - raw_step))
    nx = max(2, int(round(full_x / step_um))) if step_um > 0 else 8
    ny = max(2, int(round(full_y / step_um))) if step_um > 0 else 8
    cell_dx = full_x / nx
    cell_dy = full_y / ny

    # Base checkerboard floor plane at Z=0 (offset slightly down to avoid z-fighting with slice 0)
    z_floor = -0.002 * (full_z if full_z > 0 else 1.0)
    checker_plane = pv.Plane(
        center=(full_x / 2.0, full_y / 2.0, z_floor),
        direction=(0, 0, 1),
        i_size=full_x,
        j_size=full_y,
        i_resolution=nx,
        j_resolution=ny
    )
    # Checkerboard scalar field (alternating 0 and 1)
    checker_pattern = (np.indices((ny, nx)).sum(axis=0) % 2).flatten()
    if len(checker_pattern) != checker_plane.n_cells:
        checker_pattern = np.resize(checker_pattern, checker_plane.n_cells)
    checker_plane.cell_data["checker"] = checker_pattern

    checker_actor = plotter.add_mesh(
        checker_plane,
        scalars="checker",
        cmap=["#0d1117", "#161b22"],
        opacity=0.38,
        show_edges=True,
        edge_color="#2b3648",
        line_width=1.0,
        show_scalar_bar=False,
        name="checker_floor"
    )

    # Architectural bounding frame: crisp corner uprights and outer perimeter frames
    pts = []
    lines_arr = []
    corners_xy = [(0.0, 0.0), (full_x, 0.0), (full_x, full_y), (0.0, full_y)]

    if num_z > 1:
        for cx, cy in corners_xy:
            idx0 = len(pts)
            pts.append([cx, cy, 0.0])
            pts.append([cx, cy, full_z])
            lines_arr.extend([2, idx0, idx0 + 1])

    for z_level in ([0.0, full_z] if num_z > 1 else [0.0]):
        base_idx = len(pts)
        for cx, cy in corners_xy:
            pts.append([cx, cy, z_level])
        lines_arr.extend([
            2, base_idx + 0, base_idx + 1,
            2, base_idx + 1, base_idx + 2,
            2, base_idx + 2, base_idx + 3,
            2, base_idx + 3, base_idx + 0,
        ])

    slice_cage = pv.PolyData()
    slice_cage.points = np.array(pts, dtype=np.float32)
    slice_cage.lines = np.array(lines_arr, dtype=np.int64)

    slice_cage_actor = plotter.add_mesh(
        slice_cage,
        color="#4b5d73",
        opacity=0.45,
        line_width=1.2,
        name="volume_slice_cage"
    )

    # Keyboard toggle [G] for slice cage and checkerboard floor
    def toggle_grid():
        vis = None
        if slice_cage_actor is not None:
            vis = not slice_cage_actor.GetVisibility()
            slice_cage_actor.SetVisibility(vis)
        if checker_actor is not None:
            new_vis = (not checker_actor.GetVisibility()) if vis is None else vis
            checker_actor.SetVisibility(new_vis)
        plotter.render()

    plotter.add_key_event("g", toggle_grid)
    plotter.add_key_event("G", toggle_grid)

    # Physical axis scales showing Length (X), Width (Y), and Height (Z) in um
    n_z_ticks = min(5, max(2, num_z))
    plotter.show_bounds(
        axes_ranges=[0, full_x, 0, full_y, 0, full_z],
        all_edges=False,
        show_xaxis=True,
        show_yaxis=True,
        show_zaxis=True,
        xtitle="Length X (µm)",
        ytitle="Width Y (µm)",
        ztitle="Height Z (µm)",
        color="#999999",
        font_size=9,
        font_family="arial",
        bold=False,
        grid=None,
        ticks="outside",
        location="outer",
        n_xlabels=3,
        n_ylabels=3,
        n_zlabels=n_z_ticks,
        use_2d=True
    )

    # Orientation axes gizmo in lower-left
    plotter.add_axes()

    # =========================================================================
    # TOP-LEFT HUD: HIGH-RESOLUTION ACQUISITION & SCALE CARD OVERLAY
    # =========================================================================

    def render_acquisition_hud_card():
        """Generate the complete high-resolution acquisition HUD card matching the top-right card."""
        img = Image.new("RGBA", (TL_HUD_W, TL_HUD_H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Translucent glass panel matching top-right card exactly
        draw.rectangle(
            [2, 2, TL_HUD_W - 3, TL_HUD_H - 3],
            fill=(15, 20, 30, 235),
            outline=(52, 66, 88, 255),
            width=1
        )

        try:
            font_title = ImageFont.truetype("arialbd.ttf", 13)
            font_label = ImageFont.truetype("arialbd.ttf", 11)
            font_val = ImageFont.truetype("arial.ttf", 12)
            font_badge = ImageFont.truetype("arialbd.ttf", 10)
        except Exception:
            try:
                font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 13)
                font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 11)
                font_val = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12)
                font_badge = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 10)
            except Exception:
                font_title = font_label = font_val = font_badge = ImageFont.load_default()

        # Header Title
        title = "ACQUISITION & SCALE"
        draw.text((20, 14), title, fill="#FFFFFF", font=font_title)
        draw.line([(16, 36), (TL_HUD_W - 16, 36)], fill=(45, 58, 76), width=1)

        # Grid measurements
        y_start = 46
        row_pitch = 27
        colon_x = 142
        val_x = 154

        voxel_z_scaled = voxel_z * Z_SCALE
        
        # Structure summary with active Z plane tracking
        curr_z = state.get("z_max", num_z)
        if curr_z < num_z:
            struct_val = f"{num_z} Z-planes (1–{curr_z} active)  ·  {num_channels} Channels"
        else:
            struct_val = f"{num_z} Z-planes   ·   {num_channels} Channels"

        filter_status = "Active (1-5 px dots filtered)" if state.get("filter_speckles") else "Off (raw threshold)"
        meas_list = state.get("ruler_measurements", [])
        pts_list = state.get("ruler_points", [])
        if meas_list:
            last_m = meas_list[-1]
            ruler_val = f"{last_m['dist_3d_nm']:,.1f} nm ({last_m['dist_3d_um']:.3f} µm)"
            ruler_badge = f"#{last_m['id']}"
        elif state.get("ruler_mode", False):
            if len(pts_list) > 0:
                ruler_val = "Point A set · Click Point B"
                ruler_badge = "PT B"
            else:
                ruler_val = "Ready · Click specimen point"
                ruler_badge = "RULER"
        else:
            ruler_val = "Press [M] or Shift+Click to measure"
            ruler_badge = "nm"

        rows = [
            ("Dimensions (XYZ)", f"{full_x:.1f} × {full_y:.1f} × {full_z:.1f} µm", None),
            ("Voxel Resolution", f"{voxel_x:.3f} × {voxel_y:.3f} × {voxel_z_scaled:.3f} µm", f"Z {Z_SCALE:.1f}×"),
            ("Noise Filter", filter_status, "1-5 px" if state.get("filter_speckles") else None),
            ("Volume Structure", struct_val, None),
            ("Nano-Ruler (nM)", ruler_val, ruler_badge)
        ]

        for i, (label, val, badge) in enumerate(rows):
            y = y_start + i * row_pitch

            # Field Label
            draw.text((20, y), label, fill="#7E90A5", font=font_label)
            draw.text((colon_x, y), ":", fill="#556578", font=font_label)

            # Field Value
            draw.text((val_x, y), val, fill="#EBF0F5", font=font_val)

            # Optional accent badge
            if badge:
                tw = draw.textlength(val, font=font_val) if hasattr(draw, "textlength") else len(val) * 7.5
                bx = int(val_x + tw + 10)
                by = y - 1
                bw = 48
                bh = 16
                draw.rounded_rectangle([bx, by, bx + bw, by + bh], radius=3, fill=(24, 38, 56), outline=(0, 180, 216), width=1)
                draw.text((bx + 6, by + 1), badge, fill="#00D4FF", font=font_badge)

        return img

    tl_card_img = render_acquisition_hud_card()
    tl_pv_img = pil_to_pyvista_imagedata(tl_card_img)

    cur_w, cur_h = plotter.window_size
    tl_pos_x = max(0.001, TL_HUD_MARGIN_LEFT / float(max(1, cur_w)))
    tl_pos_y = max(0.001, (cur_h - TL_HUD_H - TL_HUD_MARGIN_TOP) / float(max(1, cur_h)))
    tl_sz_w = min(0.98, TL_HUD_W / float(max(1, cur_w)))
    tl_sz_h = min(0.98, TL_HUD_H / float(max(1, cur_h)))

    tl_hud_widget = plotter.add_logo_widget(
        tl_pv_img,
        position=(tl_pos_x, tl_pos_y),
        size=(tl_sz_w, tl_sz_h),
        opacity=1.0
    )
    try:
        tl_hud_widget.ProcessEventsOff()
    except Exception:
        pass

    def update_acquisition_hud():
        try:
            new_img = render_acquisition_hud_card()
            new_pv = pil_to_pyvista_imagedata(new_img)
            rep = tl_hud_widget.GetRepresentation()
            rep.SetImage(new_pv)
            rep.Modified()
        except Exception:
            pass

    # ====================================
    # BOTTOM PULL-UP TAB: HISTOGRAM & THRESHOLDS
    # ====================================
    def render_bottom_tab_card(is_open=True):
        img = Image.new("RGBA", (TAB_W, TAB_H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        bg = (14, 28, 22, 235) if is_open else (16, 22, 32, 230)
        border = (48, 160, 95, 255) if is_open else (60, 85, 115, 255)
        draw.rounded_rectangle([1, 1, TAB_W - 2, TAB_H - 2], radius=6, fill=bg, outline=border, width=1)

        dot_color = (75, 235, 130, 255) if is_open else (60, 200, 255, 255)
        text = "▼  HIDE HISTOGRAM PANEL  [ H ]" if is_open else "▲  HISTOGRAM & THRESHOLDS  [ H ]"

        try:
            font_tab = ImageFont.truetype("arialbd.ttf", 11)
        except Exception:
            try:
                font_tab = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
            except Exception:
                font_tab = ImageFont.load_default()

        bbox = font_tab.getbbox(text)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = (TAB_W - tw) // 2
        ty = (TAB_H - th) // 2

        draw.ellipse([tx - 16, TAB_H // 2 - 3, tx - 10, TAB_H // 2 + 3], fill=dot_color)
        draw.text((tx, ty), text, fill=(240, 246, 253, 255), font=font_tab)
        return img

    tab_card_img = render_bottom_tab_card(is_open=True)
    tab_pv_img = pil_to_pyvista_imagedata(tab_card_img)

    tab_pos_x = max(0.001, ((cur_w - TAB_W) / 2.0) / float(max(1, cur_w)))
    tab_pos_y = max(0.001, float(TAB_MARGIN_BOTTOM) / float(max(1, cur_h)))
    tab_sz_w = min(0.98, TAB_W / float(max(1, cur_w)))
    tab_sz_h = min(0.98, TAB_H / float(max(1, cur_h)))

    tab_widget = plotter.add_logo_widget(
        tab_pv_img,
        position=(tab_pos_x, tab_pos_y),
        size=(tab_sz_w, tab_sz_h),
        opacity=1.0
    )
    try:
        tab_widget.ProcessEventsOff()
    except Exception:
        pass

    def update_tab_widget(is_visible):
        try:
            new_img = render_bottom_tab_card(is_open=is_visible)
            new_pv = pil_to_pyvista_imagedata(new_img)
            rep = tab_widget.GetRepresentation()
            rep.SetImage(new_pv)
            rep.Modified()
            plotter.render()
        except Exception:
            pass

    # Link bidirectional visibility updates
    zen_popup.on_visibility_listener = update_tab_widget

    # ====================================
    # BOTTOM BUTTON: BACKGROUND NOISE FILTER (1-5 px DOTS)
    # ====================================
    def render_filter_button_card(is_active=False):
        img = Image.new("RGBA", (FILTER_BTN_W, FILTER_BTN_H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        bg = (14, 38, 26, 240) if is_active else (18, 24, 34, 230)
        border = (52, 199, 116, 255) if is_active else (60, 85, 115, 255)
        draw.rounded_rectangle([1, 1, FILTER_BTN_W - 2, FILTER_BTN_H - 2], radius=6, fill=bg, outline=border, width=1)

        dot_color = (75, 245, 140, 255) if is_active else (120, 140, 160, 255)
        text = "●  FILTER 1-5px DOTS: ON  [ D ]" if is_active else "○  FILTER 1-5px DOTS: OFF  [ D ]"

        try:
            font_btn = ImageFont.truetype("arialbd.ttf", 11)
        except Exception:
            try:
                font_btn = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
            except Exception:
                font_btn = ImageFont.load_default()

        bbox = font_btn.getbbox(text)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = (FILTER_BTN_W - tw) // 2
        ty = (FILTER_BTN_H - th) // 2

        draw.ellipse([tx - 16, FILTER_BTN_H // 2 - 3, tx - 10, FILTER_BTN_H // 2 + 3], fill=dot_color)
        text_color = (240, 255, 245, 255) if is_active else (180, 195, 215, 255)
        draw.text((tx, ty), text, fill=text_color, font=font_btn)
        return img

    filter_card_img = render_filter_button_card(is_active=state.get("filter_speckles", False))
    filter_pv_img = pil_to_pyvista_imagedata(filter_card_img)

    fb_init_x = max(0.001, (((cur_w - TAB_W) / 2.0) - FILTER_BTN_W - 14) / float(max(1, cur_w)))
    fb_init_y = max(0.001, float(TAB_MARGIN_BOTTOM) / float(max(1, cur_h)))
    fb_init_w = min(0.98, FILTER_BTN_W / float(max(1, cur_w)))
    fb_init_h = min(0.98, FILTER_BTN_H / float(max(1, cur_h)))

    filter_btn_widget = plotter.add_logo_widget(
        filter_pv_img,
        position=(fb_init_x, fb_init_y),
        size=(fb_init_w, fb_init_h),
        opacity=1.0
    )
    try:
        filter_btn_widget.ProcessEventsOff()
    except Exception:
        pass

    def update_filter_widget(is_active):
        try:
            new_img = render_filter_button_card(is_active=is_active)
            new_pv = pil_to_pyvista_imagedata(new_img)
            rep = filter_btn_widget.GetRepresentation()
            rep.SetImage(new_pv)
            rep.Modified()
            plotter.render()
        except Exception:
            pass

    # =========================================================================
    # 3D NANO-RULER: PHYSICAL NANOMETER MEASUREMENT ENGINE (nM)
    # =========================================================================
    vol_diag = float(np.sqrt(full_x**2 + full_y**2 + full_z**2)) if (full_x > 0) else 100.0
    ruler_tube_radius = max(0.04, vol_diag * 0.0028)
    ruler_marker_radius = ruler_tube_radius * 2.2

    def pick_specimen_3d_point(click_x, click_y):
        """
        Picks an exact 3D coordinate on the visible specimen at screen (x, y).
        Prioritizes snapping to active point cloud vertices via vtkPointPicker,
        with fallback to vtkCellPicker and vtkPropPicker.
        Returns numpy array [x_um, y_um, z_scene] or None.
        """
        ren = plotter.renderer
        if not ren:
            return None

        # 1. Point picker (snaps directly to vertex of fluorescent point cloud)
        pt_picker = vtk.vtkPointPicker()
        pt_picker.SetTolerance(0.025)
        if pt_picker.Pick(click_x, click_y, 0, ren):
            pt_id = pt_picker.GetPointId()
            ds = pt_picker.GetDataSet()
            if pt_id >= 0 and ds is not None:
                try:
                    return np.array(ds.GetPoint(pt_id), dtype=float)
                except Exception:
                    pass
            pos = pt_picker.GetPickPosition()
            if pos and not np.allclose(pos, 0):
                return np.array(pos, dtype=float)

        # 2. Cell picker fallback
        cell_picker = vtk.vtkCellPicker()
        cell_picker.SetTolerance(0.02)
        if cell_picker.Pick(click_x, click_y, 0, ren):
            pos = cell_picker.GetPickPosition()
            if pos and not np.allclose(pos, 0):
                return np.array(pos, dtype=float)

        # 3. Prop picker fallback (ray-cast surface intersection)
        prop_picker = vtk.vtkPropPicker()
        if prop_picker.Pick(click_x, click_y, 0, ren):
            pos = prop_picker.GetPickPosition()
            if pos and not np.allclose(pos, 0):
                return np.array(pos, dtype=float)

        return None

    def render_ruler_button_card():
        img = Image.new("RGBA", (RULER_BTN_W, RULER_BTN_H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        is_active = state.get("ruler_mode", False)
        pts_list = state.get("ruler_points", [])
        meas_list = state.get("ruler_measurements", [])
        pending = len(pts_list) > 0
        has_meas = len(meas_list) > 0

        if is_active:
            if pending:
                bg = (42, 32, 10, 245)
                border = (255, 204, 0, 255)
                dot_color = (255, 220, 50, 255)
                text = "●  RULER: CLICK PT B  [ M ]"
                text_color = (255, 248, 220, 255)
            else:
                bg = (10, 34, 48, 245)
                border = (0, 210, 255, 255)
                dot_color = (0, 235, 255, 255)
                text = "●  RULER: CLICK PT A  [ M ]"
                text_color = (220, 250, 255, 255)
        else:
            if has_meas:
                bg = (14, 26, 36, 235)
                border = (0, 160, 220, 255)
                dot_color = (0, 200, 255, 255)
                text = f"📏  {len(meas_list)} RULER ({meas_list[-1]['dist_3d_nm']:,.0f} nm)"
                text_color = (210, 240, 255, 255)
            else:
                bg = (18, 24, 34, 230)
                border = (60, 85, 115, 255)
                dot_color = (120, 140, 160, 255)
                text = "○  RULER (nM): OFF  [ M ]"
                text_color = (180, 195, 215, 255)

        draw.rounded_rectangle([1, 1, RULER_BTN_W - 2, RULER_BTN_H - 2], radius=6, fill=bg, outline=border, width=1)

        try:
            font_btn = ImageFont.truetype("arialbd.ttf", 11)
        except Exception:
            try:
                font_btn = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
            except Exception:
                font_btn = ImageFont.load_default()

        bbox = font_btn.getbbox(text)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = (RULER_BTN_W - tw) // 2
        ty = (RULER_BTN_H - th) // 2

        if not text.startswith("📏"):
            draw.ellipse([tx - 16, RULER_BTN_H // 2 - 3, tx - 10, RULER_BTN_H // 2 + 3], fill=dot_color)
        draw.text((tx, ty), text, fill=text_color, font=font_btn)
        return img

    ruler_card_img = render_ruler_button_card()
    ruler_pv_img = pil_to_pyvista_imagedata(ruler_card_img)

    rb_init_x = max(0.001, (((cur_w - TAB_W) / 2.0) + TAB_W + 14) / float(max(1, cur_w)))
    rb_init_y = max(0.001, float(TAB_MARGIN_BOTTOM) / float(max(1, cur_h)))
    rb_init_w = min(0.98, RULER_BTN_W / float(max(1, cur_w)))
    rb_init_h = min(0.98, RULER_BTN_H / float(max(1, cur_h)))

    ruler_btn_widget = plotter.add_logo_widget(
        ruler_pv_img,
        position=(rb_init_x, rb_init_y),
        size=(rb_init_w, rb_init_h),
        opacity=1.0
    )
    try:
        ruler_btn_widget.ProcessEventsOff()
    except Exception:
        pass

    def update_ruler_widget():
        try:
            if ruler_btn_widget is not None:
                new_img = render_ruler_button_card()
                new_pv = pil_to_pyvista_imagedata(new_img)
                rep = ruler_btn_widget.GetRepresentation()
                rep.SetImage(new_pv)
                rep.Modified()
        except Exception:
            pass

    def handle_ruler_click(click_x, click_y):
        """Processes a click when ruler mode is active or Shift+Click is held."""
        picked_pt = pick_specimen_3d_point(click_x, click_y)
        if picked_pt is None:
            print("[Ruler Notice] Click missed specimen points. Click directly on fluorescence signal.")
            return False

        pts_list = state.setdefault("ruler_points", [])
        meas_list = state.setdefault("ruler_measurements", [])

        if len(pts_list) == 0:
            # Set Point A
            pts_list.append(picked_pt)
            sphere_a = pv.Sphere(radius=ruler_marker_radius * 1.25, center=picked_pt)
            state["ruler_active_marker"] = plotter.add_mesh(
                sphere_a,
                color="#FFD700",
                render_points_as_spheres=True,
                name="ruler_pending_marker"
            )
            update_ruler_widget()
            if "update_acquisition_hud" in locals():
                update_acquisition_hud()
            plotter.render()
            pt_a_um = (picked_pt[0], picked_pt[1], picked_pt[2] / float(Z_SCALE))
            print(f"\n[Ruler] Point A set at: X={pt_a_um[0]:.3f} µm, Y={pt_a_um[1]:.3f} µm, Z={pt_a_um[2]:.3f} µm")
            print("  ==> Click Point B on the specimen to complete nanometer measurement.")
            return True

        elif len(pts_list) == 1:
            # Set Point B and complete measurement
            pt_a = pts_list.pop(0)
            pt_b = picked_pt

            # Clear temporary marker
            if state.get("ruler_active_marker") is not None:
                try:
                    plotter.remove_actor("ruler_pending_marker")
                except Exception:
                    pass
                state["ruler_active_marker"] = None

            # Physical coordinates in nanometers (nm):
            # 1 µm = 1000 nm.
            # Z is divided by Z_SCALE to reverse visual Z stretching and recover true physical Z.
            dx_nm = (pt_b[0] - pt_a[0]) * 1000.0
            dy_nm = (pt_b[1] - pt_a[1]) * 1000.0
            dz_nm = ((pt_b[2] - pt_a[2]) / float(Z_SCALE)) * 1000.0

            dist_3d_nm = float(np.sqrt(dx_nm**2 + dy_nm**2 + dz_nm**2))
            dist_xy_nm = float(np.sqrt(dx_nm**2 + dy_nm**2))
            dist_z_nm = float(abs(dz_nm))
            dist_3d_um = dist_3d_nm / 1000.0

            meas_idx = len(meas_list) + 1

            # 3D Visual Artifacts: Connecting Tube, Endpoint Spheres, and Floating Label
            line_mesh = pv.Line(pt_a, pt_b)
            tube_mesh = line_mesh.tube(radius=ruler_tube_radius)
            sphere_a_mesh = pv.Sphere(radius=ruler_marker_radius, center=pt_a)
            sphere_b_mesh = pv.Sphere(radius=ruler_marker_radius, center=pt_b)

            act_a_name = f"ruler_sph_a_{meas_idx}"
            act_b_name = f"ruler_sph_b_{meas_idx}"
            act_tube_name = f"ruler_tube_{meas_idx}"
            act_lbl_name = f"ruler_lbl_{meas_idx}"

            plotter.add_mesh(sphere_a_mesh, color="#FFD700", name=act_a_name, render_points_as_spheres=True)
            plotter.add_mesh(sphere_b_mesh, color="#00E5FF", name=act_b_name, render_points_as_spheres=True)
            plotter.add_mesh(tube_mesh, color="#00E5FF", opacity=0.92, name=act_tube_name)

            mid_point = (pt_a + pt_b) / 2.0
            if dist_3d_nm >= 1000.0:
                lbl_text = f"#{meas_idx}: {dist_3d_nm:,.1f} nm ({dist_3d_um:.3f} µm)"
            else:
                lbl_text = f"#{meas_idx}: {dist_3d_nm:,.1f} nm"

            try:
                plotter.add_point_labels(
                    [mid_point],
                    [lbl_text],
                    font_size=11,
                    text_color="#FFFFFF",
                    shape_color="#0d1520",
                    shape_opacity=0.88,
                    always_visible=True,
                    name=act_lbl_name
                )
            except Exception:
                try:
                    plotter.add_point_labels([mid_point], [lbl_text], font_size=11, name=act_lbl_name)
                except Exception:
                    pass

            record = {
                "id": meas_idx,
                "pt_a": pt_a,
                "pt_b": pt_b,
                "dist_3d_nm": dist_3d_nm,
                "dist_3d_um": dist_3d_um,
                "dist_xy_nm": dist_xy_nm,
                "dist_z_nm": dist_z_nm,
                "actors": [act_a_name, act_b_name, act_tube_name, act_lbl_name]
            }
            meas_list.append(record)

            pt_a_um = (pt_a[0], pt_a[1], pt_a[2] / float(Z_SCALE))
            pt_b_um = (pt_b[0], pt_b[1], pt_b[2] / float(Z_SCALE))

            print("\n" + "=" * 58)
            print(f" [3D NANO-RULER MEASUREMENT #{meas_idx}]")
            print(f"  Direct 3D Distance:  {dist_3d_nm:,.1f} nm  ({dist_3d_um:.4f} µm)")
            print(f"  Lateral (XY):        {dist_xy_nm:,.1f} nm  ({dist_xy_nm/1000.0:.4f} µm)")
            print(f"  Axial (ΔZ):          {dist_z_nm:,.1f} nm  ({dist_z_nm/1000.0:.4f} µm)")
            print(f"  Point A (Physical):  X={pt_a_um[0]:.3f} µm, Y={pt_a_um[1]:.3f} µm, Z={pt_a_um[2]:.3f} µm")
            print(f"  Point B (Physical):  X={pt_b_um[0]:.3f} µm, Y={pt_b_um[1]:.3f} µm, Z={pt_b_um[2]:.3f} µm")
            print("  ==> Hotkeys: [C] Clear All · [Z] Undo Last · [M] Toggle Ruler")
            print("=" * 58 + "\n")

            update_ruler_widget()
            if "update_acquisition_hud" in locals():
                update_acquisition_hud()
            plotter.render()
            return True

        return False

    def toggle_ruler_mode(target_state=None):
        if target_state is None:
            state["ruler_mode"] = not state.get("ruler_mode", False)
        else:
            state["ruler_mode"] = bool(target_state)

        pts_list = state.get("ruler_points", [])
        if not state["ruler_mode"]:
            # Leaving ruler mode
            if len(pts_list) > 0:
                pts_list.clear()
                if state.get("ruler_active_marker") is not None:
                    try:
                        plotter.remove_actor("ruler_pending_marker")
                    except Exception:
                        pass
                    state["ruler_active_marker"] = None
            print("\n[Ruler] Mode OFF. Standard 3D orbit controls active. Measurements stay visible.")
        else:
            print("\n[Ruler] Mode ON. Click any point on the specimen to set Point A, then click Point B.")
            print("  ==> Physical distance calculated in nm (nanometers).")
            print("  ==> Hotkeys: [M] Toggle Mode · [C] Clear All · [Z] Undo · Shift+Click Instant Measure")

        update_ruler_widget()
        if "update_acquisition_hud" in locals():
            update_acquisition_hud()
        plotter.render()

    def clear_all_measurements():
        if state.get("ruler_active_marker") is not None:
            try:
                plotter.remove_actor("ruler_pending_marker")
            except Exception:
                pass
            state["ruler_active_marker"] = None

        state.get("ruler_points", []).clear()

        meas_list = state.get("ruler_measurements", [])
        for m in meas_list:
            for act_name in m["actors"]:
                try:
                    plotter.remove_actor(act_name)
                except Exception:
                    pass
        meas_list.clear()
        update_ruler_widget()
        if "update_acquisition_hud" in locals():
            update_acquisition_hud()
        plotter.render()
        print("\n[Ruler] Cleared all measurements and 3D ruler lines.")

    def undo_last_measurement():
        pts_list = state.get("ruler_points", [])
        if len(pts_list) > 0:
            pts_list.clear()
            if state.get("ruler_active_marker") is not None:
                try:
                    plotter.remove_actor("ruler_pending_marker")
                except Exception:
                    pass
                state["ruler_active_marker"] = None
            update_ruler_widget()
            if "update_acquisition_hud" in locals():
                update_acquisition_hud()
            plotter.render()
            print("\n[Ruler] Cancelled in-progress Point A selection.")
            return

        meas_list = state.get("ruler_measurements", [])
        if meas_list:
            m = meas_list.pop()
            for act_name in m["actors"]:
                try:
                    plotter.remove_actor(act_name)
                except Exception:
                    pass
            update_ruler_widget()
            if "update_acquisition_hud" in locals():
                update_acquisition_hud()
            plotter.render()
            print(f"\n[Ruler] Removed measurement #{m['id']} ({m['dist_3d_nm']:,.1f} nm).")

    plotter.add_key_event("m", toggle_ruler_mode)
    plotter.add_key_event("M", toggle_ruler_mode)
    plotter.add_key_event("c", clear_all_measurements)
    plotter.add_key_event("C", clear_all_measurements)
    plotter.add_key_event("z", undo_last_measurement)
    plotter.add_key_event("Z", undo_last_measurement)
    plotter.add_key_event("BackSpace", undo_last_measurement)

    def toggle_speckle_filter(target_state=None):
        if target_state is None:
            state["filter_speckles"] = not state.get("filter_speckles", False)
        else:
            state["filter_speckles"] = bool(target_state)

        is_active = state["filter_speckles"]
        status_str = "ENABLED (removes isolated 1-5px dots)" if is_active else "DISABLED"
        print(f"\n[Noise Filter] Isolated dot filter {status_str}")

        update_filter_widget(is_active)
        if zen_popup is not None:
            zen_popup.set_filter_state(is_active)

        update_all_clouds()
        if "update_acquisition_hud" in locals() or "update_acquisition_hud" in globals():
            try:
                update_acquisition_hud()
            except Exception:
                pass
        plotter.render()

    def refocus_3d_viewer():
        try:
            rw = plotter.render_window
            if rw and sys.platform == "win32":
                import ctypes
                hwnd = rw.GetGenericWindowId()
                if hwnd:
                    ctypes.windll.user32.SetForegroundWindow(hwnd)
        except Exception:
            pass
        try:
            plotter.render()
        except Exception:
            pass

    zen_popup.refocus_viewer_callback = refocus_3d_viewer

    # Register H shortcut in 3D viewer
    plotter.add_key_event("h", lambda: zen_popup.toggle_visibility())
    plotter.add_key_event("H", lambda: zen_popup.toggle_visibility())
    plotter.add_key_event("d", lambda: toggle_speckle_filter())
    plotter.add_key_event("D", lambda: toggle_speckle_filter())

    # Smart initial window positioning on Windows
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes
            user32 = ctypes.windll.user32
            rect = wintypes.RECT()
            user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0)
            sw = rect.right - rect.left
            sh = rect.bottom - rect.top
            hwnd_hist = zen_popup._get_hwnd()
            if hwnd_hist:
                hist_w = min(1120, int(sw * 0.65))
                hist_h = min(540, int(sh * 0.45))
                hist_x = rect.left + (sw - hist_w) // 2
                hist_y = rect.bottom - hist_h - 12
                user32.SetWindowPos(hwnd_hist, 0, hist_x, hist_y, hist_w, hist_h, 0x0004 | 0x0040)
        except Exception:
            pass

    # ====================================
    # ====================================

        # =========================================================================
    # FILE EXPORT & KEYBOARD SHORTCUTS
    # =========================================================================

    base_name = "czi_specimen"
    if source_filename:
        base_name = os.path.splitext(os.path.basename(source_filename))[0]

    def do_export_html(filepath=None):
        out = filepath or f"{base_name}_3d_viewer.html"
        print(f"\n--> Exporting standalone interactive 3D Web Viewer to: {out} ...")
        try:
            plotter.export_html(out)
            abs_p = os.path.abspath(out)
            print(f"[SUCCESS] Standalone HTML saved: {abs_p}")
            print("  ==> You can double-click this file to view & orbit smoothly in any web browser!\n")
        except Exception as e:
            print(f"[Export Notice] PyVista export_html returned: {e}")
            print("  Attempting VTK WebGL fallback...")
            try:
                combined = build_combined_polydata(clouds, channel_actors, channel_colors, processed_volumes, state)
                if combined:
                    combined.save(f"{base_name}_points.ply", binary=True)
                    print(f"[SUCCESS] Exported high-fidelity 3D point cloud: {base_name}_points.ply")
            except Exception as e2:
                print(f"[Export Error]: {e2}")

    def do_export_glb(filepath=None):
        out = filepath or f"{base_name}_model.glb"
        print(f"\n--> Exporting 3D Model to: {out} ...")
        try:
            combined = build_combined_polydata(clouds, channel_actors, channel_colors, processed_volumes, state)
            if combined and combined.n_points > 0:
                try:
                    combined.save(out)
                    print(f"[SUCCESS] Saved 3D model: {os.path.abspath(out)}")
                except Exception:
                    ply_out = f"{base_name}_model.ply"
                    combined.save(ply_out, binary=True)
                    print(f"[SUCCESS] Saved standard colored binary PLY: {os.path.abspath(ply_out)}")
                    print("  ==> Opens in Windows 3D Viewer, Blender, MeshLab, CloudCompare, or ImageJ!\n")
            else:
                print("[Export Warning] No active points found.")
        except Exception as e:
            print(f"[Export Error]: {e}")

    def do_export_ply(filepath=None):
        out = filepath or f"{base_name}_pointcloud.ply"
        print(f"\n--> Exporting scientific colored point cloud to: {out} ...")
        try:
            combined = build_combined_polydata(clouds, channel_actors, channel_colors, processed_volumes, state)
            if combined and combined.n_points > 0:
                combined.save(out, binary=True)
                print(f"[SUCCESS] Saved PLY point cloud: {os.path.abspath(out)}")
                print("  ==> True physical microns + RGB fluorescence colors included.\n")
            else:
                print("[Export Warning] No active points found.")
        except Exception as e:
            print(f"[Export Error]: {e}")

    def do_save_snapshot(filepath=None):
        out = filepath or f"{base_name}_snapshot.png"
        try:
            plotter.screenshot(out, return_img=False)
            print(f"\n[SUCCESS] Saved high-resolution snapshot: {os.path.abspath(out)}")
        except Exception as e:
            print(f"[Snapshot Error]: {e}")

    def reset_camera_view():
        plotter.camera_position = [
            (full_x * 1.5, -full_y * 1.5, full_z * 2),
            image_center,
            (0, 0, 1)
        ]
        plotter.camera.focal_point = image_center
        try:
            plotter.renderer.GetActiveCamera().SetWindowCenter(0.0, 0.0)
        except Exception:
            pass
        plotter.reset_camera_clipping_range()
        plotter.render()
        print("\n[Camera] Reset view to centered orientation.")

    # Hotkey registrations
    plotter.add_key_event("e", lambda: do_export_html())
    plotter.add_key_event("E", lambda: do_export_html())
    plotter.add_key_event("g", lambda: do_export_glb())
    plotter.add_key_event("G", lambda: do_export_glb())
    plotter.add_key_event("p", lambda: do_export_ply())
    plotter.add_key_event("P", lambda: do_export_ply())
    plotter.add_key_event("s", lambda: do_save_snapshot())
    plotter.add_key_event("S", lambda: do_save_snapshot())
    plotter.add_key_event("r", reset_camera_view)
    plotter.add_key_event("R", reset_camera_view)

    # Automatic headless export if requested
    if auto_export:
        if auto_export.get("html"):
            do_export_html(auto_export["html"])
        if auto_export.get("glb"):
            do_export_glb(auto_export["glb"])
        if auto_export.get("ply"):
            do_export_ply(auto_export["ply"])
        if auto_export.get("exit_after"):
            print("[Batch Export Complete] Exiting without opening GUI.")
            plotter.close()
            return

    # Use the geometric center of the rendered image volume as the permanent
    # camera orbit pivot, shared by the initial view and cursor-anchored wheel zoom.
    image_center = (full_x / 2.0, full_y / 2.0, full_z / 2.0)
    plotter.camera_position = [
        (full_x * 1.5, -full_y * 1.5, full_z * 2),
        image_center,
        (0, 0, 1)
    ]
    plotter.reset_camera_clipping_range()

    # Mouse-wheel zoom stays centered on the image, even when zooming at the cursor.
    install_cursor_zoom(plotter, focal_point=image_center)

    plotter.show()

# ====================================
# MAIN
# ====================================

def main():
    parser = argparse.ArgumentParser(
        description="Fast High-Fidelity 3D CZI Multi-Channel Microscopy Viewer & File Exporter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Interactive Hotkeys in Viewer:
  [M]        Toggle 3D Nano-Ruler (measure in nm between chosen points)
  [C]        Clear all ruler measurements and markers
  [Z]        Undo last ruler measurement (or Backspace)
  Shift+LMB  Instant point measure (works anytime without toggling mode)
  [E]        Export standalone interactive 3D HTML web file (opens in any browser)
  [G]        Export standard 3D model (.glb / .gltf)
  [P]        Export scientific colored point cloud (.ply)
  [S]        Save high-resolution snapshot (.png)
  [R]        Reset camera view to centered orientation
  [1] - [9]  Toggle channel visibility
  [H]        Toggle histogram and intensity thresholds
  [D]        Toggle background noise filter (removes isolated 1-5 px dots)
  [+] / [-]  Adjust point size
"""
    )
    parser.add_argument("filename", nargs="?", help="Path to .czi microscopy file (optional, opens file picker if omitted)")
    parser.add_argument("--export-html", dest="export_html", metavar="OUT.html", help="Export to standalone interactive 3D HTML web file")
    parser.add_argument("--export-glb", dest="export_glb", metavar="OUT.glb", help="Export to 3D GLB/GLTF model")
    parser.add_argument("--export-ply", dest="export_ply", metavar="OUT.ply", help="Export to colored binary PLY point cloud")
    parser.add_argument("--headless", action="store_true", help="Perform export and exit immediately without opening the GUI viewer")
    parser.add_argument("--filter-speckles", "--despeckle", action="store_true", dest="filter_speckles", help="Enable isolated noise speckle filter (removes 1-5 px dots) on launch")

    args = parser.parse_args()

    filename = args.filename
    if not filename:
        filename = select_file()

    if not filename:
        print("No file selected.")
        return

    if not os.path.exists(filename):
        print(f"Error: Specified file does not exist: {filename}")
        return

    channels, voxel_x, voxel_y, voxel_z, channel_names, channel_colors = load_czi(filename)

    auto_export = {}
    if args.export_html:
        auto_export["html"] = args.export_html
    if args.export_glb:
        auto_export["glb"] = args.export_glb
    if args.export_ply:
        auto_export["ply"] = args.export_ply
    if args.filter_speckles:
        auto_export["filter_speckles"] = True

    if args.headless:
        auto_export["exit_after"] = True
        # If headless requested but no specific export given, default to HTML and PLY
        if not (args.export_html or args.export_glb or args.export_ply):
            base = os.path.splitext(os.path.basename(filename))[0]
            auto_export["html"] = f"{base}_3d_viewer.html"
            auto_export["ply"] = f"{base}_pointcloud.ply"

    render(
        channels,
        voxel_x,
        voxel_y,
        voxel_z,
        channel_names,
        channel_colors,
        source_filename=filename,
        auto_export=auto_export if auto_export else None
    )

if __name__ == "__main__":
    main()
