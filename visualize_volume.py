import struct
import argparse
import sys
import numpy as np
import pyvista as pv

HEADER_FORMAT = "<8s I I I I I d d d d d d d I d I 16s 16s"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

def read_volume_dump(filepath):
    """Reads a fourwing volumetric dump file."""
    with open(filepath, "rb") as f:
        header_data = f.read(HEADER_SIZE)
        if len(header_data) < HEADER_SIZE:
            raise ValueError("File is too small to contain a valid header.")
        
        parsed = struct.unpack(HEADER_FORMAT, header_data)
        
        magic = parsed[0].decode('ascii').strip('\x00')
        if magic != "FW7VOL1":
            raise ValueError(f"Invalid magic signature: {magic}")
            
        header = {
            "version": parsed[1],
            "system": parsed[2],
            "a_count": parsed[3],
            "b_count": parsed[4],
            "c_count": parsed[5],
            "a_min": parsed[6],
            "a_max": parsed[7],
            "b_min": parsed[8],
            "b_max": parsed[9],
            "c_min": parsed[10],
            "c_max": parsed[11],
            "dt": parsed[12],
            "max_steps": parsed[13],
            "escape_radius": parsed[14],
            "seed_axis_count": parsed[15],
            "layout": parsed[16].decode('ascii').strip('\x00')
        }
        
        # Calculate exactly how many items we should read
        total_items = header["a_count"] * header["b_count"] * header["c_count"]
        
        # Read the raw steps data
        volume_data = np.frombuffer(f.read(), dtype=np.uint32)
        
        if len(volume_data) != total_items:
            raise ValueError(f"Expected {total_items} cells, but read {len(volume_data)} from file.")
            
        # Reshape to a 3D grid: C-order matches a-major,b,c
        # array[ia, ib, ic]
        volume_data = volume_data.reshape((header["a_count"], header["b_count"], header["c_count"]))
        
        return header, volume_data

def visualize_volume(filepath):
    print(f"Loading '{filepath}'...")
    header, volume = read_volume_dump(filepath)
    
    print("Header info:")
    for k, v in header.items():
        print(f"  {k}: {v}")
        
    a_count, b_count, c_count = header["a_count"], header["b_count"], header["c_count"]
    
    # Create the spatial reference
    grid = pv.ImageData()
    grid.dimensions = (a_count + 1, b_count + 1, c_count + 1)
    
    # Set spacing (cell size) and origin based on bounds
    a_min, a_max = header["a_min"], header["a_max"]
    b_min, b_max = header["b_min"], header["b_max"]
    c_min, c_max = header["c_min"], header["c_max"]

    if a_max < a_min:
        volume = np.flip(volume, axis=0)
        a_min, a_max = a_max, a_min
    if b_max < b_min:
        volume = np.flip(volume, axis=1)
        b_min, b_max = b_max, b_min
    if c_max < c_min:
        volume = np.flip(volume, axis=2)
        c_min, c_max = c_max, c_min

    # For PyVista ImageData, spacing must be >= 0
    dx = (a_max - a_min) / max(1, a_count - 1) if a_count > 1 else 1.0
    dy = (b_max - b_min) / max(1, b_count - 1) if b_count > 1 else 1.0
    dz = (c_max - c_min) / max(1, c_count - 1) if c_count > 1 else 1.0
    
    grid.spacing = (dx, dy, dz)
    grid.origin = (a_min - dx/2, b_min - dy/2, c_min - dz/2)

    # Flatten the volume in Fortran (column-major) order to correctly assign to cells in PyVista
    # PyVista is VTK-based, which expects x, y, z ordering (Fortran).
    grid.cell_data["steps"] = volume.flatten(order="F")

    print(f"Grid bounds: {grid.bounds}")
    
    # We want to completely eliminate voxels that hit `max_steps`.
    max_steps = header["max_steps"]
    
    # Threshold out everything exactly equal to max_steps
    # threshold gives points strictly between [0, max_steps - 0.5]
    print(f"Applying threshold to hide max_steps ({max_steps})...")
    visible_cells = grid.threshold([0, max_steps - 0.5], scalars="steps")

    real_max = 0
    if visible_cells.n_cells == 0:
        print("Warning: All cells reached max_steps. Nothing to display!")
    else:
        real_max = float(visible_cells.cell_data["steps"].max())
        print(f"Highest visible step found: {real_max}")

    # Set up interactive plotter
    plotter = pv.Plotter(title="Fourwing Volumetric Visualizer")
    
    # Visually normalize the scaling of the 3 axes so the rendered object is a cube
    # This prevents the volume from looking stretched if (a_max - a_min) is much larger than (b_max - b_min)
    lx = max(1e-9, a_max - a_min)
    ly = max(1e-9, b_max - b_min)
    lz = max(1e-9, c_max - c_min)
    max_len = max(lx, ly, lz)
    plotter.set_scale(xscale=max_len/lx, yscale=max_len/ly, zscale=max_len/lz)
    
    # Add initial mesh to configure bounds and camera correctly
    plotter.add_mesh(
        visible_cells, 
        cmap="plasma",
        show_edges=False, 
        name="main_volume",
        clim=[0, real_max],
        scalar_bar_args={'title': 'Number of Steps'}
    )
    
    # I'll replace it with two sliders: MIN and MAX.
    def set_min(val):
        ranges['min'] = val
        recalc_threshold()
        
    def set_max(val):
        ranges['max'] = val
        recalc_threshold()
        
    def recalc_threshold():
        # Keep inside valid ranges
        valid_min = min(ranges['min'], ranges['max'])
        valid_max = max(ranges['min'], ranges['max'])
        
        thresh_mesh = grid.threshold([valid_min, valid_max], scalars="steps")
        
        plotter.remove_actor('main_volume')
        if thresh_mesh.n_cells > 0:
            plotter.add_mesh(
                thresh_mesh, 
                cmap="plasma", 
                show_edges=False, 
                name="main_volume",
                clim=[0, real_max],
                show_scalar_bar=False
            )

    ranges = {'min': 0, 'max': real_max}

    plotter.clear_slider_widgets()
    
    # Ensure slider range doesn't completely collapse to 0
    slider_max = max(1.0, real_max)
    plotter.add_slider_widget(set_min, [0, slider_max], value=0, title="Minimum Steps", pointa=(0.65, 0.9), pointb=(0.95, 0.9))
    plotter.add_slider_widget(set_max, [0, slider_max], value=slider_max, title="Maximum Steps", pointa=(0.65, 0.75), pointb=(0.95, 0.75))

    plotter.add_axes()
    plotter.show_grid()
    plotter.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Fourwing Volumes")
    parser.add_argument("file", help="Path to the .dump or .vol file")
    args = parser.parse_args()
    
    visualize_volume(args.file)
