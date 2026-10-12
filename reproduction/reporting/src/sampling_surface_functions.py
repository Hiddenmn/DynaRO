def box_edges_xyz(extent_zyx: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    lz, ly, lx = (float(value) for value in extent_zyx)
    corners = {
        (ix, iy, iz): np.array([ix * lx, iy * ly, iz * lz], dtype=float)
        for ix in (0, 1)
        for iy in (0, 1)
        for iz in (0, 1)
    }
    edges: list[tuple[np.ndarray, np.ndarray]] = []
    for key, start in corners.items():
        for axis in range(3):
            neighbour = list(key)
            neighbour[axis] = 1 - neighbour[axis]
            neighbour_key = tuple(neighbour)
            if key < neighbour_key:
                edges.append((start, corners[neighbour_key]))
    return edges


def deterministic_points(
    mask: np.ndarray,
    class_id: int,
    modulus: int = POINT_HASH_MODULUS,
) -> np.ndarray:
    coords_zyx = np.argwhere(mask)
    z = coords_zyx[:, 0].astype(np.uint64)
    y = coords_zyx[:, 1].astype(np.uint64)
    x = coords_zyx[:, 2].astype(np.uint64)
    hashed = (
        (z * 73856093)
        ^ (y * 19349663)
        ^ (x * 83492791)
        ^ np.uint64(class_id * 2654435761)
    )
    return coords_zyx[(hashed % modulus) == 0]


def add_class_layer(
    ax: plt.Axes,
    labels: np.ndarray,
    class_id: int,
) -> dict[str, int]:
    class_mask = labels == class_id
    surface_mask = class_mask[
        ::SURFACE_GRID_COARSEN,
        ::SURFACE_GRID_COARSEN,
        ::SURFACE_GRID_COARSEN,
    ]
    surface_spacing = DISPLAY_SPACING_MM_ZYX * SURFACE_GRID_COARSEN
    vertices_zyx, faces, _normals, _values = marching_cubes(
        surface_mask.astype(np.uint8),
        level=0.5,
        spacing=tuple(float(value) for value in surface_spacing),
        step_size=1,
        allow_degenerate=False,
    )
    vertices_xyz = vertices_zyx[:, [2, 1, 0]]
    surface = Poly3DCollection(
        vertices_xyz[faces],
        facecolors=CLASS_COLORS[class_id],
        edgecolor="none",
        alpha=SURFACE_ALPHA,
        shade=True,
        lightsource=mpl.colors.LightSource(azdeg=LIGHT_AZIMUTH, altdeg=LIGHT_ALTITUDE),
        linewidth=0,
        rasterized=True,
    )
    ax.add_collection3d(surface)

    coords_zyx = deterministic_points(class_mask, class_id)
    coords_mm_zyx = coords_zyx * DISPLAY_SPACING_MM_ZYX
    ax.scatter(
        coords_mm_zyx[:, 2],
        coords_mm_zyx[:, 1],
        coords_mm_zyx[:, 0],
        s=POINT_SIZE,
        c=CLASS_COLORS[class_id],
        marker=CLASS_MARKERS[class_id],
        alpha=POINT_ALPHA,
        depthshade=False,
        linewidths=0,
        rasterized=True,
    )
    return {
        "display_grid_voxels": int(np.count_nonzero(class_mask)),
        "surface_vertices": int(len(vertices_xyz)),
        "surface_faces": int(len(faces)),
        "render_points": int(len(coords_zyx)),
    }


def add_surface_layer(
    ax: plt.Axes,
    mask: np.ndarray,
    color: str,
    alpha: float,
) -> dict[str, int]:
    """Add a fixed 0.5 isosurface from a binary display-grid class mask."""
    if not np.any(mask):
        return {"display_grid_voxels": 0, "surface_vertices": 0, "surface_faces": 0}
    vertices_zyx, faces, _normals, _values = marching_cubes(
        mask.astype(np.uint8),
        level=0.5,
        spacing=tuple(float(value) for value in DISPLAY_SPACING_MM_ZYX),
        step_size=1,
        allow_degenerate=False,
    )
    vertices_xyz = vertices_zyx[:, [2, 1, 0]]
    ax.add_collection3d(
        Poly3DCollection(
            vertices_xyz[faces],
            facecolors=color,
            edgecolor="none",
            alpha=alpha,
            shade=True,
            lightsource=mpl.colors.LightSource(azdeg=LIGHT_AZIMUTH, altdeg=LIGHT_ALTITUDE),
            linewidth=0,
            rasterized=True,
        )
    )
    return {
        "display_grid_voxels": int(np.count_nonzero(mask)),
        "surface_vertices": int(len(vertices_xyz)),
        "surface_faces": int(len(faces)),
    }
