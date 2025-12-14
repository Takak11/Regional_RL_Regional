import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point, Polygon
from scipy.spatial import Voronoi
from sklearn.cluster import KMeans


# ------------------------
# 辅助函数：处理无限Voronoi区域
# 这个函数的作用是将scipy生成的、可能包含无限区域的Voronoi图，转换为一组封闭的多边形。
# ------------------------
def voronoi_finite_polygons_2d(vor, radius=None):
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge
            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)

        # sort region vertices counter-clockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.asarray(new_region)[np.argsort(angles)]

        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


# ------------------------
# 区域边界

west, south, east, north = 103.9808, 30.5963, 104.1614, 30.7291
boundary_poly = Polygon([
    (west, south), (east, south),
    (east, north), (west, north),
    (west, south)
])
gdf_box = gpd.GeoDataFrame(geometry=[boundary_poly], crs='EPSG:4326')
csv_path = "../dataset/fcs_regions.csv"  # 替换成你的文件路径
df_extra = pd.read_csv(csv_path)
extra_points = df_extra[['longitude', 'latitude']].to_numpy()

# ------------------------
# 1. 在区域内生成均匀网格点
num_grid = 10000
lons = np.random.uniform(west, east, num_grid)
lats = np.random.uniform(south, north, num_grid)
grid_points = np.array(list(zip(lons, lats)))

all_points = np.vstack([extra_points])

# fcs_ids = [f"FCS_{i + 1}" for i in range(len(extra_points))]
fcs_ids = [f"FCS_{i + 1}" for i in range(len(all_points))]
# fcs_ids = [f"FCS_{i + 1}" for i in range(len(fcs_coords))]


# 3. 构建 GeoDataFrame
gdf_fcs = gpd.GeoDataFrame({'id': fcs_ids},
                           # geometry=[Point(lon, lat) for lon, lat in extra_points],
                           geometry=[Point(lon, lat) for lon, lat in all_points],
                           # geometry=[Point(lon, lat) for lon, lat in fcs_coords],
                           crs='EPSG:4326'
                           )

# ------------------------
# 4. Voronoi 构建 + 裁剪 (修正后的逻辑)
# ------------------------
# 使用原始坐标点生成Voronoi图
# vor = Voronoi(extra_points)
vor = Voronoi(all_points)
# vor = Voronoi(fcs_coords)
# 使用辅助函数将所有区域（包括无限区域）转换为封闭的多边形
regions, vertices = voronoi_finite_polygons_2d(vor)

# 将所有生成的多边形与研究区域边界进行裁剪
clipped_regions = []
for region in regions:
    # 从顶点坐标创建Shapely多边形
    polygon_coords = vertices[region]
    poly = Polygon(polygon_coords)

    # 与研究区域边界进行裁剪
    clipped = poly.intersection(boundary_poly)
    if not clipped.is_empty:
        clipped_regions.append(clipped)

# 确保我们得到了10个区域
print(f"成功生成并裁剪了 {len(clipped_regions)} 个 Voronoi 区域。")

# 创建包含裁剪后Voronoi区域的GeoDataFrame
gdf_regions = gpd.GeoDataFrame(geometry=clipped_regions, crs='EPSG:4326')

# 将每个区域与原始FCS点对应起来 (可选，但推荐)
# 通过空间连接，将FCS点的ID赋给其所在的Voronoi区域
gdf_regions = gpd.sjoin(gdf_regions, gdf_fcs, how="inner", predicate='contains').drop(columns=['index_right'])

# ------------------------
# 5. 可视化：底图 + 区域 + 点
# ------------------------
gdf_fcs_web = gdf_fcs.to_crs(epsg=3857)
gdf_regions_web = gdf_regions.to_crs(epsg=3857)
gdf_box_web = gdf_box.to_crs(epsg=3857)

fig, ax = plt.subplots(figsize=(12, 12))

# 绘制裁剪后的Voronoi区域
gdf_regions_web.plot(ax=ax, alpha=0.5, edgecolor='black', cmap='viridis', linewidth=1.5)

# 绘制研究区域边界
gdf_box_web.plot(ax=ax, edgecolor='gray', facecolor='none', linestyle='--', linewidth=2)

# 绘制FCS中心点
gdf_fcs_web.plot(ax=ax, color='red', markersize=80, label='FCS_Location', zorder=5)

# 为FCS中心点添加标签
for idx, row in gdf_fcs_web.iterrows():
    ax.annotate(row['id'], xy=(row.geometry.x, row.geometry.y), xytext=(5, 5),
                textcoords='offset points', fontsize=10, color='white',
                bbox=dict(boxstyle="round,pad=0.3", fc="red", ec="none", alpha=0.7))

# 添加在线地图底图
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

# 设置图表属性
plt.title("Voronoi Regions")
plt.legend()
plt.axis('off')
plt.tight_layout()
plt.show()

# 添加经纬度列
gdf_fcs['lon'] = gdf_fcs.geometry.x
gdf_fcs['lat'] = gdf_fcs.geometry.y

# 2. 保存 Voronoi 区域为 GeoJSON
gdf_regions.to_file("../dataset/fcs_regions.geojson", driver="GeoJSON")

# 3. 保存图像（PNG）
fig.savefig("../dataset/fcs_voronoi_map.png", dpi=300)

print("保存完成")