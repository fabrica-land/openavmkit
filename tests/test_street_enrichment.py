import math

import geopandas as gpd
import pandas as pd
import pytest
from pyproj import CRS, Transformer
from shapely.geometry import LineString, Polygon

import openavmkit.data as data


SETTINGS = {
    "locality": {"units": "imperial"},
    "data": {"process": {"enrich": {"streets": {"enabled": True}}}},
}


def _fixture(edge_y_m: float) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    center_lon = -116.05
    center_lat = 34.15
    utm = CRS.from_user_input(
        f"+proj=utm +zone={int((center_lon + 180) / 6) + 1} +datum=WGS84 +units=m"
    )
    to_utm = Transformer.from_crs("EPSG:4326", utm, always_xy=True)
    to_ll = Transformer.from_crs(utm, "EPSG:4326", always_xy=True)
    cx, cy = to_utm.transform(center_lon, center_lat)
    def polygon_from_offsets(offsets):
        return Polygon([to_ll.transform(cx + dx, cy + dy) for dx, dy in offsets])
    def line_from_offsets(offsets):
        return LineString([to_ll.transform(cx + dx, cy + dy) for dx, dy in offsets])
    parcel = polygon_from_offsets(
        [(-10, -10), (10, -10), (10, 10), (-10, 10), (-10, -10)]
    )
    parcels = gpd.GeoDataFrame(
        {"key": ["subject"], "latitude": [center_lat], "longitude": [center_lon]},
        geometry=[parcel],
        crs="EPSG:4326",
    )
    edges = gpd.GeoDataFrame(
        {"name": ["Mock Road"], "highway": ["residential"], "osmid": [123]},
        geometry=[line_from_offsets([(-80, edge_y_m), (80, edge_y_m)])],
        crs="EPSG:4326",
    )
    return parcels, edges


def _mock_osmnx(monkeypatch, edges: gpd.GeoDataFrame) -> None:
    monkeypatch.setattr(data.ox, "graph_from_bbox", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        data.ox, "graph_to_gdfs", lambda *args, **kwargs: edges.copy()
    )


def _assert_street_contract(out: gpd.GeoDataFrame) -> None:
    for i in range(1, 5):
        assert f"frontage_ft_{i}" in out
        assert f"depth_ft_{i}" in out
        assert f"dist_to_road_ft_{i}" in out
        assert f"osm_road_type_{i}" in out
    assert "osm_total_frontage_ft" in out
    assert "land_area_somers_ft" in out


def test_enrich_df_streets_empty_ray_returns_default_frontage_columns(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    parcels, edges = _fixture(edge_y_m=500.0)
    _mock_osmnx(monkeypatch, edges)
    out = data.enrich_df_streets(
        parcels, SETTINGS, spacing=5.0, max_ray_length=25.0, network_buffer=600.0
    )
    _assert_street_contract(out)
    row = out.iloc[0]
    assert row["key"] == "subject"
    assert row["frontage_ft_1"] == 0.0
    assert row["frontage_ft_4"] == 0.0
    assert row["depth_ft_1"] == 0.0
    assert row["dist_to_road_ft_1"] == 0.0
    assert row["osm_total_frontage_ft"] == 0.0
    assert row["land_area_somers_ft"] == 0.0
    assert pd.isna(row["osm_road_type_1"])


def test_enrich_df_streets_normal_frontage_preserves_first_slot(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    parcels, edges = _fixture(edge_y_m=30.0)
    _mock_osmnx(monkeypatch, edges)
    out = data.enrich_df_streets(
        parcels, SETTINGS, spacing=5.0, max_ray_length=25.0, network_buffer=600.0
    )
    _assert_street_contract(out)
    row = out.iloc[0]
    assert row["key"] == "subject"
    assert row["frontage_ft_1"] > 0.0
    assert row["depth_ft_1"] > 0.0
    assert row["dist_to_road_ft_1"] > 0.0
    assert row["frontage_ft_2"] == 0.0
    assert row["osm_road_type_1"] == "residential"
    assert row["osm_total_frontage_ft"] == pytest.approx(row["frontage_ft_1"])
    assert math.isfinite(row["land_area_somers_ft"])
