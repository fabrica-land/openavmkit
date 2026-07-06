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
METRIC_SETTINGS = {
    "locality": {"units": "metric"},
    "data": {"process": {"enrich": {"streets": {"enabled": True}}}},
}
SUBJECT_KEY = "06071-062124-0000:subject"


def _fixture(
    edge_y_m: float, edge_half_length_m: float = 80.0
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
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
        {"key": [SUBJECT_KEY], "latitude": [center_lat], "longitude": [center_lon]},
        geometry=[parcel],
        crs="EPSG:4326",
    )
    edges = gpd.GeoDataFrame(
        {"name": ["Mock Road"], "highway": ["residential"], "osmid": [123]},
        geometry=[
            line_from_offsets(
                [(-edge_half_length_m, edge_y_m), (edge_half_length_m, edge_y_m)]
            )
        ],
        crs="EPSG:4326",
    )
    return parcels, edges


def _empty_edges() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "name": pd.Series(dtype="object"),
            "highway": pd.Series(dtype="object"),
            "osmid": pd.Series(dtype="object"),
        },
        geometry=gpd.GeoSeries([], crs="EPSG:4326"),
        crs="EPSG:4326",
    )


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


def _enrich_streets(
    tmp_path,
    monkeypatch,
    edge_y_m: float,
    settings: dict = SETTINGS,
    edges: gpd.GeoDataFrame | None = None,
    edge_half_length_m: float = 80.0,
) -> gpd.GeoDataFrame:
    monkeypatch.chdir(tmp_path)
    parcels, fixture_edges = _fixture(edge_y_m, edge_half_length_m=edge_half_length_m)
    _mock_osmnx(monkeypatch, edges if edges is not None else fixture_edges)
    return data.enrich_df_streets(
        parcels,
        settings,
        spacing=5.0,
        max_ray_length=25.0,
        network_buffer=600.0,
    )


def _assert_default_imperial_row(row: pd.Series) -> None:
    assert row["key"] == SUBJECT_KEY
    assert row["frontage_ft_1"] == 0.0
    assert row["depth_ft_1"] == 0.0
    assert row["dist_to_road_ft_1"] == 0.0
    assert row["osm_total_frontage_ft"] == 0.0
    assert row["land_area_somers_ft"] == 0.0
    assert pd.isna(row["osm_road_type_1"])


def _enrich_with_stale_slots(
    tmp_path,
    monkeypatch,
    edge_y_m: float,
    stale_slots: dict[str, object],
    edges: gpd.GeoDataFrame | None = None,
) -> gpd.GeoDataFrame:
    monkeypatch.chdir(tmp_path)
    parcels, fixture_edges = _fixture(edge_y_m=edge_y_m)
    for col, value in stale_slots.items():
        parcels[col] = value
    _mock_osmnx(monkeypatch, edges if edges is not None else fixture_edges)
    return data.enrich_df_streets(
        parcels, SETTINGS, spacing=5.0, max_ray_length=25.0, network_buffer=600.0
    )


def test_enrich_df_streets_empty_ray_returns_default_frontage_columns(
    tmp_path, monkeypatch, capsys
):
    out = _enrich_streets(tmp_path, monkeypatch, edge_y_m=500.0)
    _assert_street_contract(out)
    row = out.iloc[0]
    assert row["frontage_ft_4"] == 0.0
    _assert_default_imperial_row(row)
    assert "Ray par is empty, return early" in capsys.readouterr().out


def test_enrich_df_streets_roadless_edges_return_default_frontage_columns(
    tmp_path, monkeypatch, capsys
):
    out = _enrich_streets(
        tmp_path, monkeypatch, edge_y_m=500.0, edges=_empty_edges()
    )
    _assert_street_contract(out)
    _assert_default_imperial_row(out.iloc[0])
    assert "No street edges found, return early" in capsys.readouterr().out


def test_enrich_df_streets_short_edges_return_default_frontage_columns(
    tmp_path, monkeypatch, capsys
):
    out = _enrich_streets(
        tmp_path, monkeypatch, edge_y_m=30.0, edge_half_length_m=0.25
    )
    _assert_street_contract(out)
    _assert_default_imperial_row(out.iloc[0])
    assert "No street rays generated, return early" in capsys.readouterr().out


def test_enrich_df_streets_roadless_edges_discard_stale_existing_slots(
    tmp_path, monkeypatch
):
    out = _enrich_with_stale_slots(
        tmp_path,
        monkeypatch,
        edge_y_m=500.0,
        stale_slots={"frontage_1": 99.0, "depth_1": 88.0, "dist_to_road_1": 77.0},
        edges=_empty_edges(),
    )
    row = out.iloc[0]
    assert row["frontage_ft_1"] == 0.0
    assert row["depth_ft_1"] == 0.0
    assert row["dist_to_road_ft_1"] == 0.0
    assert row["land_area_somers_ft"] == 0.0


def test_enrich_df_streets_empty_ray_returns_metric_somers_columns(
    tmp_path, monkeypatch
):
    out = _enrich_streets(
        tmp_path, monkeypatch, edge_y_m=500.0, settings=METRIC_SETTINGS
    )
    row = out.iloc[0]
    assert row["key"] == SUBJECT_KEY
    assert row["frontage_m_1"] == 0.0
    assert row["depth_m_1"] == 0.0
    assert row["dist_to_road_m_1"] == 0.0
    assert row["osm_total_frontage_m"] == 0.0
    assert row["land_area_somers_m"] == 0.0
    assert "land_area_somers_ft" not in out


def test_enrich_df_streets_normal_frontage_preserves_first_slot(
    tmp_path, monkeypatch
):
    out = _enrich_streets(tmp_path, monkeypatch, edge_y_m=30.0)
    _assert_street_contract(out)
    row = out.iloc[0]
    assert row["key"] == SUBJECT_KEY
    assert row["frontage_ft_1"] > 0.0
    assert row["depth_ft_1"] > 0.0
    assert row["dist_to_road_ft_1"] > 0.0
    assert row["frontage_ft_2"] == 0.0
    assert row["osm_road_type_1"] == "residential"
    assert row["osm_total_frontage_ft"] == pytest.approx(row["frontage_ft_1"])
    assert math.isfinite(row["land_area_somers_ft"])


def test_enrich_df_streets_normal_frontage_discards_stale_existing_slots(
    tmp_path, monkeypatch
):
    out = _enrich_with_stale_slots(
        tmp_path,
        monkeypatch,
        edge_y_m=30.0,
        stale_slots={"frontage_1": 99.0, "depth_1": 88.0, "dist_to_road_1": 77.0},
    )
    row = out.iloc[0]
    cached = pd.read_parquet(tmp_path / "in/osm/streets.parquet")
    assert row["frontage_ft_1"] > 0.0
    assert row["frontage_ft_1"] != pytest.approx(99.0 * 3.28084)
    assert row["depth_ft_1"] != pytest.approx(88.0 * 3.28084)
    assert row["dist_to_road_ft_1"] != pytest.approx(77.0 * 3.28084)
    assert "frontage_ft_1" in cached


def test_enrich_df_streets_normal_frontage_rejects_malformed_existing_slot(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    parcels, edges = _fixture(edge_y_m=30.0)
    parcels["frontage_1"] = "not-a-number"
    _mock_osmnx(monkeypatch, edges)
    with pytest.raises(ValueError):
        data.enrich_df_streets(
            parcels,
            SETTINGS,
            spacing=5.0,
            max_ray_length=25.0,
            network_buffer=600.0,
        )


def test_enrich_df_streets_normal_frontage_preserves_metric_somers(
    tmp_path, monkeypatch
):
    imperial = _enrich_streets(tmp_path, monkeypatch, edge_y_m=30.0)
    (tmp_path / "metric").mkdir()
    metric = _enrich_streets(
        tmp_path / "metric", monkeypatch, edge_y_m=30.0, settings=METRIC_SETTINGS
    )
    imperial_row = imperial.iloc[0]
    metric_row = metric.iloc[0]
    assert metric_row["frontage_m_1"] > 0.0
    assert metric_row["depth_m_1"] > 0.0
    assert metric_row["dist_to_road_m_1"] > 0.0
    assert metric_row["osm_total_frontage_m"] == pytest.approx(
        metric_row["frontage_m_1"]
    )
    assert metric_row["land_area_somers_m"] == pytest.approx(
        imperial_row["land_area_somers_ft"] * 0.3048
    )


def test_enrich_df_streets_metric_ignores_imperial_street_cache(
    tmp_path, monkeypatch
):
    imperial = _enrich_streets(tmp_path, monkeypatch, edge_y_m=30.0)
    metric = _enrich_streets(
        tmp_path, monkeypatch, edge_y_m=30.0, settings=METRIC_SETTINGS
    )
    assert "frontage_ft_1" in imperial
    assert "frontage_m_1" in metric
    assert "land_area_somers_m" in metric
    assert "land_area_somers_ft" not in metric


def test_enrich_df_streets_roadless_edges_discard_suffixed_existing_slots(
    tmp_path, monkeypatch
):
    out = _enrich_with_stale_slots(
        tmp_path,
        monkeypatch,
        edge_y_m=500.0,
        stale_slots={
            "frontage_ft_1": 99.0,
            "depth_ft_1": 88.0,
            "dist_to_road_ft_1": 77.0,
        },
        edges=_empty_edges(),
    )
    assert out.columns.is_unique
    _assert_default_imperial_row(out.iloc[0])


def test_enrich_df_streets_cache_discards_suffixed_existing_slots(
    tmp_path, monkeypatch
):
    _enrich_streets(tmp_path, monkeypatch, edge_y_m=30.0)
    out = _enrich_with_stale_slots(
        tmp_path,
        monkeypatch,
        edge_y_m=30.0,
        stale_slots={
            "frontage_ft_1": 99.0,
            "depth_ft_1": 88.0,
            "dist_to_road_ft_1": 77.0,
        },
        edges=_empty_edges(),
    )
    row = out.iloc[0]
    assert out.columns.is_unique
    assert "frontage_ft_1_x" not in out
    assert "frontage_ft_1_y" not in out
    assert row["frontage_ft_1"] > 0.0
