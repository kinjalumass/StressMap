import geopandas as gpd
import pytest
from shapely.geometry import Point, box

from census.assignment import assign_population_to_nodes_by_tract_area


def test_population_is_conserved_for_simple_tract():
    nodes = gpd.GeoDataFrame(
        geometry=[Point(0, 0), Point(10, 0)],
        crs="EPSG:26986",
    )
    tracts = gpd.GeoDataFrame(
        {"GEOID": ["test"], "population": [100]},
        geometry=[box(-5, -5, 15, 5)],
        crs="EPSG:26986",
    )

    nodes_out, allocation = assign_population_to_nodes_by_tract_area(
        nodes,
        tracts,
        projected_crs="EPSG:26986",
        tract_filter_method="none",
    )

    assert allocation["assigned_population"].sum() == pytest.approx(100)
    assert nodes_out["assigned_population"].sum() == pytest.approx(100)
    assert allocation["area_share"].sum() == pytest.approx(1)
    assert (allocation["assigned_population"] >= 0).all()


def test_single_candidate_node_receives_entire_population():
    nodes = gpd.GeoDataFrame(
        geometry=[Point(0, 0)],
        crs="EPSG:26986",
    )
    tracts = gpd.GeoDataFrame(
        {"GEOID": ["single"], "population": [75]},
        geometry=[box(-5, -5, 5, 5)],
        crs="EPSG:26986",
    )

    nodes_out, allocation = assign_population_to_nodes_by_tract_area(
        nodes,
        tracts,
        projected_crs="EPSG:26986",
        tract_filter_method="none",
    )

    assert len(allocation) == 1
    assert allocation.iloc[0]["area_share"] == pytest.approx(1)
    assert nodes_out.iloc[0]["assigned_population"] == pytest.approx(75)


def test_invalid_parameters_raise_errors():
    nodes = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:26986")
    tracts = gpd.GeoDataFrame(
        {"GEOID": ["test"], "population": [1]},
        geometry=[box(-1, -1, 1, 1)],
        crs="EPSG:26986",
    )

    with pytest.raises(ValueError, match="candidate_buffer_m"):
        assign_population_to_nodes_by_tract_area(
            nodes,
            tracts,
            candidate_buffer_m=-1,
        )

    with pytest.raises(ValueError, match="min_region_overlap_share"):
        assign_population_to_nodes_by_tract_area(
            nodes,
            tracts,
            min_region_overlap_share=1.1,
        )
