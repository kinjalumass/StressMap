import math

import networkx as nx
import pandas as pd
import pytest

from accessibility.network import build_accessibility_graph


def make_graph(
    *,
    reversed_value=False,
    length=100.0,
):
    graph = nx.MultiDiGraph()
    graph.add_node(1)
    graph.add_node(2)
    graph.add_edge(
        1,
        2,
        key=0,
        length=length,
        reversed=reversed_value,
    )
    return graph


def make_lts_row(**overrides):
    row = {
        "u": 1,
        "v": 2,
        "key": 0,
        "LTS_fwd": 2,
        "LTS_rev": 4,
        "bike_allowed_fwd": True,
        "bike_allowed_rev": True,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_forward_edge_uses_forward_lts_and_travel_time():
    graph, diagnostics = build_accessibility_graph(
        make_graph(reversed_value=False),
        make_lts_row(),
        bike_speed_kph=15.0,
    )

    edge = graph[1][2][0]

    assert edge["directional_lts"] == 2
    assert edge["bike_allowed"] is True
    assert edge["lts_direction"] == "fwd"
    assert edge["travel_time_seconds"] == pytest.approx(
        24.0
    )
    assert diagnostics.iloc[0]["status"] == "kept"


def test_reversed_edge_uses_reverse_fields():
    graph, diagnostics = build_accessibility_graph(
        make_graph(reversed_value=True),
        make_lts_row(
            LTS_fwd=1,
            LTS_rev=3,
            bike_allowed_fwd=False,
            bike_allowed_rev=True,
        ),
    )

    assert graph[1][2][0]["directional_lts"] == 3
    assert graph[1][2][0]["lts_direction"] == "rev"
    assert diagnostics.iloc[0]["direction"] == "rev"


def test_lts_row_reversed_value_takes_precedence():
    graph, _ = build_accessibility_graph(
        make_graph(reversed_value=False),
        make_lts_row(
            reversed=True,
            LTS_fwd=1,
            LTS_rev=4,
        ),
    )

    assert graph[1][2][0]["directional_lts"] == 4
    assert graph[1][2][0]["lts_direction"] == "rev"


def test_disallowed_and_invalid_lts_edges_are_removed():
    graph = nx.MultiDiGraph()

    graph.add_edge(
        1,
        2,
        key=0,
        length=50,
        reversed=False,
    )
    graph.add_edge(
        2,
        3,
        key=0,
        length=50,
        reversed=False,
    )

    lts = pd.DataFrame(
        [
            {
                "u": 1,
                "v": 2,
                "key": 0,
                "LTS_fwd": 2,
                "LTS_rev": 2,
                "bike_allowed_fwd": False,
                "bike_allowed_rev": False,
            },
            {
                "u": 2,
                "v": 3,
                "key": 0,
                "LTS_fwd": 0,
                "LTS_rev": 0,
                "bike_allowed_fwd": True,
                "bike_allowed_rev": True,
            },
        ]
    )

    output, diagnostics = build_accessibility_graph(
        graph,
        lts,
    )

    assert output.number_of_edges() == 0

    assert set(diagnostics["status"]) == {
        "bicycle_not_allowed",
        "invalid_directional_lts",
    }


def test_edge_without_lts_row_is_removed_and_reported():
    output, diagnostics = build_accessibility_graph(
        make_graph(),
        make_lts_row().iloc[0:0],
    )

    assert output.number_of_edges() == 0
    assert (
        diagnostics.iloc[0]["status"]
        == "missing_lts_row"
    )


def test_duplicate_lts_edge_ids_raise_error():
    lts = pd.concat(
        [
            make_lts_row(),
            make_lts_row(),
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="duplicate"):
        build_accessibility_graph(
            make_graph(),
            lts,
        )


def test_invalid_speed_and_length_raise_errors():
    with pytest.raises(
        ValueError,
        match="bike_speed_kph",
    ):
        build_accessibility_graph(
            make_graph(),
            make_lts_row(),
            bike_speed_kph=0,
        )

    with pytest.raises(ValueError, match="length"):
        build_accessibility_graph(
            make_graph(length=-1),
            make_lts_row(),
        )


def test_parallel_edges_are_matched_by_key():
    graph = nx.MultiDiGraph()

    graph.add_edge(
        1,
        2,
        key=0,
        length=100,
        reversed=False,
    )
    graph.add_edge(
        1,
        2,
        key=1,
        length=200,
        reversed=False,
    )

    lts = pd.DataFrame(
        [
            {
                "u": 1,
                "v": 2,
                "key": 0,
                "LTS_fwd": 1,
                "LTS_rev": 1,
                "bike_allowed_fwd": True,
                "bike_allowed_rev": True,
            },
            {
                "u": 1,
                "v": 2,
                "key": 1,
                "LTS_fwd": 4,
                "LTS_rev": 4,
                "bike_allowed_fwd": True,
                "bike_allowed_rev": True,
            },
        ]
    )

    output, _ = build_accessibility_graph(
        graph,
        lts,
    )

    assert output[1][2][0]["directional_lts"] == 1
    assert output[1][2][1]["directional_lts"] == 4

    assert math.isclose(
        output[1][2][1]["travel_time_seconds"],
        48.0,
    )
