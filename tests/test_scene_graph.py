"""
Tests for Persistent 4D Scene Graph, Query DSL interpreter, and Graph Diff Event detection.
"""

import pytest
from src.memory.scene_graph import (
    Persistent4DSceneGraph,
    SceneGraphNode,
    SceneGraphEdge,
    GraphDiffEvent,
    compute_graph_diff,
)


def test_node_and_edge_operations():
    sg = Persistent4DSceneGraph()

    n1 = SceneGraphNode(
        node_id="person_1",
        label="person",
        track_id=1,
        time_span=(0, 50),
        centroid_3d=[0.0, 0.0, 2.0],
        attributes={"posture": "standing", "color": "blue"},
    )
    n2 = SceneGraphNode(
        node_id="laptop_1",
        label="laptop",
        track_id=2,
        time_span=(0, 50),
        centroid_3d=[0.0, 0.5, 1.8],
        attributes={"state": "open"},
    )

    sg.add_node(n1)
    sg.add_node(n2)

    assert sg.get_node("person_1") is not None
    assert sg.get_node("person_1").label == "person"
    assert len(sg.list_nodes()) == 2

    # Attempting to add edge with non-existent node must raise KeyError
    bad_edge = SceneGraphEdge(
        source_id="person_1",
        target_id="ghost_node",
        relation_type="using",
        time_span=(0, 30),
    )
    with pytest.raises(KeyError):
        sg.add_edge(bad_edge)

    # Valid edge
    edge = SceneGraphEdge(
        source_id="person_1",
        target_id="laptop_1",
        relation_type="interacting_with",
        time_span=(10, 40),
        metric_distance_meters=0.5,
    )
    sg.add_edge(edge)
    assert len(sg.list_edges()) == 1

    edges_found = sg.get_edges(source_id="person_1", relation_type="interacting_with")
    assert len(edges_found) == 1
    assert edges_found[0].target_id == "laptop_1"


def test_temporal_subgraph_slicing():
    sg = Persistent4DSceneGraph()

    n1 = SceneGraphNode(node_id="obj_a", label="car", time_span=(0, 30))
    n2 = SceneGraphNode(node_id="obj_b", label="truck", time_span=(20, 60))
    n3 = SceneGraphNode(node_id="obj_c", label="bicycle", time_span=(50, 90))

    sg.add_node(n1)
    sg.add_node(n2)
    sg.add_node(n3)

    edge_12 = SceneGraphEdge(source_id="obj_a", target_id="obj_b", relation_type="behind", time_span=(20, 30))
    edge_23 = SceneGraphEdge(source_id="obj_b", target_id="obj_c", relation_type="left_of", time_span=(50, 60))
    sg.add_edge(edge_12)
    sg.add_edge(edge_23)

    # At frame 10: only obj_a is active
    sub_10 = sg.get_active_subgraph_at_frame(10)
    assert [n.node_id for n in sub_10.list_nodes()] == ["obj_a"]
    assert len(sub_10.list_edges()) == 0

    # At frame 25: obj_a and obj_b are active, edge_12 is active
    sub_25 = sg.get_active_subgraph_at_frame(25)
    node_ids_25 = {n.node_id for n in sub_25.list_nodes()}
    assert node_ids_25 == {"obj_a", "obj_b"}
    assert len(sub_25.list_edges()) == 1

    # At frame 55: obj_b and obj_c are active, edge_23 is active
    sub_55 = sg.get_active_subgraph_at_frame(55)
    node_ids_55 = {n.node_id for n in sub_55.list_nodes()}
    assert node_ids_55 == {"obj_b", "obj_c"}
    assert len(sub_55.list_edges()) == 1


def test_serialization_and_deserialization():
    sg = Persistent4DSceneGraph()
    n = SceneGraphNode(
        node_id="robot_1",
        label="robot",
        time_span=(0, 100),
        centroid_3d=[1.0, 2.0, 3.0],
        attributes={"battery": 95},
    )
    sg.add_node(n)

    json_str = sg.to_json()
    assert "robot_1" in json_str

    sg_loaded = Persistent4DSceneGraph.from_json(json_str)
    assert sg_loaded.get_node("robot_1") is not None
    assert sg_loaded.get_node("robot_1").attributes["battery"] == 95
    assert sg_loaded.get_node("robot_1").centroid_3d == [1.0, 2.0, 3.0]


def test_query_dsl():
    sg = Persistent4DSceneGraph()

    p1 = SceneGraphNode(
        node_id="p1",
        label="person",
        time_span=(0, 20),
        centroid_3d=[0.0, 0.0, 3.0],
        attributes={"color": "red", "posture": "sitting"},
    )
    p2 = SceneGraphNode(
        node_id="p2",
        label="person",
        time_span=(0, 20),
        centroid_3d=[0.0, 0.0, 7.0],
        attributes={"color": "blue", "posture": "standing"},
    )
    c1 = SceneGraphNode(
        node_id="c1",
        label="chair",
        time_span=(0, 20),
        centroid_3d=[0.0, 0.5, 3.0],
    )
    u1 = SceneGraphNode(
        node_id="u1",
        label="umbrella",
        time_span=(0, 20),
        centroid_3d=[0.2, 0.0, 3.0],
    )

    sg.add_node(p1)
    sg.add_node(p2)
    sg.add_node(c1)
    sg.add_node(u1)

    sg.add_edge(SceneGraphEdge(source_id="p1", target_id="u1", relation_type="holding", time_span=(0, 20)))

    # Query 1: Find by label
    res1 = sg.execute_query("FIND ?x WHERE ?x.label == 'person'")
    assert len(res1) == 2

    # Query 2: Find by label and color attribute
    res2 = sg.execute_query("FIND ?x WHERE ?x.label == 'person' AND ?x.color == 'red'")
    assert len(res2) == 1
    assert res2[0]["x"]["node_id"] == "p1"

    # Query 3: Distance threshold
    res3 = sg.execute_query("FIND ?x WHERE ?x.distance < 5.0")
    # p1 (3.0), c1 (~3.04), u1 (~3.0) should match; p2 (7.0) should not
    matched_ids = {r["x"]["node_id"] for r in res3}
    assert "p1" in matched_ids
    assert "p2" not in matched_ids

    # Query 4: Two-variable join with relation
    res4 = sg.execute_query("FIND ?x, ?y WHERE ?x.label == 'person' AND ?y.label == 'umbrella' AND ?x.holding == ?y")
    assert len(res4) == 1
    assert res4[0]["x"]["node_id"] == "p1"
    assert res4[0]["y"]["node_id"] == "u1"

    # Query 5: Invalid syntax raises ValueError
    with pytest.raises(ValueError):
        sg.execute_query("SELECT ALL OBJECTS")


def test_graph_diff_events():
    g1 = Persistent4DSceneGraph()
    g2 = Persistent4DSceneGraph()

    # t1: Person standing near bag
    g1.add_node(
        SceneGraphNode(
            node_id="pers",
            label="person",
            attributes={"posture": "standing"},
        )
    )
    g1.add_node(
        SceneGraphNode(
            node_id="bag",
            label="backpack",
            attributes={"color": "black"},
        )
    )
    g1.add_edge(
        SceneGraphEdge(
            source_id="pers",
            target_id="bag",
            relation_type="near",
        )
    )

    # t2: Person sat down (posture change), picked up bag (relation changed from near to holding), and a cup appeared
    g2.add_node(
        SceneGraphNode(
            node_id="pers",
            label="person",
            attributes={"posture": "sitting"},
        )
    )
    g2.add_node(
        SceneGraphNode(
            node_id="bag",
            label="backpack",
            attributes={"color": "black"},
        )
    )
    g2.add_node(
        SceneGraphNode(
            node_id="cup",
            label="coffee_cup",
            attributes={"color": "white"},
        )
    )
    g2.add_edge(
        SceneGraphEdge(
            source_id="pers",
            target_id="bag",
            relation_type="holding",
        )
    )

    events = compute_graph_diff(g1, g2, frame_index=45)
    event_types = [ev.event_type for ev in events]

    assert "INSTANCE_APPEARED" in event_types
    assert "ATTRIBUTE_CHANGED" in event_types
    assert "RELATION_FORMED" in event_types
    assert "RELATION_BROKEN" in event_types

    # Verify appearance event
    appear_ev = next(ev for ev in events if ev.event_type == "INSTANCE_APPEARED")
    assert appear_ev.subject_id == "cup"

    # Verify attribute change
    attr_ev = next(ev for ev in events if ev.event_type == "ATTRIBUTE_CHANGED")
    assert attr_ev.subject_id == "pers"
    assert attr_ev.details["changed_attributes"]["posture"]["before"] == "standing"
    assert attr_ev.details["changed_attributes"]["posture"]["after"] == "sitting"

    # Verify relation changes
    formed_ev = next(ev for ev in events if ev.event_type == "RELATION_FORMED")
    assert formed_ev.details["relation_type"] == "holding"

    broken_ev = next(ev for ev in events if ev.event_type == "RELATION_BROKEN")
    assert broken_ev.details["relation_type"] == "near"
