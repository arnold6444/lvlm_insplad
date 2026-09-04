from insplad_benchmark.agent_iad_lite import parse_agent_decision, should_route, strong_patchcore_conflict


def test_parse_agent_decision_and_routing():
    decision = parse_agent_decision(
        '{"anomaly":"normal","confidence":0.9,"request_tool":"none"}',
        {"none", "zoom"},
    )
    assert decision.valid and decision.anomaly == "normal"
    route, reasons = should_route(
        decision,
        expected_tool="zoom",
        patchcore_percentile=95.0,
    )
    assert route and "patchcore_conflict" in reasons

    decision2 = parse_agent_decision(
        '{"anomaly":"anomalous","confidence":0.5,"request_tool":"reference"}',
        {"none", "reference"},
    )
    route2, reasons2 = should_route(
        decision2,
        expected_tool="reference",
        patchcore_percentile=70.0,
    )
    assert route2
    assert "model_requested_tool" in reasons2
    assert "low_confidence" in reasons2


def test_invalid_and_conflict_edges():
    bad = parse_agent_decision("not json", {"none"})
    route, reasons = should_route(
        bad,
        expected_tool="zoom",
        patchcore_percentile=50,
    )
    assert route and reasons == ["invalid_output"]
    assert strong_patchcore_conflict("normal", 90)
    assert strong_patchcore_conflict("anomalous", 10)
    assert not strong_patchcore_conflict("normal", 89.9)
