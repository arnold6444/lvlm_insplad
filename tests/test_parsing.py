from insplad_benchmark.parsing import parse_anomaly_output, parse_structured_output


def test_parse_anomaly_plain_and_json():
    assert parse_anomaly_output("normal").valid
    assert parse_anomaly_output('{"anomaly":"anomalous"}').anomaly == "anomalous"
    assert not parse_anomaly_output("maybe").valid


def test_parse_structured_output_and_malformed():
    parsed = parse_structured_output('{"anomaly":"anomalous","defect_class":"rust"}', {"normal", "rust"})
    assert parsed.valid and parsed.defect_class == "rust"
    assert not parse_structured_output("not json", {"normal", "rust"}).valid
    assert not parse_structured_output('{"anomaly":"anomalous","defect_class":"made_up"}', {"normal", "rust"}).valid
