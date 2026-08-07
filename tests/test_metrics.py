from insplad_benchmark.metrics import balanced_accuracy, binary_f1, consistency_error, false_defect_rate, joint_accuracy, macro_f1


def test_balanced_accuracy_and_binary_f1():
    y_true = ["normal", "normal", "anomalous", "anomalous"]
    y_pred = ["normal", "anomalous", "anomalous", "anomalous"]
    assert balanced_accuracy(y_true, y_pred, ["normal", "anomalous"]) == 0.75
    assert round(binary_f1(y_true, y_pred), 6) == round(0.8, 6)


def test_macro_f1():
    assert macro_f1(["a", "b", "b"], ["a", "a", "b"]) > 0


def test_joint_accuracy():
    assert joint_accuracy(["normal", "anomalous"], ["normal", "anomalous"], ["normal", "rust"], ["normal", "rust"]) == 1.0
    assert joint_accuracy(["normal", "anomalous"], ["normal", "normal"], ["normal", "rust"], ["normal", "rust"]) == 0.5


def test_false_defect_rate():
    assert false_defect_rate(["normal", "normal", "anomalous"], ["normal", "rust", "rust"]) == 0.5


def test_consistency_error():
    assert consistency_error(["normal", "anomalous", "anomalous"], ["rust", "normal", "rust"]) == 2 / 3
