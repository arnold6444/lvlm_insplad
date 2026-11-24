from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import score_agent_sft_gold_lenient as lenient  # noqa: E402

Q1 = {"A": "Anomalous.", "B": "Normal."}
Q3 = {"A": "Lower-right region.", "D": "Center region.", "E": "No localized defect region is present.", "J": "Lower-center region."}


def test_strict_json_is_kept_and_a_tool_request_is_not_an_answer():
    assert lenient.lenient_choice('{"action":"finish","choice":"B"}', Q1) == ("B", "json")
    assert lenient.lenient_choice('{"action":"reference"}', Q1) == (None, "tool request")


def test_a_letter_counts_alone_or_with_its_own_option_text_only():
    assert lenient.lenient_choice("J", Q3) == ("J", "letter")
    assert lenient.lenient_choice("A. Anomalous.", Q1) == ("A", "letter")
    assert lenient.lenient_choice("b) normal", {"B": "Normal."}) == (None, "no parsable answer")  # lower case is not an option letter
    assert lenient.lenient_choice("B. Anomalous.", Q1) == (None, "conflict")  # the letter says normal, the text says anomalous
    assert lenient.lenient_choice("C. Normal.", Q1) == (None, "no parsable answer")  # C is not an option here


def test_prose_is_never_read_for_its_meaning():
    prose = "Based on the visual inspection of the target image, there is a clear localized defect in the center."
    assert lenient.lenient_choice(prose, Q3) == (None, "no parsable answer")
    assert lenient.lenient_choice("A. Anomalous. The chain shows rust.", Q1) == (None, "no parsable answer")


def test_option_text_alone_and_one_embedded_json_count():
    assert lenient.lenient_choice("Center region.", Q3) == ("D", "text")
    assert lenient.lenient_choice('I think it is fine. {"action":"finish","choice":"B"}', Q1) == ("B", "embedded")
    # Prose and option texts often start with a capital that is itself an option letter; that must not hide the answer.
    nine = {**Q3, "B": "Middle-left region.", "C": "Upper-center region."}
    assert lenient.lenient_choice("Center region.", nine) == ("D", "text")
    assert lenient.lenient_choice("Anomalous.", Q1) == ("A", "text")
    assert lenient.lenient_choice('Based on the image, the defect is central. {"action":"finish","choice":"D"}', nine) == ("D", "embedded")
    two = '{"action":"finish","choice":"A"} or {"action":"finish","choice":"B"}'
    assert lenient.lenient_choice(two, Q1) == (None, "no parsable answer")
