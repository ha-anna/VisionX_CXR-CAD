import numpy as np
import pytest

from src.api.schemas import LABELS
from src.data.labels import DISEASES, encode_labels


def test_label_order_matches_api():
    assert list(DISEASES) == LABELS


def test_no_finding_is_all_zeros():
    encoded = encode_labels("No Finding")

    assert encoded.shape == (14,)
    assert not encoded.any()


def test_two_findings_set_exactly_two_ones():
    encoded = encode_labels("Cardiomegaly|Effusion")

    assert encoded.sum() == 2
    assert encoded[DISEASES.index("Cardiomegaly")] == 1
    assert encoded[DISEASES.index("Effusion")] == 1
    assert set(np.flatnonzero(encoded)) == {
        DISEASES.index("Cardiomegaly"),
        DISEASES.index("Effusion"),
    }


def test_single_finding_and_order_independence():
    assert encode_labels("Hernia")[DISEASES.index("Hernia")] == 1
    np.testing.assert_array_equal(
        encode_labels("Effusion|Cardiomegaly"), encode_labels("Cardiomegaly|Effusion")
    )


@pytest.mark.parametrize("value", ["", "Cardiomegaly|Unknown", "No Finding|Effusion"])
def test_invalid_labels_are_rejected(value):
    with pytest.raises(ValueError):
        encode_labels(value)
