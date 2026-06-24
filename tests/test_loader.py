"""Tests for the data loading layer."""

from pathlib import Path

import pytest

from sms_analysis.data.loader import Label, SMSDataset, SMSMessage, load_dataset

DATASET_PATH = Path("assignment/3D8026C4.zip/SMSSpamCollection")


class TestSMSMessage:
    def test_is_spam_true(self):
        msg = SMSMessage(label=Label.SPAM, text="Win a prize now!")
        assert msg.is_spam is True

    def test_is_spam_false(self):
        msg = SMSMessage(label=Label.HAM, text="Hey, are you coming?")
        assert msg.is_spam is False

    def test_empty_text_raises(self):
        with pytest.raises(Exception):
            SMSMessage(label=Label.HAM, text="   ")


class TestSMSDataset:
    @pytest.fixture
    def dataset(self):
        return SMSDataset(
            messages=[
                SMSMessage(label=Label.HAM, text="Hello"),
                SMSMessage(label=Label.HAM, text="Good morning"),
                SMSMessage(label=Label.SPAM, text="Win a prize"),
            ]
        )

    def test_len(self, dataset):
        assert len(dataset) == 3

    def test_ham_filter(self, dataset):
        assert len(dataset.ham) == 2

    def test_spam_filter(self, dataset):
        assert len(dataset.spam) == 1

    def test_texts(self, dataset):
        assert dataset.texts == ["Hello", "Good morning", "Win a prize"]

    def test_labels(self, dataset):
        assert dataset.labels == ["ham", "ham", "spam"]

    def test_to_dataframe_shape(self, dataset):
        df = dataset.to_dataframe()
        assert df.shape == (3, 2)
        assert list(df.columns) == ["label", "text"]


@pytest.mark.skipif(
    not DATASET_PATH.exists(),
    reason="Dataset file not present",
)
class TestLoadDataset:
    def test_loads_expected_count(self):
        ds = load_dataset(DATASET_PATH)
        # quoting=QUOTE_NONE ensures all 5574 lines are parsed regardless of quote characters
        assert len(ds) == 5574

    def test_class_balance(self):
        ds = load_dataset(DATASET_PATH)
        assert len(ds.spam) == 747
        assert len(ds.ham) == len(ds) - 747

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_dataset(Path("nonexistent/file"))
