"""Tests for text preprocessing — critical because train-serve consistency depends on it."""

from sms_analysis.data.preprocessing import preprocess


class TestPreprocess:
    def test_html_url_placeholder(self):
        # NUS anonymisation token — presence of URL is the signal, map to __URL__
        assert preprocess("Check &lt;URL&gt; now") == "Check __URL__ now"

    def test_html_num_placeholder(self):
        assert preprocess("I sent &lt;#&gt; bucks") == "I sent __NUM__ bucks"

    def test_html_decimal_placeholder(self):
        assert preprocess("Amount: &lt;DECIMAL&gt;") == "Amount: __NUM__"

    def test_html_amp(self):
        assert preprocess("fish &amp; chips") == "fish  and  chips".strip() or \
               "and" in preprocess("fish &amp; chips")

    def test_real_url_https(self):
        result = preprocess("Visit http://win-prize.com today")
        assert "__URL__" in result
        assert "win-prize.com" not in result

    def test_real_url_www(self):
        result = preprocess("Go to www.example.com now")
        assert "__URL__" in result
        assert "www.example.com" not in result

    def test_standalone_numbers_replaced(self):
        result = preprocess("Call 08001234 to win")
        assert "__NUM__" in result
        assert "08001234" not in result

    def test_single_digit_preserved(self):
        # Single digits ("1 apple", "I am 1") are not replaced — not meaningful noise
        result = preprocess("I have 1 cat")
        assert "1" in result

    def test_plain_text_unchanged(self):
        text = "Hey, are you coming to the party?"
        assert preprocess(text) == text

    def test_idempotent(self):
        # Applying preprocess twice should give the same result as once
        text = "Win &lt;URL&gt; and &lt;#&gt; prizes!"
        assert preprocess(preprocess(text)) == preprocess(text)
