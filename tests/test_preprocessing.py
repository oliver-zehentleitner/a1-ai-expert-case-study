"""Tests for text preprocessing — critical because train-serve consistency depends on it."""

from sms_analysis.data.preprocessing import preprocess


class TestPreprocess:
    def test_html_url_placeholder(self):
        # NUS anonymisation token — presence of URL is the signal, map to TOKEN_URL
        assert preprocess("Check &lt;URL&gt; now") == "Check TOKEN_URL now"

    def test_html_num_placeholder(self):
        assert preprocess("I sent &lt;#&gt; bucks") == "I sent TOKEN_NUM bucks"

    def test_html_decimal_placeholder(self):
        assert preprocess("Amount: &lt;DECIMAL&gt;") == "Amount: TOKEN_NUM"

    def test_html_amp(self):
        assert preprocess("fish &amp; chips") == "fish  and  chips".strip() or \
               "and" in preprocess("fish &amp; chips")

    def test_real_url_https(self):
        result = preprocess("Visit http://win-prize.com today")
        assert "TOKEN_URL" in result
        assert "win-prize.com" not in result

    def test_real_url_www(self):
        result = preprocess("Go to www.example.com now")
        assert "TOKEN_URL" in result
        assert "www.example.com" not in result

    def test_standalone_numbers_replaced(self):
        result = preprocess("Call 08001234 to win")
        assert "TOKEN_NUM" in result
        assert "08001234" not in result

    def test_single_digit_replaced(self):
        # Single standalone digits are numbers too — consistent with multi-digit normalisation
        assert "TOKEN_NUM" in preprocess("I have 1 cat")
        assert "TOKEN_NUM" in preprocess("c u 2 night")
        assert "TOKEN_NUM" in preprocess("4 free tickets")

    def test_digit_in_alphanumeric_preserved(self):
        # \b prevents replacement inside alphanumeric tokens
        assert "hasi69" in preprocess("hasi69 is my username")
        assert "3a" in preprocess("flat 3a")

    def test_plain_text_unchanged(self):
        text = "Hey, are you coming to the party?"
        assert preprocess(text) == text

    def test_idempotent(self):
        # Applying preprocess twice should give the same result as once
        text = "Win &lt;URL&gt; and &lt;#&gt; prizes!"
        assert preprocess(preprocess(text)) == preprocess(text)

    def test_empty_string(self):
        assert preprocess("") == ""

    def test_whitespace_only(self):
        assert preprocess("   ") == ""

    def test_umlaut_normalised(self):
        assert preprocess("Ü coming?") == "u coming?"
        assert preprocess("ü coming?") == "u coming?"

    def test_currency_before_number(self):
        result = preprocess("Win £500 now")
        assert "TOKEN_CURRENCY" in result
        assert "TOKEN_NUM" in result
        assert "£" not in result
        assert "500" not in result

    def test_currency_iso_code(self):
        # ISO codes carry the same signal as symbols — locale-independent
        for code in ("EUR", "USD", "GBP", "CHF", "eur", "usd"):
            assert "TOKEN_CURRENCY" in preprocess(f"Send 100 {code} now")
        # embedded in longer word — must NOT match
        assert "TOKEN_CURRENCY" not in preprocess("Call EUROPE hotline")
        assert "TOKEN_CURRENCY" not in preprocess("GBPUSD pair")

    def test_html_time_placeholder(self):
        assert preprocess("Meet at &lt;TIME&gt;") == "Meet at TOKEN_TIME"

    def test_html_email_placeholder(self):
        assert preprocess("Reply to &lt;EMAIL&gt;") == "Reply to TOKEN_EMAIL"

    def test_mention_html(self):
        assert preprocess("follow @ &lt;URL&gt; ok") == "follow TOKEN_USER ok"

    def test_mention_raw(self):
        result = preprocess("@johnsmith said hi")
        assert "TOKEN_USER" in result
        assert "@johnsmith" not in result
