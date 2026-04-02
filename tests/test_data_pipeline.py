import pytest
from src.lib.medicine_data import clean_corpus, skip_download


# ---------------------------------------------------------------------------
# Sample corpus — one realistic medical-leaflet paragraph that exercises
# every replacement branch plus the known edge cases flagged in review.
# ---------------------------------------------------------------------------

SAMPLE_CORPUS = """
Paracetamol 500mg Tablets\\n
Store below 25°C (or 77°F).\\n
\\n
<b>Dosage</b>: Adults &amp; children &gt;12 years: 1–2 tablets every 4–6 hours.
Do not exceed 8 tablets in 24 hours.

Pharmacokinetics: Half-life ≈ 2 hours. Protein binding ≥ 10% but ≤ 30%.
Clearance rate: 5 ± 0.5 mL/min/kg.  Molecular weight: 151.16 g/mol.

Active metabolite: N-acetyl-p-benzoquinone imine (NAPQI).
Reaction: paracetamol × CYP2E1 → NAPQI (toxic if dose > therapeutic range).
Rate ÷ clearance ≠ elimination constant.  ΔG ≈ −33 kJ/mol.

Dose adjustment: renal impairment (CrCl < 30 mL/min) — reduce to ½ tablet.
Paediatric: ¼ to ¾ of adult dose (weight-based).
Area² = πr².  Volume³ reference only.

Side-effects include: α-adrenergic stimulation • headache · nausea … vomiting.
β-blocker interaction possible.  γ-glutamyl transferase may rise.
μ-opioid receptors not targeted.

Residual unicode escape (should be removed): \\u00e9 and \\u03B2.
"""


def test_literal_backslash_n_removed():
    assert "\\n" not in clean_corpus(SAMPLE_CORPUS)


def test_real_newline_removed():
    assert "\n" not in clean_corpus(SAMPLE_CORPUS)


def test_html_entities_unescaped():
    result = clean_corpus(SAMPLE_CORPUS)
    assert "&amp;" not in result
    assert "&gt;" not in result


def test_degree_celsius_lowercase():
    result = clean_corpus("Store below 25°c.")
    assert "degrees celsius" in result
    assert "°c" not in result


def test_degree_fahrenheit_lowercase():
    result = clean_corpus("Store below 77°f.")
    assert "degrees fahrenheit" in result


def test_degree_celsius_uppercase():
    # Known bug: "°C" (uppercase) is NOT currently handled.
    # This test documents the gap — update once fixed.
    result = clean_corpus("Store below 25°C.")
    assert "°C" not in result, (
        "°C (uppercase) is not replaced — fix clean_corpus or add uppercase variant"
    )


def test_math_symbols():
    cases = [
        ("≥ 10%", "greater than or equal"),
        ("≤ 30%", "less than or equal"),
        ("5 ± 0.5", "plus minus"),
        ("2 × 3", "times"),
        ("6 ÷ 2", "divided by"),
        ("≈ 2 hours", "approximately"),
        ("rate ≠ constant", "not equal"),
    ]
    for symbol_text, expected_word in cases:
        assert expected_word in clean_corpus(symbol_text), f"Failed for: {symbol_text!r}"


def test_gt_lt_from_html_entities():
    # &gt; and &lt; come from HTML; html.unescape converts them first,
    # then the > / < replacements should catch them.
    result = clean_corpus("children &gt;12 years and dose &lt;500mg")
    assert "greater than" in result
    assert "less than" in result


def test_en_dash_and_em_dash_replaced_with_space():
    assert "–" not in clean_corpus("1–2 tablets")
    assert "—" not in clean_corpus("reduce — carefully")


def test_bullet_and_ellipsis_replaced():
    assert "•" not in clean_corpus("side-effects • nausea")
    assert "…" not in clean_corpus("vomiting…")


def test_greek_letters():
    cases = [
        ("α-adrenergic", "alpha"),
        ("β-blocker", "beta"),
        ("γ-glutamyl", "gamma"),
        ("μ-opioid", "micro"),
    ]
    for symbol_text, expected_word in cases:
        assert expected_word in clean_corpus(symbol_text), f"Failed for: {symbol_text!r}"


def test_fractions():
    assert "one half" in clean_corpus("½ tablet")
    assert "one quarter" in clean_corpus("¼ dose")
    assert "three quarters" in clean_corpus("¾ dose")


def test_superscripts():
    assert "squared" in clean_corpus("Area²")
    assert "cubed" in clean_corpus("Volume³")


def test_residual_unicode_escapes_removed():
    result = clean_corpus("residual \\u00e9 escape \\u03B2 here")
    assert "\\u00e9" not in result
    assert "\\u03B2" not in result


def test_no_double_spaces():
    result = clean_corpus(SAMPLE_CORPUS)
    assert "  " not in result


def test_no_leading_trailing_whitespace():
    result = clean_corpus("  leading and trailing  ")
    assert result == result.strip()


def test_empty_string():
    assert clean_corpus("") == ""


def test_plain_text_unchanged_structure():
    result = clean_corpus("Take one tablet daily with water.")
    assert result == "Take one tablet daily with water.".lower()
