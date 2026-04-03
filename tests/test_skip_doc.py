from src.lib.medicine_data import skip_download

EMA_NUMBER = "EMEA/H/C/000001"

BASE_METADATA = {
    EMA_NUMBER: {
        "status": "Authorised",
        "category": "Human",
        "updated_at": "2024-01-10T00:00:00",
    }
}

BASE_RAW = {"last_updated_date": "15/01/2024"}

def test_unknown_ema_number_returns_false():
    assert skip_download("EMEA/H/C/UNKNOWN", BASE_RAW, BASE_METADATA) is False


def test_non_authorised_status_skips():
    metadata = {EMA_NUMBER: {**BASE_METADATA[EMA_NUMBER], "status": "Withdrawn"}}
    assert skip_download(EMA_NUMBER, BASE_RAW, metadata) is True


def test_non_human_category_skips():
    metadata = {EMA_NUMBER: {**BASE_METADATA[EMA_NUMBER], "category": "Veterinary"}}
    assert skip_download(EMA_NUMBER, BASE_RAW, metadata) is True


def test_missing_updated_at_returns_false():
    metadata = {EMA_NUMBER: {**BASE_METADATA[EMA_NUMBER], "updated_at": None}}
    assert skip_download(EMA_NUMBER, BASE_RAW, metadata) is True


def test_empty_updated_at_returns_false():
    metadata = {EMA_NUMBER: {**BASE_METADATA[EMA_NUMBER], "updated_at": ""}}
    assert skip_download(EMA_NUMBER, BASE_RAW, metadata) is False

def test_missing_last_updated_date_skips():
    raw = {}
    assert skip_download(EMA_NUMBER, raw, BASE_METADATA) is True


def test_empty_last_updated_date_skips():
    raw = {"last_updated_date": "   "}
    assert skip_download(EMA_NUMBER, raw, BASE_METADATA) is True


def test_source_older_than_local_skips():
    # local = 2024-01-10, source = 2024-01-05 → source < local+1 → skip
    metadata = {EMA_NUMBER: {**BASE_METADATA[EMA_NUMBER], "updated_at": "2024-01-10T00:00:00"}}
    raw = {"last_updated_date": "05/01/2024"}
    assert skip_download(EMA_NUMBER, raw, metadata) is True


def test_source_same_day_as_local_skips():
    # local = 2024-01-10, source = 2024-01-10 → source < local+1 → skip
    metadata = {EMA_NUMBER: {**BASE_METADATA[EMA_NUMBER], "updated_at": "2024-01-10T00:00:00"}}
    raw = {"last_updated_date": "10/01/2024"}
    assert skip_download(EMA_NUMBER, raw, metadata) is True


def test_source_newer_than_local_does_not_skip():
    # local = 2024-01-10, source = 2024-01-15 → source >= local+1 → don't skip
    metadata = {EMA_NUMBER: {**BASE_METADATA[EMA_NUMBER], "updated_at": "2024-01-10T00:00:00"}}
    raw = {"last_updated_date": "15/01/2024"}
    assert skip_download(EMA_NUMBER, raw, metadata) is False


def test_source_one_day_after_local_does_not_skip():
    # boundary: local = 2024-01-10, source = 2024-01-11 → source == local+1 → don't skip
    metadata = {EMA_NUMBER: {**BASE_METADATA[EMA_NUMBER], "updated_at": "2024-01-10T00:00:00"}}
    raw = {"last_updated_date": "11/01/2024"}
    assert skip_download(EMA_NUMBER, raw, metadata) is False


# ---------------------------------------------------------------------------
# invalid date formats → ValueError → return False
# ---------------------------------------------------------------------------

def test_invalid_updated_at_format_returns_false():
    metadata = {EMA_NUMBER: {**BASE_METADATA[EMA_NUMBER], "updated_at": "not-a-date"}}
    raw = {"last_updated_date": "15/01/2024"}
    assert skip_download(EMA_NUMBER, raw, metadata) is False


def test_invalid_last_updated_date_format_returns_false():
    metadata = {EMA_NUMBER: {**BASE_METADATA[EMA_NUMBER], "updated_at": "2024-01-10T00:00:00"}}
    raw = {"last_updated_date": "2024-01-15"}  # wrong format (ISO, not DD/MM/YYYY)
    assert skip_download(EMA_NUMBER, raw, metadata) is False
