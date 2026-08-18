from src.data_task.data_ingestion import get_aws_credentials
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_get_aws_credentials_prefers_standard_names(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")
    monkeypatch.delenv("Access_Key", raising=False)
    monkeypatch.delenv("Secret_Access_Key", raising=False)

    access_key, secret_key = get_aws_credentials()

    assert access_key == "test-access-key"
    assert secret_key == "test-secret-key"


def test_get_aws_credentials_raises_when_missing(monkeypatch):
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("Access_Key", raising=False)
    monkeypatch.delenv("Secret_Access_Key", raising=False)

    try:
        get_aws_credentials()
        assert False, "Expected ValueError when AWS credentials are missing"
    except ValueError:
        pass
