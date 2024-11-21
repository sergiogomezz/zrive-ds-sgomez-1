import requests
import pandas as pd
import unittest
from unittest.mock import patch, Mock
import pytest
from src.module_1.module_1_meteo_api import (_request_with_cooloff, compute_monthly_statistics)

def test_compute_monthly_statistics():
    test_variable = "test_variable"
    data = pd.DataFrame(
        {
            "city": ["Madrid"] * 6,
            "time": pd.date_range(start="2021-01-01", periods=6),
            f"{test_variable}": [10,20,30,15,25,35],
        }
    )
    expected = pd.DataFrame(
        {
            "city": ["Madrid"],
            "month": pd.to_datetime(["2021-01-01"]),
            f"{test_variable}_max": [35.0],
            f"{test_variable}_mean": [22.5],
            f"{test_variable}_min": [10.0],
            f"{test_variable}_std": [9.354143],
        }
    )

    actual = compute_monthly_statistics(data, [test_variable])
    pd.testing.assert_frame_equal(
        actual, expected, check_dtype=False, check_index_type=False
    )


class MockResponse:
    def __init__(self, json_data, status_code):
        self.json_data = json_data
        self.status_code = status_code

    def json(self):
        return self.json_data

    def raise_for_status(self):
        if self.status_code != 200:
            raise requests.exceptions.HTTPError(f"HTTPError: {self.status_code}")


def test_request_with_cooloff_200(monkeypatch):
    headers = {}
    mocked_response = Mock(return_value=MockResponse("json_dummy", 200))
    monkeypatch.setattr(requests, "get", mocked_response)
    response = _request_with_cooloff("mock_url", headers, num_attemps=10, payload=None)
    assert response.status_code == 200
    assert response.json() == "json_dummy"


def test_request_with_cooloff_404(monkeypatch):
    with pytest.raises(requests.exceptions.HTTPError):
        headers = {}
        mocked_response = Mock(return_value=MockResponse("mocked_response", 404))
        monkeypatch.setattr(requests, "get", mocked_response)
        _ = _request_with_cooloff("mock_url", headers, num_attemps=10, payload=None)


def test_request_with_cooloff_429(monkeypatch):
    with pytest.raises(requests.exceptions.HTTPError):
        headers = {}
        mocked_response = Mock(return_value=MockResponse("mocked_response", 429))
        monkeypatch.setattr(requests, "get", mocked_response)
        response = _request_with_cooloff(
            "mock_url", headers, num_attemps=10, payload=None
        )
        expected_msgs = [
            f"API return code {response.status_code} cooloff at {1}"
            f"API return code {response.status_code} cooloff at {2}"
            f"API return code {response.status_code} cooloff at {4}"
        ]
        assert [r.msg for r in caplog.records] == expected_msgs