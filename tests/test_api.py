# This file contains pytest tests for FastAPI endpoints

import unittest
from fastapi.testclient import TestClient

import sys
import os

# add project root path
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from api.main import app

client = TestClient(app)


class TestAPI(unittest.TestCase):

    def test_home(self):

        response = client.get("/")

        self.assertEqual(response.status_code, 200)

    def test_predict(self):

        sample_data = {
            "Administrative": 2,
            "Administrative_Duration": 80.0,
            "Informational": 1,
            "Informational_Duration": 25.0,
            "ProductRelated": 15,
            "ProductRelated_Duration": 450.0,
            "BounceRates": 0.02,
            "ExitRates": 0.05,
            "PageValues": 25.5,
            "SpecialDay": 0.0,
            "Month": 5,
            "OperatingSystems": 2,
            "Browser": 2,
            "Region": 1,
            "TrafficType": 2,
            "VisitorType": 2,
            "Weekend": 1
        }

        response = client.post(
            "/predict",
            json=sample_data
        )

        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":

    unittest.main(verbosity=2)