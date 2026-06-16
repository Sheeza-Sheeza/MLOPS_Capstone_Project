import unittest
from flask_app.app import app

class FlaskAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_home_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<title>Sentiment Analysis</title>', response.data)

    def test_predict_page(self):
        response = self.client.post('/predict', data=dict(text="I love this!"))
        self.assertEqual(response.status_code, 200)
        response_lower = response.data.lower()
        self.assertTrue(
            b'positive' in response_lower or b'negative' in response_lower,
            "Response should contain either 'positive' or 'negative'",
        )

if __name__ == '__main__':
    unittest.main()