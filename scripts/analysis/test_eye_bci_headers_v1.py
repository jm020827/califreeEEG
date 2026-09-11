"""One bounded non-network reader unit suite."""

import unittest

from probe_eye_bci_headers_v1 import allowed_url, parse_header


class HeaderTests(unittest.TestCase):
    def test_expected_headers(self):
        self.assertEqual(
            parse_header(b"Time,FP1,OZ,Trig\n", "Neuroscan"), ["Time", "FP1", "OZ", "Trig"]
        )
        self.assertEqual(
            parse_header(b"Time\tValidityLeft\tValidityRight\n", "Tobii"),
            ["Time", "ValidityLeft", "ValidityRight"],
        )

    def test_numeric_is_not_header(self):
        with self.assertRaisesRegex(ValueError, "expected_header"):
            parse_header(b"0.1,2,3\n", "Tobii")

    def test_size_and_unterminated(self):
        for line in (b"", b"Time,FP1", b"a" * 65537 + b"\n"):
            with self.assertRaisesRegex(ValueError, "boundary"):
                parse_header(line, "Neuroscan")

    def test_url_allowlist(self):
        self.assertTrue(allowed_url("https://repo-prod.prod.sagebase.org/repo/v1/entity/syn1"))
        self.assertTrue(allowed_url("https://bucket.s3.amazonaws.com/object?signature=redacted"))
        for url in (
            "http://bucket.s3.amazonaws.com/a",
            "https://user:pass@bucket.s3.amazonaws.com/a",
            "https://amazonaws.com.evil.example/a",
            "file:///etc/passwd",
        ):
            self.assertFalse(allowed_url(url))


if __name__ == "__main__":
    unittest.main()
