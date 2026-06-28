import unittest

from harnessgym.timeouts import parse_timeout


class TimeoutTests(unittest.TestCase):
    def test_parse_timeout_units(self) -> None:
        self.assertEqual(parse_timeout("45m"), 2700)
        self.assertEqual(parse_timeout("1h30m"), 5400)
        self.assertEqual(parse_timeout("2m10s"), 130)
        self.assertEqual(parse_timeout("90"), 90)
        self.assertEqual(parse_timeout(5), 5)

    def test_parse_timeout_rejects_invalid_values(self) -> None:
        for value in ["", "abc", "1x", "1m 2s", 0, -1]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_timeout(value)
