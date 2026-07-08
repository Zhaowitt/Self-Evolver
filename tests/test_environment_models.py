"""Real per-test parsing for TestResult/TestCase (formerly always zero)."""

from src.environment import models

# Captured `pytest -rA` short-summary output: a header, a progress line, the
# "short test summary info" section, and the final tally.
RA_LOG = """============================= test session starts ==============================
platform linux -- Python 3.11.0, pytest-7.4.0, pluggy-1.3.0
collected 5 items

test_x.py .F.sE                                                          [100%]

=========================== short test summary info ============================
PASSED test_x.py::test_alpha
PASSED test_x.py::test_beta
FAILED test_x.py::test_gamma - assert 1 == 2
SKIPPED test_x.py::test_delta
ERROR test_x.py::test_epsilon
=================== 1 failed, 2 passed, 1 skipped, 1 error in 0.42s ============
"""

# Captured `pytest -v` verbose output (status suffixes with progress column).
VERBOSE_LOG = """test_x.py::test_a PASSED                                          [ 33%]
test_x.py::test_b FAILED                                          [ 66%]
test_x.py::test_c XFAIL                                           [100%]
==================== 1 failed, 1 passed, 1 xfailed in 0.10s ====================
"""

# Captured plain `pytest -q` output: only the tally line, no per-test lines.
QUIET_LOG = """....F
==================== 1 failed, 4 passed in 0.05s ====================
"""


def test_short_summary_counts_and_cases():
    r = models.TestResult.from_pytest_output(passed=False, output=RA_LOG)
    assert r.total_tests == 5
    assert r.passed_tests == 2
    assert r.failed_tests == 1
    assert r.error_tests == 1
    assert r.skipped_tests == 1
    # Per-test cases are populated (the bug was these being empty).
    names = {tc.name: tc.status for tc in r.test_cases}
    assert names["test_x.py::test_alpha"] == models.TestStatus.PASSED
    assert names["test_x.py::test_gamma"] == models.TestStatus.FAILED
    assert names["test_x.py::test_epsilon"] == models.TestStatus.ERROR
    assert names["test_x.py::test_delta"] == models.TestStatus.SKIPPED


def test_failed_test_names_include_errors():
    r = models.TestResult.from_pytest_output(passed=False, output=RA_LOG)
    assert r.failed_test_names == [
        "test_x.py::test_gamma",
        "test_x.py::test_epsilon",
    ]


def test_failed_case_carries_error_message():
    r = models.TestResult.from_pytest_output(passed=False, output=RA_LOG)
    gamma = next(tc for tc in r.test_cases if tc.name == "test_x.py::test_gamma")
    assert gamma.error_message == "assert 1 == 2"


def test_verbose_suffix_format_and_xfail_counts_as_pass():
    r = models.TestResult.from_pytest_output(passed=False, output=VERBOSE_LOG)
    assert r.total_tests == 3
    # xfailed counts as passed, matching swebench grading semantics.
    assert r.passed_tests == 2
    assert r.failed_tests == 1
    statuses = {tc.name: tc.status for tc in r.test_cases}
    assert statuses["test_x.py::test_c"] == models.TestStatus.PASSED


def test_quiet_tally_only_still_yields_real_counts():
    # No per-test lines, but the tally line is authoritative.
    r = models.TestResult.from_pytest_output(passed=False, output=QUIET_LOG)
    assert r.total_tests == 5
    assert r.passed_tests == 4
    assert r.failed_tests == 1
    assert r.test_cases == []


def test_colorized_output_is_parsed():
    # pytest colorizes when FORCE_COLOR/tty is set; ANSI must not defeat parsing.
    colored = (
        "\x1b[32mPASSED\x1b[0m test_x.py::test_a\n"
        "\x1b[31mFAILED\x1b[0m test_x.py::test_b - boom\n"
        "\x1b[32m1 passed\x1b[0m, \x1b[31m1 failed\x1b[0m in 0.01s\n"
    )
    r = models.TestResult.from_pytest_output(passed=False, output=colored)
    assert r.passed_tests == 1
    assert r.failed_tests == 1
    assert r.failed_test_names == ["test_x.py::test_b"]


def test_summary_string_reports_real_counts():
    r = models.TestResult.from_pytest_output(passed=False, output=RA_LOG)
    # Regression guard: summary used to always read "0/0 passed".
    assert "2/5 passed" in r.summary
    assert "0/0" not in r.summary
