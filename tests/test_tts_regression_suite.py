from services.tts_regression_suite import TTSRegressionSuite


def test_regression_suite_runs_fixed_sentences():
    suite = TTSRegressionSuite(sentences=["Hello world."])

    run = suite.run(lambda sentence: b"\x00\x00\x00\x00")

    assert len(run["results"]) == 1
    assert run["results"][0]["sentence"] == "Hello world."


def test_regression_suite_compares_current_against_previous():
    suite = TTSRegressionSuite(sentences=["Hello world."])
    first = suite.run(lambda sentence: b"\x00\x00\x00\x00")
    second = suite.run(lambda sentence: b"\x00\x00\x00\x00")

    comparison = suite.compare(second, first)

    assert comparison["compared"] is True
    assert comparison["changed"] >= 0
