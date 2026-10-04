"""How an epoch's numbers are printed.

A small thing with a sharp edge: `learning_rate` was formatted to four decimal places like
every other quantity, so a cosine schedule decaying 1e-4 to 1e-8 printed
`learning_rate=0.0000` from epoch 90 onwards and a run with a schedule looked exactly like
a run without one. The whole point of adding the schedule was to make that visible, and
the output hid it.

Found by watching a real run and trying to confirm the rate was moving, which is the only
way this kind of thing is found - no test would have asked.
"""

from pyplatypus.training.trainer import format_logs


def test_a_small_learning_rate_is_still_legible():
    text = format_logs({"train_loss": 5.9335, "learning_rate": 1.097e-08})
    assert "learning_rate=1.1e-08" in text
    assert "learning_rate=0.0000" not in text


def test_an_ordinary_learning_rate_reads_normally():
    assert "learning_rate=0.0001" in format_logs({"learning_rate": 1e-4})


def test_the_rates_of_a_decaying_schedule_are_all_distinguishable():
    """The actual requirement. Five points down a 150-epoch cosine from 1e-4 printed the
    same string for three of them before this."""
    import math

    shown = [
        format_logs({"learning_rate": 1e-4 * (1 + math.cos(math.pi * e / 150)) / 2})
        for e in (0, 33, 66, 99, 132, 149)
    ]
    assert len(set(shown)) == len(shown), shown


def test_other_quantities_keep_four_places():
    text = format_logs({"train_loss": 5.9335, "val_dice": 0.123456})
    assert "train_loss=5.9335" in text
    assert "val_dice=0.1235" in text


def test_seconds_is_left_out_because_it_is_printed_separately():
    assert "seconds" not in format_logs({"train_loss": 1.0, "seconds": 16.5})
