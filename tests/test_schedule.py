"""Tests for completion_core/schedule.py: the pure description of how the learning rate changes per epoch."""
from __future__ import annotations

import re

import pytest

from completion_core.schedule import format_lr_schedule, lr_factor, parse_lr_schedule


# ---- parsing: only one canonical spelling is accepted, so a string in a spec, a command and a protocol means one thing ----

def test_parse_one_milestone():
    assert parse_lr_schedule("step:2000:0.1") == {"type": "step", "milestones": [2000], "gamma": 0.1}


def test_parse_two_milestones():
    assert parse_lr_schedule("step:1000,2000:0.5") == {"type": "step", "milestones": [1000, 2000], "gamma": 0.5}


@pytest.mark.parametrize("text", ["step:2000:0.1", "step:1000,2000:0.5", "step:1:1.0", "step:10,20,30:0.25"])
def test_format_is_the_inverse_of_parse(text):
    assert format_lr_schedule(parse_lr_schedule(text)) == text


@pytest.mark.parametrize(
    "text",
    [
        "", "step", "step:2000", "step:2000:0.1:extra", "cosine:2000:0.1", "STEP:2000:0.1",
        "step:0:0.1", "step:-1:0.1", "step:02000:0.1", "step:2000,2000:0.1", "step:3000,2000:0.1",
        "step: 2000:0.1", "step:2000 :0.1", "step:2000,:0.1", "step:,2000:0.1", "step:2000.5:0.1",
        "step:2000:abc", "step:2000:nan", "step:2000:inf", "step:2000:0", "step:2000:0.0", "step:2000:-0.0",
        "step:2000:-0.1", "step:2000:1.5",
        "step:2000:1e-1", "step:2000:0.10", "step:2000:1", "step:2000:.1",
    ],
)
def test_parse_rejects_everything_that_is_not_the_canonical_spelling(text):
    with pytest.raises(ValueError):
        parse_lr_schedule(text)


@pytest.mark.parametrize("value", [None, 2000, 0.1, ["step:2000:0.1"], b"step:2000:0.1"])
def test_parse_rejects_non_strings(value):
    with pytest.raises(ValueError):
        parse_lr_schedule(value)


def test_gamma_of_one_is_allowed_and_changes_nothing():
    schedule = parse_lr_schedule("step:5:1.0")
    assert [lr_factor(schedule, epoch) for epoch in (1, 5, 6, 100)] == [1.0, 1.0, 1.0, 1.0]


# ---- lr_factor: epochs are 1-based, and the milestone epoch itself still trains at the full rate -------------------------

def test_one_milestone_the_milestone_epoch_is_not_decayed():
    schedule = parse_lr_schedule("step:2000:0.1")
    assert lr_factor(schedule, 1) == 1.0
    assert lr_factor(schedule, 1999) == 1.0
    assert lr_factor(schedule, 2000) == 1.0
    assert lr_factor(schedule, 2001) == 0.1
    assert lr_factor(schedule, 4000) == 0.1


def test_two_milestones_compound():
    schedule = parse_lr_schedule("step:1000,2000:0.5")
    assert [lr_factor(schedule, epoch) for epoch in (1, 1000, 1001, 2000, 2001, 4000)] == [1.0, 1.0, 0.5, 0.5, 0.25, 0.25]


def test_epochs_up_to_the_first_milestone_leave_the_rate_bit_identical():
    schedule = parse_lr_schedule("step:2000:0.1")
    base = 0.001
    assert all(base * lr_factor(schedule, epoch) == base for epoch in range(1, 2001))


@pytest.mark.parametrize("epoch", [0, -1, 1.5, "3", None, True, False])
def test_lr_factor_rejects_bad_epochs(epoch):
    with pytest.raises(ValueError):
        lr_factor(parse_lr_schedule("step:2000:0.1"), epoch)


# ---- added in E7.1a after the planted-bug harness: each refusal also says what is wrong (the text reaches the user) ----------

@pytest.mark.parametrize(
    "text,message",
    [
        (None, "lr schedule must be a string"),
        ("cosine:2000:0.1", "lr schedule must look like step:2000:0.1, got 'cosine:2000:0.1'"),
        ("step:2000", "lr schedule must look like step:2000:0.1, got 'step:2000'"),
        ("step:0:0.1", "lr schedule milestones must be positive integers separated by commas, got '0'"),
        ("step:3000,2000:0.1", "lr schedule milestones must be strictly increasing"),
        ("step:2000,2000:0.1", "lr schedule milestones must be strictly increasing"),
        ("step:2000:abc", "lr schedule gamma must be a number, got 'abc'"),
        ("step:2000:1.5", "lr schedule gamma must be a finite number in (0, 1]"),
        ("step:2000:0.0", "lr schedule gamma must be a finite number in (0, 1]"),
        ("step:2000:nan", "lr schedule gamma must be a finite number in (0, 1]"),
        ("step:2000:0.10", "lr schedule must be written canonically as 'step:2000:0.1', got 'step:2000:0.10'"),
    ],
)
def test_each_refusal_says_what_is_wrong(text, message):
    with pytest.raises(ValueError, match="^" + re.escape(message) + "$"):      # the whole text: a search would miss a changed start
        parse_lr_schedule(text)


def test_lr_factor_refusal_says_what_is_wrong():
    with pytest.raises(ValueError, match="^" + re.escape("epoch must be an integer >= 1") + "$"):
        lr_factor(parse_lr_schedule("step:2000:0.1"), 0)
