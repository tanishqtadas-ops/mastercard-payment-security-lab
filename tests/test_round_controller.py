"""
tests/test_round_controller.py — Focused unit tests for Phase 3.

Tests verify the Round Controller's orchestration behaviour using simple
inline fakes/doubles.  No real ML models, no family-specific logic.
All tests are deterministic.
"""

import pytest
from schemas import (
    AttackEvent,
    AttackFamily,
    PredictionResult,
    BlueTeamFeedback,
    RoundResult,
)
from simulation import RoundController, RoundControllerError
from simulation.interfaces import AttackGenerator, BlueTeamDetector, FeedbackEvaluator


# ---------------------------------------------------------------------------
# Shared test doubles
# ---------------------------------------------------------------------------

def _make_attack_event(round_id: str = "r_test") -> AttackEvent:
    return AttackEvent(
        attack_id="a_001",
        round_id=round_id,
        attack_family=AttackFamily.ADAPTIVE_EVASION,
        attack_genome={"amount_deviation": 0.7, "velocity_deviation": 0.4},
        scenario={"channel": "online"},
        ground_truth=True,
    )


def _make_prediction(detected: bool = True, risk_score: float = 0.85) -> PredictionResult:
    return PredictionResult(
        prediction_id="p_001",
        prediction=detected,
        risk_score=risk_score,
        model_version="v_test",
    )


def _make_feedback(
    round_id: str = "r_test",
    detected: bool = True,
    false_negative: bool = False,
) -> BlueTeamFeedback:
    return BlueTeamFeedback(
        feedback_id="fb_001",
        round_reference=round_id,
        detected=detected,
        false_positive=False,
        false_negative=false_negative,
        risk_score=0.85,
        important_features={"amount_deviation": 0.6},
    )


class _FakeGenerator:
    """Returns a pre-built AttackEvent; records the round_id it received."""

    def __init__(self, event: AttackEvent | None = None):
        self._event = event or _make_attack_event()
        self.called_with: list[str] = []

    def generate(self, round_id: str) -> AttackEvent:
        self.called_with.append(round_id)
        return self._event


class _FakeDetector:
    """Returns a pre-built PredictionResult; records the event it received."""

    def __init__(self, prediction: PredictionResult | None = None):
        self._prediction = prediction or _make_prediction()
        self.received_events: list[AttackEvent] = []

    def detect(self, event: AttackEvent) -> PredictionResult:
        self.received_events.append(event)
        return self._prediction


class _FakeFeedbackEvaluator:
    """Returns a pre-built BlueTeamFeedback; records inputs it received."""

    def __init__(self, feedback: BlueTeamFeedback | None = None):
        self._feedback = feedback or _make_feedback()
        self.received: list[tuple] = []

    def evaluate(self, event: AttackEvent, prediction: PredictionResult) -> BlueTeamFeedback:
        self.received.append((event, prediction))
        return self._feedback


def _make_controller(
    generator=None,
    detector=None,
    evaluator=None,
) -> tuple[RoundController, _FakeGenerator, _FakeDetector, _FakeFeedbackEvaluator]:
    gen = generator or _FakeGenerator()
    det = detector or _FakeDetector()
    ev = evaluator or _FakeFeedbackEvaluator()
    ctrl = RoundController(gen, det, ev)
    return ctrl, gen, det, ev


# ---------------------------------------------------------------------------
# Test 1 — A successful round executes the complete pipeline in order
# ---------------------------------------------------------------------------

def test_successful_round_returns_round_result():
    ctrl, gen, det, ev = _make_controller()
    result = ctrl.run_round(round_id="r_1")

    assert isinstance(result, RoundResult)
    assert result.round_id == "r_1"
    assert isinstance(result.attack_event, AttackEvent)
    assert isinstance(result.prediction_result, PredictionResult)
    assert isinstance(result.feedback, BlueTeamFeedback)


# ---------------------------------------------------------------------------
# Test 2 — AttackGenerator is invoked with the correct round_id
# ---------------------------------------------------------------------------

def test_attack_generator_is_invoked():
    ctrl, gen, det, ev = _make_controller()
    ctrl.run_round(round_id="round_xyz")

    assert gen.called_with == ["round_xyz"]


# ---------------------------------------------------------------------------
# Test 3 — The AttackEvent from the generator reaches the BlueTeamDetector
# ---------------------------------------------------------------------------

def test_attack_event_reaches_detector():
    expected_event = _make_attack_event()
    gen = _FakeGenerator(event=expected_event)
    det = _FakeDetector()
    ev = _FakeFeedbackEvaluator()
    ctrl = RoundController(gen, det, ev)

    ctrl.run_round(round_id="r_1")

    assert len(det.received_events) == 1
    assert det.received_events[0] is expected_event


# ---------------------------------------------------------------------------
# Test 4 — The PredictionResult from the detector reaches the FeedbackEvaluator
# ---------------------------------------------------------------------------

def test_prediction_reaches_feedback_evaluator():
    expected_prediction = _make_prediction(detected=False, risk_score=0.2)
    gen = _FakeGenerator()
    det = _FakeDetector(prediction=expected_prediction)
    ev = _FakeFeedbackEvaluator()
    ctrl = RoundController(gen, det, ev)

    ctrl.run_round(round_id="r_1")

    assert len(ev.received) == 1
    _, received_prediction = ev.received[0]
    assert received_prediction is expected_prediction


# ---------------------------------------------------------------------------
# Test 5 — Feedback is incorporated into the resulting RoundResult
# ---------------------------------------------------------------------------

def test_feedback_in_round_result():
    expected_feedback = _make_feedback(detected=False, false_negative=True)
    ev = _FakeFeedbackEvaluator(feedback=expected_feedback)
    ctrl, gen, det, _ = _make_controller(evaluator=ev)

    result = ctrl.run_round(round_id="r_1")

    assert result.feedback is expected_feedback
    assert result.feedback.detected is False
    assert result.feedback.false_negative is True


# ---------------------------------------------------------------------------
# Test 6a — Detected attack completes the round lifecycle correctly
# ---------------------------------------------------------------------------

def test_detected_attack_round():
    attack = _make_attack_event()
    prediction = _make_prediction(detected=True, risk_score=0.9)
    feedback = _make_feedback(detected=True, false_negative=False)

    ctrl = RoundController(
        _FakeGenerator(event=attack),
        _FakeDetector(prediction=prediction),
        _FakeFeedbackEvaluator(feedback=feedback),
    )
    result = ctrl.run_round(round_id="det_round")

    assert result.prediction_result.prediction is True
    assert result.feedback.detected is True
    assert result.feedback.false_negative is False


# ---------------------------------------------------------------------------
# Test 6b — Missed attack completes the round lifecycle correctly
# ---------------------------------------------------------------------------

def test_missed_attack_round():
    attack = _make_attack_event()
    prediction = _make_prediction(detected=False, risk_score=0.1)
    feedback = _make_feedback(detected=False, false_negative=True)

    ctrl = RoundController(
        _FakeGenerator(event=attack),
        _FakeDetector(prediction=prediction),
        _FakeFeedbackEvaluator(feedback=feedback),
    )
    result = ctrl.run_round(round_id="miss_round")

    assert result.prediction_result.prediction is False
    assert result.feedback.detected is False
    assert result.feedback.false_negative is True


# ---------------------------------------------------------------------------
# Test 7 — Controller does not depend on concrete Family 1/2/3 implementations
# ---------------------------------------------------------------------------

def test_controller_works_with_all_attack_families():
    """
    Run the controller with each attack family to confirm the controller
    never branches on family identity.
    """
    for family in AttackFamily:
        attack = AttackEvent(
            attack_id="a_fam",
            round_id="r_fam",
            attack_family=family,
            attack_genome={"dim": 0.5},
            scenario={},
            ground_truth=True,
        )
        ctrl = RoundController(
            _FakeGenerator(event=attack),
            _FakeDetector(),
            _FakeFeedbackEvaluator(),
        )
        result = ctrl.run_round(round_id="r_fam")
        assert result.attack_event.attack_family == family


# ---------------------------------------------------------------------------
# Test 8 — Dependencies can be replaced with simple test doubles (protocol check)
# ---------------------------------------------------------------------------

def test_dependencies_satisfy_protocols():
    """
    Verify that the fakes declared above satisfy the Protocol checks at
    runtime, confirming the interfaces are correctly defined and the fakes
    are valid implementations.
    """
    gen = _FakeGenerator()
    det = _FakeDetector()
    ev = _FakeFeedbackEvaluator()

    assert isinstance(gen, AttackGenerator)
    assert isinstance(det, BlueTeamDetector)
    assert isinstance(ev, FeedbackEvaluator)


# ---------------------------------------------------------------------------
# Test 9 — Invalid / missing dependency types raise RoundControllerError
# ---------------------------------------------------------------------------

def test_invalid_generator_type_raises():
    with pytest.raises(RoundControllerError, match="generator"):
        RoundController(
            generator=object(),      # not an AttackGenerator
            detector=_FakeDetector(),
            evaluator=_FakeFeedbackEvaluator(),
        )


def test_invalid_detector_type_raises():
    with pytest.raises(RoundControllerError, match="detector"):
        RoundController(
            generator=_FakeGenerator(),
            detector=object(),        # not a BlueTeamDetector
            evaluator=_FakeFeedbackEvaluator(),
        )


def test_invalid_evaluator_type_raises():
    with pytest.raises(RoundControllerError, match="evaluator"):
        RoundController(
            generator=_FakeGenerator(),
            detector=_FakeDetector(),
            evaluator=object(),       # not a FeedbackEvaluator
        )


def test_generator_exception_wrapped():
    """If the generator raises, RoundControllerError is raised."""
    class _BrokenGenerator:
        def generate(self, round_id: str) -> AttackEvent:
            raise RuntimeError("generator broken")

    ctrl = RoundController(_BrokenGenerator(), _FakeDetector(), _FakeFeedbackEvaluator())
    with pytest.raises(RoundControllerError, match="generator broken"):
        ctrl.run_round(round_id="r_1")


def test_detector_exception_wrapped():
    """If the detector raises, RoundControllerError is raised."""
    class _BrokenDetector:
        def detect(self, event: AttackEvent) -> PredictionResult:
            raise RuntimeError("detector broken")

    ctrl = RoundController(_FakeGenerator(), _BrokenDetector(), _FakeFeedbackEvaluator())
    with pytest.raises(RoundControllerError, match="detector broken"):
        ctrl.run_round(round_id="r_1")


def test_evaluator_exception_wrapped():
    """If the evaluator raises, RoundControllerError is raised."""
    class _BrokenEvaluator:
        def evaluate(self, event, prediction):
            raise RuntimeError("evaluator broken")

    ctrl = RoundController(_FakeGenerator(), _FakeDetector(), _BrokenEvaluator())
    with pytest.raises(RoundControllerError, match="evaluator broken"):
        ctrl.run_round(round_id="r_1")


def test_generator_wrong_return_type_raises():
    """If the generator returns the wrong type, RoundControllerError is raised."""
    class _BadGenerator:
        def generate(self, round_id: str):
            return {"not": "an AttackEvent"}

    ctrl = RoundController(_BadGenerator(), _FakeDetector(), _FakeFeedbackEvaluator())
    with pytest.raises(RoundControllerError, match="AttackEvent"):
        ctrl.run_round(round_id="r_1")


def test_detector_wrong_return_type_raises():
    """If the detector returns the wrong type, RoundControllerError is raised."""
    class _BadDetector:
        def detect(self, event: AttackEvent):
            return "not a PredictionResult"

    ctrl = RoundController(_FakeGenerator(), _BadDetector(), _FakeFeedbackEvaluator())
    with pytest.raises(RoundControllerError, match="PredictionResult"):
        ctrl.run_round(round_id="r_1")


def test_evaluator_wrong_return_type_raises():
    """If the evaluator returns the wrong type, RoundControllerError is raised."""
    class _BadEvaluator:
        def evaluate(self, event, prediction):
            return 42

    ctrl = RoundController(_FakeGenerator(), _FakeDetector(), _BadEvaluator())
    with pytest.raises(RoundControllerError, match="BlueTeamFeedback"):
        ctrl.run_round(round_id="r_1")


# ---------------------------------------------------------------------------
# Miscellaneous
# ---------------------------------------------------------------------------

def test_round_id_auto_generated_if_omitted():
    ctrl, _, _, _ = _make_controller()
    result = ctrl.run_round()          # no round_id supplied
    assert result.round_id             # should be a non-empty UUID string
    assert len(result.round_id) > 0


def test_outcome_metrics_forwarded():
    ctrl, _, _, _ = _make_controller()
    metrics = {"wall_time_ms": 42, "round_index": 3}
    result = ctrl.run_round(round_id="r_m", outcome_metrics=metrics)
    assert result.outcome_metrics == metrics
