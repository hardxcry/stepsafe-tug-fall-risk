import numpy as np
import pytest

from tugrisk.pose import L_HIP, R_HIP, PoseSequence, select_subject
from tugrisk.simulate import Protocol, random_protocol, simulate
from tugrisk.tug import PHASES, STEADI_CUTOFF_S, TugError, analyse, posture_signal


def test_total_time_within_one_second_clean():
    p = Protocol()
    r = analyse(simulate(p))
    assert abs(r.total_time - p.total) < 1.0
    assert set(r.phases) == set(PHASES)


@pytest.mark.parametrize("fps,noise,drop", [(30, 0.02, 0.05), (15, 0.04, 0.10)])
def test_robust_to_noise_dropout_and_frame_rate(fps, noise, drop):
    p = Protocol(walk_out=4, walk_back=4)
    r = analyse(simulate(p, fps=fps, noise_m=noise, dropout=drop, seed=3))
    assert abs(r.total_time - p.total) < 1.5


def test_phases_are_ordered_and_contiguous():
    r = analyse(simulate(Protocol(), noise_m=0.01, seed=0))
    edges = [r.phases[k] for k in PHASES]
    assert all(b >= a for a, b in edges)
    assert all(edges[i][1] == pytest.approx(edges[i + 1][0]) for i in range(4))
    assert edges[0][0] == pytest.approx(r.start) and edges[-1][1] == pytest.approx(r.end)


def test_risk_flag_follows_cdc_cutoff():
    fast = analyse(simulate(Protocol(sit_to_stand=1, walk_out=2, turn=1, walk_back=2, turn_to_sit=2)))
    slow = analyse(simulate(Protocol(sit_to_stand=3, walk_out=7, turn=3, walk_back=7, turn_to_sit=5)))
    assert fast.total_time < STEADI_CUTOFF_S and not fast.increased_risk
    assert slow.total_time >= STEADI_CUTOFF_S and slow.increased_risk


def test_second_turn_direction_does_not_matter():
    a = analyse(simulate(Protocol(second_turn_same_direction=True)))
    b = analyse(simulate(Protocol(second_turn_same_direction=False)))
    assert a.total_time == pytest.approx(b.total_time, abs=0.2)


def test_step_count_tracks_cadence():
    p = Protocol(walk_out=5, walk_back=5, cadence_spm=100)
    r = analyse(simulate(p))
    expected = 100 / 60 * 10
    assert abs(r.steps - expected) <= 4


def test_posture_signal_separates_sitting_from_standing():
    seq = simulate(Protocol())
    post = posture_signal(seq.world)
    assert post[:20].mean() < 120 and post[int(4.5 * seq.fps)] > 160


def test_rejects_recording_without_standing_up():
    p = Protocol()
    seq = simulate(p)
    seated = PoseSequence(seq.fps, np.repeat(seq.world[:1], 200, axis=0), seq.image[:200], seq.vis[:200])
    with pytest.raises(TugError):
        analyse(seated)


def test_rejects_recording_with_no_person():
    seq = simulate(Protocol())
    empty = PoseSequence(seq.fps, np.full_like(seq.world, np.nan), seq.image, np.zeros_like(seq.vis))
    with pytest.raises(TugError):
        analyse(empty)


def test_random_protocols_always_analysable():
    rng = np.random.default_rng(0)
    for _ in range(25):
        p = random_protocol(rng)
        r = analyse(simulate(p, noise_m=0.02, dropout=0.05, seed=int(rng.integers(1e6))))
        assert abs(r.total_time - p.total) < 2.0


def test_select_subject_follows_seated_person_not_bystander():
    """Two people per frame: the subject sits then drifts right; a standing bystander stays left."""
    def person(x, seated):
        world = np.zeros((33, 3))
        world[[23, 24]] = [[0.1, 0, 0], [-0.1, 0, 0]]
        knee = [0, 0.05, 0.42] if seated else [0, 0.42, 0]
        world[25], world[26] = np.add(world[23], knee), np.add(world[24], knee)
        world[27], world[28] = world[25] + [0, 0.43, 0], world[26] + [0, 0.43, 0]
        image = np.zeros((33, 2))
        image[[L_HIP, R_HIP]] = [[x + 0.01, 0.5], [x - 0.01, 0.5]]
        return world, image, np.ones(33)

    frames = []
    for f in range(40):
        people = [person(0.2, False), person(0.6 + 0.005 * f, f < 10)]
        frames.append(people if f % 2 else people[::-1])      # detection order flips every frame
    seq = select_subject(frames, fps=30)
    x = seq.image[:, [L_HIP, R_HIP], 0].mean(axis=1)
    assert np.all(x > 0.55) and seq.detected.all()
