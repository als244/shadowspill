import pytest

from shadowspill.communication.fitting import fit_copy_cost


def test_recovers_startup_and_payload_rate():
    samples = [(size, 4e-6 + size / 25e9) for size in (1024, 1 << 20, 64 << 20)]
    rate, latency, error = fit_copy_cost(samples)
    assert rate == pytest.approx(25e9)
    assert latency == pytest.approx(4e-6)
    assert error < 1e-15


def test_no_negative_latency_from_noisy_measurements():
    rate, latency, error = fit_copy_cost([(1 << 20, 20e-6), (64 << 20, 3e-3)])
    assert latency == 0
    assert rate > 0
    assert error > 0


@pytest.mark.parametrize(
    "samples",
    [
        [(1024, 1e-6)],
        [(1024, 1e-6), (1024, 2e-6)],
        [(1024, 1e-6), (2048, 0)],
        [(1024, 2e-6), (2048, 1e-6)],
    ],
)
def test_refuses_unusable_samples(samples):
    with pytest.raises(ValueError):
        fit_copy_cost(samples)
