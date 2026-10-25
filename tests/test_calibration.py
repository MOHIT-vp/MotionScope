import numpy as np
import pytest

from motiontrack.calibration import CalibrationError, GroundCalibration, fit_speed, to_ground

SQUARE = [(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)]


def test_homography_maps_corners_to_rectangle():
    cal = GroundCalibration(SQUARE, width_m=16, length_m=12)
    H = cal.homography(640, 480)
    g = to_ground(H, [(160, 120), (480, 120), (480, 360), (160, 360)])
    np.testing.assert_allclose(g, [[0, 0], [16, 0], [16, 12], [0, 12]], atol=1e-3)


def test_perspective_quad_is_accepted():
    cal = GroundCalibration([(0.45, 0.4), (0.55, 0.4), (0.8, 0.9), (0.2, 0.9)], 3.5, 30)
    H = cal.homography(1000, 1000)
    # the far edge (short in pixels) and the near edge (long) are both 3.5 m wide
    far = to_ground(H, [(450, 400), (550, 400)])
    near = to_ground(H, [(200, 900), (800, 900)])
    assert np.linalg.norm(far[1] - far[0]) == pytest.approx(3.5, rel=1e-3)
    assert np.linalg.norm(near[1] - near[0]) == pytest.approx(3.5, rel=1e-3)


@pytest.mark.parametrize("pts", [
    [(0.25, 0.25), (0.75, 0.75), (0.75, 0.25), (0.25, 0.75)],   # self-intersecting
    [(0.1, 0.1), (0.2, 0.2), (0.3, 0.3), (0.4, 0.4)],            # collinear
    [(0.1, 0.1), (0.2, 0.1), (0.2, 0.2)],                        # too few
    [(-0.1, 0.1), (0.2, 0.1), (0.2, 0.2), (0.1, 0.2)],           # out of range
])
def test_invalid_quads_rejected(pts):
    with pytest.raises(CalibrationError):
        GroundCalibration(pts, 3, 3)


def test_invalid_distances_rejected():
    with pytest.raises(CalibrationError):
        GroundCalibration(SQUARE, 0, 5)


def test_fit_speed_linear_motion():
    t = np.arange(10) / 25.0
    g = np.stack([5.0 * t, 2.0 + 0 * t], axis=1)   # 5 m/s along x
    speed, resid = fit_speed(t, g)
    assert speed == pytest.approx(5.0, rel=1e-6)
    assert resid < 1e-9


def test_fit_speed_needs_three_samples():
    assert fit_speed([0, 0.04], np.zeros((2, 2))) is None


def test_roundtrip_dict():
    cal = GroundCalibration(SQUARE, 3.65, 9, "lane")
    assert GroundCalibration.from_dict(cal.to_dict()).to_dict() == cal.to_dict()
