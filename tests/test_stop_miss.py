"""완전 정차인데 상자를 못 찾으면 그 화면 저장 (2026-11-17) — 가짜 팔 · 가짜 카메라만 쓴다.

    python -m unittest tests.test_stop_miss
"""
import tempfile
import unittest
from pathlib import Path

from mission.config import MissionConfig
from mission.load_link import FakeLoadLink
from mission.motions import MotionBook
from mission.place_flow import PlaceRunner, Seen, YoloEye
from mission.yolo import Det
from tests.test_mission_place import Clock, FakeOps


class StopEye:
    """차 + 상자가 보이다가, 정지 신호 뒤에는 hide_box 면 상자 없이 차만 보인다. 저장 요청은 기록만"""

    def __init__(self, link, hide_box=True, fail=False):
        self.link, self.hide_box, self.fail = link, hide_box, fail
        self.saved = []

    def look(self):
        dets = [Det("car", 0.9, 200, 180, 180, 160)]
        if not (self.hide_box and self.link.state == "STOP_REQUEST"):
            dets.append(Det("box", 0.9, 250, 210, 80, 60))
        return Seen(dets, 640)

    def save_last(self, folder, stem):
        if self.fail:
            raise OSError("디스크가 꽉 찼어요")
        self.saved.append((Path(folder), stem))
        return [Path(folder) / f"{stem}.jpg", Path(folder) / f"{stem}_yolo.jpg"]


class StopMissTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = MissionConfig(track_period=0.1)
        self.link = FakeLoadLink()
        self.events = []

    def runner(self, eye):
        clock = Clock()
        r = PlaceRunner(self.cfg, FakeOps(), eye, self.link, MotionBook(self.root / "m.json", self.root / "b"),
                        clock=clock, sleep=clock.sleep, log=lambda ev, **kw: self.events.append((ev, kw)))
        r.max_wait = 30
        r.miss_dir = self.root / "stop_miss"
        return r

    def test_stopped_without_a_box_saves_the_screen(self):
        eye = StopEye(self.link)
        r = self.runner(eye)
        self.assertTrue(r.wait_car_and_stop())                                   # 차로 정차 확인은 된다
        self.assertIsNone(r.stop_box)
        self.assertEqual(len(eye.saved), 1)
        self.assertEqual(eye.saved[0][0], self.root / "stop_miss")
        self.assertEqual(len(r.miss_saved), 2)                                   # 원본 + YOLO 그림
        saved = [kw for ev, kw in self.events if ev == "stop_miss_saved"]
        self.assertEqual(saved[0]["seen"], ["car"])

    def test_box_seen_at_the_stop_saves_nothing(self):
        eye = StopEye(self.link, hide_box=False)
        r = self.runner(eye)
        self.assertTrue(r.wait_car_and_stop())
        self.assertIsNotNone(r.stop_box)
        self.assertEqual(eye.saved, [])
        self.assertEqual(r.miss_saved, [])

    def test_switched_off_saves_nothing(self):
        self.cfg.save_stop_miss = False
        eye = StopEye(self.link)
        self.assertTrue(self.runner(eye).wait_car_and_stop())
        self.assertEqual(eye.saved, [])

    def test_a_failed_save_does_not_stop_the_loading(self):
        eye = StopEye(self.link, fail=True)
        self.assertTrue(self.runner(eye).wait_car_and_stop())
        self.assertIn("stop_miss_save_failed", [ev for ev, _ in self.events])

    def test_eye_without_save_is_skipped(self):
        eye = StopEye(self.link)
        eye.save_last = None
        self.assertTrue(self.runner(eye).wait_car_and_stop())


def _cv2_ok():
    try:
        import cv2
        import numpy  # noqa: F401
        return hasattr(cv2, "imwrite")
    except Exception:
        return False


@unittest.skipUnless(_cv2_ok(), "OpenCV 가 없는 PC (Pi 컨테이너에서 확인)")
class YoloEyeSaveTest(unittest.TestCase):
    def test_writes_the_raw_and_the_drawn_photo(self):
        import numpy as np

        class Det0:
            def detect(self, frame):
                return [Det("car", 0.9, 10, 10, 50, 40)]

        eye = YoloEye(lambda: np.full((480, 640, 3), 120, np.uint8), Det0())
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(eye.save_last(d, "x"), [])                       # 아직 사진 없음
            eye.look()
            files = eye.save_last(Path(d) / "stop_miss", "x")
            self.assertEqual([f.name for f in files], ["x.jpg", "x_yolo.jpg"])
            self.assertTrue(all(f.stat().st_size > 0 for f in files))


if __name__ == "__main__":
    unittest.main()
