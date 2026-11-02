import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "g1"))

import g1_u2_sense as S  # noqa: E402


class SceneSignatureTest(unittest.TestCase):
    def test_dhash_is_64_bits_of_left_to_right_gradient(self):
        ramp = np.tile(np.arange(9, dtype="float32"), (8, 1))  # every pixel brighter than its left neighbour
        self.assertEqual(S.dhash_from_gray(ramp), "f" * 16)
        flat = np.zeros((8, 9), dtype="float32")
        self.assertEqual(S.dhash_from_gray(flat), "0" * 16)
        falling = np.tile(np.arange(9, 0, -1, dtype="float32"), (8, 1))
        self.assertEqual(S.dhash_from_gray(falling), "0" * 16, "a falling ramp has no rising edges")

    def test_dhash_is_stable_under_brightness_and_small_noise(self):
        rng = np.random.default_rng(7)
        scene = rng.uniform(0, 255, size=(8, 9)).astype("float32")
        self.assertEqual(S.dhash_from_gray(scene), S.dhash_from_gray(scene * 0.5 + 20), "exposure changes no bit")
        noisy = scene + rng.normal(0, 3, size=scene.shape).astype("float32")
        a = int(S.dhash_from_gray(scene), 16)
        b = int(S.dhash_from_gray(noisy), 16)
        self.assertLess(bin(a ^ b).count("1"), 12, "a little noise flips few bits")
        other = rng.uniform(0, 255, size=(8, 9)).astype("float32")
        c = int(S.dhash_from_gray(other), 16)
        self.assertGreaterEqual(bin(a ^ c).count("1"), 16, "a different scene flips many (random 64-bit hashes differ in ~32 bits on average)")

    def test_dhash_rejects_the_wrong_shape(self):
        with self.assertRaises(S.SenseError):
            S.dhash_from_gray(np.zeros((9, 8)))



if __name__ == "__main__":
    unittest.main()
