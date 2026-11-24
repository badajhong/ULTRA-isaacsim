"""Fixed sensor/cover links must not introduce collisions across their assembly's joint."""

import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from env.sim.urdf_info import _adjacent_collision_pairs, parse_urdf


class UrdfCollisionPairsTest(unittest.TestCase):
    def test_fixed_components_expand_only_neighbor_exclusions(self):
        root = ET.fromstring('''<robot name="test">
          <link name="base"/><link name="cover"/><link name="sensor"/>
          <link name="upper"/><link name="pad"/><link name="lower"/>
          <joint name="cover" type="fixed"><parent link="base"/><child link="cover"/></joint>
          <joint name="sensor" type="fixed"><parent link="cover"/><child link="sensor"/></joint>
          <joint name="upper" type="revolute"><parent link="base"/><child link="upper"/></joint>
          <joint name="pad" type="fixed"><parent link="upper"/><child link="pad"/></joint>
          <joint name="lower" type="revolute"><parent link="upper"/><child link="lower"/></joint>
        </robot>''')
        pairs = set(_adjacent_collision_pairs(root))
        self.assertIn(('cover', 'upper'), pairs)
        self.assertIn(('pad', 'sensor'), pairs)
        self.assertIn(('base', 'sensor'), pairs)
        self.assertIn(('lower', 'pad'), pairs)
        self.assertNotIn(('base', 'lower'), pairs)
        self.assertNotIn(('cover', 'lower'), pairs)
        self.assertTrue(all(a < b for a, b in pairs))

    def test_g1_preserves_layout_and_filters_pelvis_cover_against_hip(self):
        path = Path(__file__).resolve().parents[1] / 'ultra/data/assets/g1/g1_29dof.urdf'
        info = parse_urdf(path)
        pairs = set(info.adjacent_collision_pairs)
        self.assertEqual(len(info.body_names), 39)
        self.assertEqual(len(info.dof_names), 29)
        self.assertIn(('left_hip_pitch_link', 'pelvis_contour_link'), pairs)
        self.assertIn(('pelvis_contour_link', 'right_hip_pitch_link'), pairs)
        self.assertNotIn(('left_rubber_hand', 'right_rubber_hand'), pairs)
        self.assertNotIn(('left_shoulder_yaw_link', 'torso_link'), pairs)


if __name__ == '__main__':
    unittest.main()
