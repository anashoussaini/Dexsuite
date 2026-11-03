"""Physical regression for a hand attached to a six-joint floating wrist."""
import numpy as np
import pytest

import dexsuite as ds
from dexsuite.options import ArmOptions, ControllerOptions, RobotOptions


@pytest.mark.slow
def test_attached_hand_has_positive_mass_and_couples_to_wrist():
    from genesis.utils.misc import qd_to_numpy

    controller = ControllerOptions(
        name="joint_position", config={"normalized": False, "clip": False}
    )
    env = ds.make(
        "reach",
        robot=RobotOptions(single=ArmOptions(
            manipulator="floating", gripper="sharpa",
            manipulator_controller=controller, gripper_controller=controller,
        )),
        cameras=None, render_mode="rgb_array",
    )
    try:
        solver = env.scene.sim.rigid_solver
        # Test physical inertia without controller damping, which can conceal
        # an indefinite assembled mass matrix during ordinary integration.
        solver._init_invweight_and_meaninertia(force_update=True)
        mass = qd_to_numpy(solver.mass_mat)[:, :, 0]
        np.testing.assert_allclose(mass, mass.T, atol=1e-7)
        assert np.linalg.eigvalsh(mass).min() > 0
        assert np.all(qd_to_numpy(solver.links_info.invweight) >= 0)
        assert np.all(qd_to_numpy(solver.dofs_info.invweight) > 0)

        hand = env.robot.hand_ent
        wrist = env.robot.arm_ent
        root = hand.get_link(env.robot.gripper.root_link).root_idx
        masses = qd_to_numpy(solver.links_state.cinr_mass)[:, 0]
        composite = qd_to_numpy(solver.links_state.crb_mass)[:, 0]
        expected = sum(masses[link.idx] for link in solver.links if link.root_idx == root)
        np.testing.assert_allclose(composite[root], expected, rtol=2e-6, atol=2e-6)

        # Moving a finger changes the floating wrist's momentum: these are
        # coupled joints of one physical assembly, even with two API handles.
        cross = mass[wrist.dof_start:wrist.dof_end, hand.dof_start:hand.dof_end]
        assert np.linalg.norm(cross) > 1e-5
    finally:
        env.close()
