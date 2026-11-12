import pytest
import torch

import dexsuite as ds
from dexsuite.core.registry import MANIP_REG
from dexsuite.options import ArmOptions, ControllerOptions, RobotOptions


def test_floating_registered_with_sharpa_adapters():
    model = MANIP_REG["floating"]()
    assert int(model.dof) == 6
    assert model.end_link == "attachment"
    assert {"sharpa", "sharpa_right", "sharpa_fardexbot"} <= set(model.adapters)


@pytest.mark.slow
def test_env_init_floating_sharpa_minimal():
    rob = RobotOptions(
        single=ArmOptions(
            manipulator="floating",
            gripper="sharpa",
            manipulator_controller=ControllerOptions(
                name="joint_position", config={"normalized": False, "clip": False}
            ),
            gripper_controller=ControllerOptions(
                name="joint_position", config={"normalized": False, "clip": False}
            ),
        ),
    )
    env = ds.make("reach", robot=rob, cameras=None, render_mode="rgb_array")
    try:
        obs, _ = env.reset()
        manip = obs["state"]["manipulator"]
        assert manip["qpos"].shape[-1] == 6
        assert manip["eef_pos"].shape[-1] == 3
        assert manip["eef_quat"].shape[-1] == 4
        assert obs["state"]["gripper"]["qpos"].shape[-1] == 22
        assert env.action_space.shape == (6 + 22,)

        home = env.robot.manip.home_q.cpu()
        eef_pos = manip["eef_pos"].reshape(-1).cpu()
        torch.testing.assert_close(eef_pos, home[:3], atol=1e-3, rtol=0)
        # gantry at zero rotation -> hand root palm down (+X_hand -> world -Z), fingers along world +X
        root = env.robot.hand_ent.get_link(env.robot.gripper.root_link)
        root_quat = root.get_quat().reshape(-1).cpu()
        expected = torch.tensor([0.7071068, 0.0, 0.7071068, 0.0], device=root_quat.device)
        assert min((root_quat - expected).abs().max(), (root_quat + expected).abs().max()) < 1e-3

        act = torch.cat((home, env.robot.gripper.home_q.cpu()))
        obs, *_ = env.step(act)
        eef_pos = obs["state"]["manipulator"]["eef_pos"].reshape(-1).cpu()
        torch.testing.assert_close(eef_pos, home[:3], atol=5e-3, rtol=0)
    finally:
        env.close()


def test_fardexbot_is_the_plain_hand_plus_the_camera_mount():
    """The hand the scripted Sharpa datasets were recorded with: same joints and gains, heavier root, mount geoms."""
    from xml.etree import ElementTree

    from dexsuite.core.registry import GRIP_REG

    plain, rig = GRIP_REG["sharpa"], GRIP_REG["sharpa_fardexbot"]
    assert rig is not plain
    for name in ("home_q", "q_lo", "q_hi", "kp", "kv"):
        torch.testing.assert_close(getattr(rig, name), getattr(plain, name))

    def root_body(path):
        bodies = ElementTree.parse(path).getroot().iter("body")  # noqa: S314 - a model file of the package
        return next(b for b in bodies if b.get("name") == plain.root_link)

    plain_root, rig_root = root_body(plain.model_path), root_body(rig.model_path)
    assert float(rig_root.find("inertial").get("mass")) > float(plain_root.find("inertial").get("mass")) + 0.4
    mount = {g.get("name") for g in rig_root.findall("geom")} - {g.get("name") for g in plain_root.findall("geom")}
    assert {"fardexbox_mount_plate_disk", "fardexbox_d405", "fardexbox_qc_body"} <= mount
