# tests/robot/_utils.py
from dexsuite.options.robot import ArmOptions


def arm_ctrl_field_name() -> str:
    fields = ArmOptions.__dataclass_fields__
    if "manipulator_controller" in fields:
        return "manipulator_controller"
    if "arm_controller" in fields:
        return "arm_controller"
    raise RuntimeError(
        "ArmOptions must define 'manipulator_controller' or 'arm_controller'.",
    )
