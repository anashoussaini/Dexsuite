"""Clipping must reach Genesis; changing only the option cannot fix renders."""
from types import SimpleNamespace
import pytest
from dexsuite.core.camera import CameraSystem
from dexsuite.options import CamerasOptions, DynamicCamOptions, StaticCamOptions

@pytest.mark.parametrize('cls', [StaticCamOptions, DynamicCamOptions])
@pytest.mark.parametrize('near,far', [(0,20),(-1,20),(.1,.1),(2,1),(.1,float('inf')),(float('nan'),20)])
def test_invalid_clipping(cls, near, far):
    kwargs = dict(pos=(0,0,0),lookat=(1,0,0)) if cls is StaticCamOptions else {}
    with pytest.raises(ValueError,match='clipping'):
        cls(near=near,far=far,**kwargs)


def test_sharpa_preset_and_genesis_arguments():
    calls=[]
    scene=SimpleNamespace(add_camera=lambda **kw: calls.append(kw) or object())
    gripper=SimpleNamespace(_registry_name='sharpa')
    system=CameraSystem(scene,object(),CamerasOptions(static={},dynamic={'wrist':'wrist_cam'}))
    system._all_grippers=lambda: [('arm',object(),'right_hand_C_MC',None,gripper)]
    system.create()
    assert calls[0]['near']==.005
    assert calls[0]['far']==20.
    assert calls[0]['res']==(224,224)
    spec=system._dyn_specs['wrist'][1]
    assert spec.pos_offset==(.073,.009,-.0102)


def test_static_clipping_passes_to_genesis():
    calls=[]
    scene=SimpleNamespace(add_camera=lambda **kw: calls.append(kw) or object())
    spec=StaticCamOptions(pos=(0,0,0),lookat=(1,0,0),near=.02,far=3.)
    system=CameraSystem(scene,object(),CamerasOptions(static={'front':spec},dynamic={}))
    system._all_grippers=lambda: [('arm',object(),'root',None,None)]
    system.create()
    assert calls[0]['near']==.02 and calls[0]['far']==3.
