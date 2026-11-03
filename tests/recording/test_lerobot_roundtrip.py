"""Exercise the installed LeRobot writer, including rejected RGB buffers."""
from types import SimpleNamespace
import numpy as np
from gymnasium import spaces
from dexsuite.options import RobotOptions, ArmOptions, SimOptions, CamerasOptions
from dexsuite.utils.lerobot_logger import LeRobotLogger


def fixture_env():
    cam=spaces.Dict({'rgb':spaces.Box(0,255,(64,64,3),np.uint8),'depth':spaces.Box(0,20,(64,64),np.float32)})
    obs=spaces.Dict({'cameras':spaces.Dict({'front':cam,'wrist':cam}),
        'state':spaces.Dict({'x':spaces.Box(-1,1,(1,),np.float32)})})
    return SimpleNamespace(observation_space=obs,action_space=spaces.Box(-1,1,(29,),np.float32),
        sim_options=SimOptions(control_hz=20),robot_options=RobotOptions(single=ArmOptions(manipulator='floating',gripper='sharpa')),
        cameras_options=CamerasOptions(),robot=SimpleNamespace(_layout=SimpleNamespace(total_dim=29,segments=[])))


def observation(color,x):
    return {'cameras':{name:{'rgb':np.full((64,64,3),color,np.uint8),'depth':np.full((64,64),.12,np.float32)} for name in ('wrist','front')},'state':{'x':np.array([x],np.float32)}}


def test_scalar_save_and_rejection(tmp_path):
    import pyarrow.parquet as pq
    import av
    logger=LeRobotLogger('test',fixture_env(),dataset_dir=tmp_path/'dataset')
    for _ in range(3):
        logger.add_step_pre(observation(240,.9),np.zeros(29,np.float32),0.,False,False,{})
    logger.clear_buffer()
    for i in range(4):
        obs = observation(20+i*10,.1*i)
        obs['state']['x'] = obs['state']['x'].astype(np.float64)
        for camera in obs['cameras'].values():
            camera['depth'] = camera['depth'].astype(np.float64)
        logger.add_step_pre(obs,np.full(29,i/10,np.float32),float(i),i==3,False,{'success':i==3})
    logger.save_episode();logger.close()
    table=pq.read_table(next((tmp_path/'dataset/data').rglob('*.parquet'))).to_pydict()
    assert len(table['action'])==4
    np.testing.assert_allclose(np.asarray(table['obs.state.x']).reshape(-1),[0,.1,.2,.3])
    for path in (tmp_path/'dataset/videos').rglob('*.mp4'):
        with av.open(str(path)) as container:
            frames=[f.to_ndarray(format='rgb24') for f in container.decode(video=0)]
        assert len(frames)==4
        assert np.mean(frames[0])<40, 'Rejected white images leaked into accepted episode'
