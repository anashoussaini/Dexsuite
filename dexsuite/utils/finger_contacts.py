"""Native per-finger external contact forces; values are not sensor-calibrated."""
import numpy as np

DIGITS=('thumb','index','middle','ring','pinky')

def _numpy(x):
    return np.asarray(x.detach().cpu().numpy() if hasattr(x,'detach') else x)

class FingerContactReader:
    def __init__(self, hand):
        self.hand=hand
        self.hand_ids={int(link.idx) for link in hand.links}
        self.digit_by_link={int(link.idx):i for link in hand.links for i,d in enumerate(DIGITS) if f'right_{d}_' in link.name}

    def read(self):
        data=self.hand.get_contacts()
        a=_numpy(data['link_a']).reshape(-1); b=_numpy(data['link_b']).reshape(-1)
        fa=_numpy(data['force_a']).reshape(-1,3); fb=_numpy(data['force_b']).reshape(-1,3)
        valid=_numpy(data.get('valid_mask',np.ones(len(a),bool))).reshape(-1).astype(bool)
        forces=np.zeros((5,3),np.float32); contact=np.zeros(5,bool)
        for left,right,force_a,force_b in zip(a[valid],b[valid],fa[valid],fb[valid]):
            for own,other,force in ((int(left),int(right),force_a),(int(right),int(left),force_b)):
                digit=self.digit_by_link.get(own)
                if digit is not None and other not in self.hand_ids:
                    forces[digit]+=force; contact[digit]=True
        if not np.isfinite(forces).all(): raise RuntimeError('Nonfinite finger contact force')
        norm=np.linalg.norm(forces,axis=-1).astype(np.float32)
        if not np.isfinite(norm).all(): raise RuntimeError('Nonfinite finger force norm')
        return {'finger_force_world':forces, 'finger_force_norm':norm,
                'finger_force_display':np.clip(norm,0,30), 'finger_force_saturated':norm>30,
                'finger_contact':contact}

FEATURES={
 'obs.state.other.finger_force_world':{'dtype':'float32','shape':(5,3)},
 'obs.state.other.finger_force_norm':{'dtype':'float32','shape':(5,)},
 'obs.state.other.finger_force_display':{'dtype':'float32','shape':(5,)},
 'obs.state.other.finger_force_saturated':{'dtype':'bool','shape':(5,)},
 'obs.state.other.finger_contact':{'dtype':'bool','shape':(5,)},
 'obs.state.other.controller_phase':{'dtype':'int64','shape':(1,)},
 'obs.state.other.controller_stage':{'dtype':'int64','shape':(1,)},
}
