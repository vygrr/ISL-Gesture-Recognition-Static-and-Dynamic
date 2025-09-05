#!/usr/bin/env python3
# root_inference.py — Unified ISL realtime inference (Static MLP + Dynamic CTR-GCN)
# Sentence formation via Gemini (gemini_client.py). TTS via pyttsx3 on key 'p'.
# - Static: MLP + LabelEncoder (exactly like static/inference.py)
# - Dynamic: CTR-GCN, params from params.json/ckpt, id→label sorted by ID
# - UI: auto/manual windows, idle/active, pre/post frames, flip, bold & wrapping text
# - Static letters are buffered; Space commits a spelled word; dynamic words are whole tokens

from __future__ import annotations
import argparse, json, math, os, re, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import deque

import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, joblib

# ---------- MediaPipe ----------
try:
    import mediapipe as mp
    mp_holistic = mp.solutions.holistic
    mp_hands    = mp.solutions.hands
except Exception as e:
    raise SystemExit("MediaPipe required (pip install mediapipe>=0.10): " + str(e))

# ---------- Gemini ----------
# Expect gemini_client.py to define GeminiFormatter(model_name, api_key, temperature)
from gemini_client import GeminiFormatter


# ======================= Static (MLP) =======================
class MLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64),  nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32),   nn.BatchNorm1d(32),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )
    def forward(self, x): return self.net(x)

class StaticPredictor:
    """
    Auto-detects static artifacts under:
      - <static_dir>/data/...
      - <static_dir>/isl/data/...
      - <static_dir>/asl/data/...
    """
    def __init__(self, static_dir: Path, domain: str, device: torch.device):
        assert domain in ("Alphabet", "Numeral")
        enc_name = "alphabets_le.pkl" if domain == "Alphabet" else "numerals_le.pkl"
        pth_name = "alphabets_model.pth" if domain == "Alphabet" else "numerals_model.pth"

        candidates = [Path(static_dir), Path(static_dir) / "isl", Path(static_dir) / "asl"]
        enc = pth = None
        tried_dirs = []

        for base in candidates:
            data_root = base / "data"
            tried_dirs.append(str(data_root.resolve()))
            e = data_root / "encoder" / enc_name
            p = data_root / "model" / pth_name
            if e.exists() and p.exists():
                enc, pth = e, p
                break

        if enc is None or pth is None:
            raise FileNotFoundError(
                f"Missing static artifacts for {domain}.\n"
                f"Looked in:\n  - " + "\n  - ".join(tried_dirs) + "\n"
                f"Need files:\n  - encoder/{enc_name}\n  - model/{pth_name}\n"
                f"Hints:\n"
                f"  • Keep your current layout and run with --static_dir static (auto-tries static/isl and static/asl),\n"
                f"  • Or place files under <static_dir>/data/encoder and <static_dir>/data/model.\n"
            )

        self.encoder = joblib.load(str(enc))
        classes = getattr(self.encoder, "classes_", None)
        if classes is None:
            raise RuntimeError(f"Encoder at {enc} has no classes_.")
        self.model = MLP(input_dim=126, num_classes=len(classes)).to(device).eval()

        state = torch.load(pth, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state)
        self.device = device

    @torch.inference_mode()
    def predict_probs(self, X: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(X.astype(np.float32)).to(self.device)
        logits = self.model(x)
        return torch.softmax(logits, dim=1).float().cpu().numpy()

    def id_to_label(self, idx: int) -> str:
        return self.encoder.inverse_transform([idx])[0]


# ======================= Dynamic (CTR-GCN) =======================
SEQ_LEN, FEAT_DIM, DTYPE = 200, 258, np.float32
V_POSE,V_LHAND,V_RHAND=33,21,21; V_ALL=75

POSE_L={ "NOSE":0,"LEFT_SHOULDER":11,"RIGHT_SHOULDER":12,"LEFT_ELBOW":13,"RIGHT_ELBOW":14,
         "LEFT_WRIST":15,"RIGHT_WRIST":16,"LEFT_HIP":23,"RIGHT_HIP":24 }

HAND_EDGES=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
POSE_EDGES=[(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(0,11),(0,12)]
POSE_WRIST_L=POSE_L["LEFT_WRIST"]; POSE_WRIST_R=POSE_L["RIGHT_WRIST"]
LHAND_ROOT=V_POSE+0; RHAND_ROOT=V_POSE+V_LHAND+0

def extract_258_from_holistic(frame_bgr, holistic)->np.ndarray:
    rgb=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB); rgb.flags.writeable=False
    res=holistic.process(rgb); vec=[]
    if res.pose_landmarks and res.pose_landmarks.landmark:
        for lm in res.pose_landmarks.landmark: vec.extend([lm.x,lm.y,lm.z,lm.visibility])
    else: vec.extend([0.0]*(33*4))
    if res.left_hand_landmarks and res.left_hand_landmarks.landmark:
        for lm in res.left_hand_landmarks.landmark: vec.extend([lm.x,lm.y,lm.z])
    else: vec.extend([0.0]*(21*3))
    if res.right_hand_landmarks and res.right_hand_landmarks.landmark:
        for lm in res.right_hand_landmarks.landmark: vec.extend([lm.x,lm.y,lm.z])
    else: vec.extend([0.0]*(21*3))
    arr=np.asarray(vec,dtype=DTYPE)
    if arr.shape[0]!=FEAT_DIM:
        arr=arr[:FEAT_DIM] if arr.shape[0]>FEAT_DIM else np.pad(arr,(0,FEAT_DIM-arr.shape[0]),"constant")
    return arr

def slice_to_joints_xyz(x_258: np.ndarray)->np.ndarray:
    T=x_258.shape[0]
    pose =x_258[:,:33*4].reshape(T,33,4)[...,:3]
    lhand=x_258[:,33*4:33*4+21*3].reshape(T,21,3)
    rhand=x_258[:,33*4+21*3:].reshape(T,21,3)
    return np.concatenate([pose,lhand,rhand],axis=1).astype(np.float32)

def body_center_scale(j: np.ndarray, eps=1e-6)->np.ndarray:
    j=j.copy(); lh, rh = 23, 24; ls, rs = 11, 12
    for t in range(j.shape[0]):
        midhip=0.5*(j[t,lh]+j[t,rh]); j[t]-=midhip
        sd=float(np.linalg.norm(j[t,ls,:2]-j[t,rs,:2])); j[t]/=(sd if sd>eps else 1.0)
    return j

def bone_vectors(j: np.ndarray)->np.ndarray:
    V=V_ALL; P=np.full((V,),-1,np.int32)
    for (a,b) in POSE_EDGES:
        if P[b]==-1: P[b]=a
        if P[a]==-1: P[a]=b
    for (u,v) in HAND_EDGES:
        gu,gv=V_POSE+u, V_POSE+v
        if P[gv]==-1: P[gv]=gu
    for (u,v) in HAND_EDGES:
        gu,gv=V_POSE+V_LHAND+u, V_POSE+V_LHAND+v
        if P[gv]==-1: P[gv]=gu
    P[LHAND_ROOT]=POSE_WRIST_L; P[RHAND_ROOT]=POSE_WRIST_R
    bones=np.zeros_like(j)
    for v in range(V):
        p=int(P[v]); bones[:,v,:]=j[:,v,:]-(j[:,p,:] if p>=0 else 0.0)
    return bones

class CTRGCNBlock(nn.Module):
    def __init__(self,C_in,C_out,V=V_ALL,kernel_t=9,stride=1,dropout=0.3):
        super().__init__()
        self.theta=nn.Conv2d(C_in,C_out//4,1); self.phi=nn.Conv2d(C_in,C_out//4,1); self.g=nn.Conv2d(C_in,C_out,1)
        A=np.eye(V,dtype=np.float32)
        for (a,b) in POSE_EDGES: A[a,b]=A[b,a]=1
        for (u,v) in HAND_EDGES:
            gu,gv=V_POSE+u, V_POSE+v; A[gu,gv]=A[gv,gu]=1
            gu,gv=V_POSE+V_LHAND+u, V_POSE+V_LHAND+v; A[gu,gv]=A[gv,gu]=1
        A[POSE_WRIST_L, LHAND_ROOT]=A[LHAND_ROOT,POSE_WRIST_L]=1
        A[POSE_WRIST_R, RHAND_ROOT]=A[RHAND_ROOT,POSE_WRIST_R]=1
        D=np.sum(A,1,keepdims=True)+1e-6; A=A/np.sqrt(D@D.T)
        self.register_buffer("A_base",torch.from_numpy(A).float())
        self.A_learn=nn.Parameter(torch.zeros_like(self.A_base)); nn.init.uniform_(self.A_learn,-0.01,0.01)
        pad=(kernel_t-1)//2
        self.tconv=nn.Conv2d(C_out,C_out,kernel_size=(kernel_t,1),stride=(stride,1),padding=(pad,0))
        self.bn=nn.BatchNorm2d(C_out); self.drop=nn.Dropout(dropout)
        self.res=None
        if C_in!=C_out or stride!=1:
            self.res=nn.Sequential(nn.Conv2d(C_in,C_out,1,stride=(stride,1)), nn.BatchNorm2d(C_out))
    def forward(self,x):
        q=self.theta(x).mean(2,keepdim=True); k=self.phi(x).mean(2,keepdim=True)
        attn=torch.einsum("nctv,nctw->nvw",q,k)/math.sqrt(q.shape[1]+1e-6)
        A_dyn=torch.softmax(attn,dim=-1)
        gx=self.g(x)
        y=torch.einsum("nctv,vw->nctw",gx,self.A_base+self.A_learn) + torch.einsum("nctv,nvw->nctw",gx,A_dyn)
        y=self.tconv(y); r=x if self.res is None else self.res(x)
        return self.drop(F.relu(self.bn(y)+r, inplace=True))

class CTRGCN(nn.Module):
    def __init__(self,num_classes,c_in=9,channels=(64,128,256),blocks=(2,2,2),kernel_t=9,dropout=0.3):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv2d(c_in,channels[0],1),nn.BatchNorm2d(channels[0]),nn.ReLU(inplace=True))
        layers=[]; cprev=channels[0]
        for i,(c,n) in enumerate(zip(channels,blocks)):
            for j in range(n):
                stride=2 if (i>0 and j==0) else 1
                layers.append(CTRGCNBlock(cprev,c,V=V_ALL,kernel_t=kernel_t,stride=stride,dropout=dropout)); cprev=c
        self.backbone=nn.Sequential(*layers); self.pool=nn.AdaptiveAvgPool2d((1,1)); self.fc=nn.Linear(cprev,num_classes)
    def forward(self,x): x=self.stem(x); x=self.backbone(x); x=self.pool(x).flatten(1); return self.fc(x)

def load_params_from_ckpt_and_sidecar(ckpt_path: Path)->Dict:
    params={}
    try:
        ck=torch.load(ckpt_path,map_location="cpu")
        if isinstance(ck,dict) and "params" in ck and isinstance(ck["params"],dict): params=dict(ck["params"])
    except: pass
    if not params:
        side=ckpt_path.parent/"params.json"
        if side.exists(): params=json.loads(side.read_text(encoding="utf-8"))
    if not params: raise SystemExit("Could not find run params.")
    return params

def _norm_model_name(n:str)->str:
    n=(n or "").lower().strip()
    return {"ctr-gcn":"ctr_gcn","ctr_gcn_bihand":"ctr_gcn"}.get(n,n)

def build_model_from_params(num_classes:int, params:Dict):
    mn=_norm_model_name(params.get("model","ctr_gcn"))
    if mn!="ctr_gcn": raise SystemExit(f"Root expects CTR-GCN; got {mn}")
    c_in=3 + (3 if params.get("use_bones",False) else 0) + (3 if params.get("use_vel",False) else 0)
    channels=tuple(params.get("channels",(64,128,256)))
    blocks  =tuple(params.get("blocks",(2,2,2)))
    kernel_t=int(params.get("kernel_t",9)); dropout=float(params.get("dropout", params.get("drop_out",0.3)))
    return CTRGCN(num_classes,c_in=c_in,channels=channels,blocks=blocks,kernel_t=kernel_t,dropout=dropout)

def clean_dyn_label(lbl: str)->str:
    s = re.sub(r"^\s*\d+\.\s*", "", lbl)      # remove leading "NN. "
    s = re.sub(r"\s*\([^)]*\)", "", s)        # remove parentheses
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def build_id_to_label_sorted(aug_root: Path)->List[str]:
    j=aug_root/"label_to_id.json"
    if not j.exists(): raise FileNotFoundError(f"{j} not found")
    m=json.loads(j.read_text(encoding="utf-8"))
    pairs=[]
    for k,v in m.items():
        try: idx=int(v)
        except: idx=int(str(v).strip())
        pairs.append((idx, k))
    pairs.sort(key=lambda x:x[0])
    max_id=pairs[-1][0]
    id2=[None]*(max_id+1)
    for idx,lbl in pairs: id2[idx]=clean_dyn_label(lbl)
    print(f"[Dynamic labels] total={len(id2)} | head={id2[:8]}")
    return id2

def slice_to_inputs_ctr(seq_258: np.ndarray, valid_len:int, params:Dict, device, half:bool):
    j=slice_to_joints_xyz(seq_258)
    if params.get("normalize_body",False): j=body_center_scale(j)
    feats=[j]
    if params.get("use_bones",False): feats.append(bone_vectors(j))
    if params.get("use_vel",False):
        v=np.zeros_like(j); v[1:]=j[1:]-j[:-1]; feats.append(v)
    x=np.concatenate(feats,axis=-1) # (T,75,C)
    xt=torch.from_numpy(x).permute(2,0,1).unsqueeze(0).contiguous().to(device)
    if half: xt=xt.half()
    return {"x":xt,"lengths":torch.tensor([valid_len],device=device)}

def impute_short_gaps(seq: np.ndarray, max_gap:int=5)->np.ndarray:
    x=seq.copy(); POSE_DIM=33*4; HAND_DIM=21*3
    for slc in (slice(POSE_DIM,POSE_DIM+HAND_DIM), slice(POSE_DIM+HAND_DIM,POSE_DIM+2*HAND_DIM)):
        sub=x[:,slc]; miss=(np.abs(sub).sum(axis=1)==0.0)
        if not miss.any(): continue
        for c in range(sub.shape[1]):
            vals=sub[:,c]; i=0
            while i<len(vals):
                if miss[i]:
                    j=i
                    while j<len(vals) and miss[j]: j+=1
                    gap=j-i
                    if gap<=max_gap:
                        li=i-1; ri=j
                        left_ok=(li>=0 and not miss[li]); right_ok=(ri<len(vals) and not miss[ri])
                        if left_ok and right_ok:
                            v0,v1=vals[li],vals[ri]
                            for k in range(gap): vals[i+k]=v0+(v1-v0)*((k+1)/(gap+1))
                        elif left_ok: vals[i:j]=vals[li]
                        elif right_ok: vals[i:j]=vals[ri]
                    i=j
                else: i+=1
            sub[:,c]=vals
        x[:,slc]=sub
    return x

@dataclass
class DynPredictor:
    id_to_label: List[str]; model: Any; device: torch.device; params: Dict[str,Any]; use_half: bool=False
    @torch.inference_mode()
    def predict(self, seq_258: np.ndarray, seq_len:int)->Tuple[str,float]:
        L=min(seq_258.shape[0], seq_len); x=seq_258[:L].astype(np.float32); x=impute_short_gaps(x,5)
        if L<seq_len: x=np.concatenate([x, np.zeros((seq_len-L, seq_258.shape[1]),dtype=np.float32)], axis=0)
        batch=slice_to_inputs_ctr(x,L,self.params,self.device,self.use_half)
        logits=self.model(batch["x"])
        prob=torch.softmax(logits.float(),dim=1)[0]; top=int(prob.argmax().item())
        return self.id_to_label[top], float(prob[top].item())

def build_dynamic(aug_root: Path, ctr_dir: Path, force_cpu:bool=False, use_half:bool=False)->DynPredictor:
    id_to_label=build_id_to_label_sorted(aug_root)
    ckpt=ctr_dir/"ckpt_best.pt"
    if not ckpt.exists():
        cands=list(ctr_dir.rglob("ckpt_best.pt"))
        if not cands: raise FileNotFoundError(f"ckpt_best.pt not found under {ctr_dir}")
        ckpt=cands[0]
    params=load_params_from_ckpt_and_sidecar(ckpt)
    device=torch.device("cpu" if (force_cpu or not torch.cuda.is_available()) else "cuda")
    if device.type=='cuda': torch.backends.cudnn.benchmark=True
    print(f"[Dynamic device] {device}")
    model=build_model_from_params(num_classes=len(id_to_label), params=params).to(device).eval()
    sd=torch.load(ckpt,map_location=device)
    sd = sd.get("model_state", sd.get("state_dict", sd))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[Dynamic load] missing={len(missing)} unexpected={len(unexpected)}")
    if use_half and device.type=='cuda':
        try: model=model.half()
        except: pass
    return DynPredictor(id_to_label=id_to_label, model=model, device=device, params=params, use_half=(use_half and device.type=='cuda'))


# ======================= TTS (Windows SAPI, comtypes) =======================
import threading, queue, time

class TTS:
    """
    Robust Windows TTS using SAPI via comtypes.
    - Single background worker creates SAPI.SpVoice in its own COM apartment.
    - Speak requests are queued and processed synchronously one-by-one.
    - Much more reliable than pyttsx3 in Py 3.12 on some setups.
    """
    def __init__(self, enabled: bool, rate: int = 0, volume: float = 1.0, voice_query: str = ""):
        self.enabled = enabled
        self._q: "queue.Queue[str]" = queue.Queue()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._worker = None
        self._rate = int(rate)              # SAPI range is typically -10..+10
        self._volume = max(0, min(int(volume * 100), 100))  # 0..100
        self._voice_query = voice_query

        if not enabled:
            print("[TTS] Disabled (flag --tts not set).")
            return

        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        if not self._ready.wait(timeout=5.0):
            print("[TTS] Warning: SAPI worker did not signal ready.")
        else:
            print("[TTS] Ready (SAPI). Press 'p' to speak.")

    def _worker_loop(self):
        try:
            # COM must be initialized in the thread that uses it
            import pythoncom  # type: ignore
            pythoncom.CoInitialize()

            from comtypes.client import CreateObject  # type: ignore
            # Create SAPI voice
            voice = CreateObject("SAPI.SpVoice")

            # Volume 0..100, Rate -10..+10
            try: voice.Volume = self._volume
            except Exception: pass
            try: voice.Rate = max(-10, min(self._rate, 10))
            except Exception: pass

            # Pick a voice by substring if requested
            if self._voice_query:
                try:
                    token_cat = CreateObject("SAPI.SpObjectTokenCategory")
                    token_cat.SetId("HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices", False)
                    tokens = token_cat.EnumerateTokens()
                    q = self._voice_query.lower()
                    for t in tokens:
                        desc = t.GetDescription() or ""
                        if q in desc.lower():
                            voice.Voice = t
                            break
                except Exception:
                    pass

            self._ready.set()

            # SAPI Speak flags
            SPF_DEFAULT = 0
            SPF_PURGEBEFORESPEAK = 2  # purge any queued audio before new speak

            while not self._stop.is_set():
                try:
                    text = self._q.get(timeout=0.2)
                except queue.Empty:
                    continue
                if not text:
                    continue
                try:
                    # Purge any leftovers (just in case) and speak synchronously
                    voice.Speak("", SPF_PURGEBEFORESPEAK)
                    voice.Speak(text, SPF_DEFAULT)
                    # Tiny sleep to yield back to UI loop
                    time.sleep(0.005)
                except Exception as e:
                    print(f"[TTS] SAPI speak error: {e}")

        except Exception as e:
            print(f"[TTS] Worker init error: {e}")
        finally:
            # Clean up COM apartment
            try:
                import pythoncom  # type: ignore
                pythoncom.CoUninitialize()
            except Exception:
                pass

    def speak(self, text: str):
        if not self.enabled:
            print("[TTS] Not initialized (use --tts).")
            return
        if not text:
            print("[TTS] Nothing to speak (empty text).")
            return
        self._q.put(text)

    def shutdown(self):
        if not self.enabled:
            return
        self._stop.set()
        self._q.put("")
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=2.0)

# ======================= Windowing, HUD =======================
class WindowState: IDLE=0; RECORDING=1

@dataclass
class FrameBundle:
    ts: float; frame_disp: np.ndarray; frame_proc: np.ndarray
    left_ok: bool; right_ok: bool; hands_126: Optional[np.ndarray]; feat_258: Optional[np.ndarray]=None

def run_hands(frame_proc_bgr, hands_proc):
    rgb=cv2.cvtColor(frame_proc_bgr, cv2.COLOR_BGR2RGB); rgb.flags.writeable=False
    res_h=hands_proc.process(rgb)
    feats={"Left":[0.0]*63,"Right":[0.0]*63}; left_ok=right_ok=False
    if res_h.multi_hand_landmarks:
        for lm,handed in zip(res_h.multi_hand_landmarks, res_h.multi_handedness):
            side=handed.classification[0].label
            coords=[c for p in lm.landmark for c in (p.x,p.y,p.z)]
            feats[side]=coords
            if side=="Left": left_ok=True
            if side=="Right": right_ok=True
    hands_126=np.array(feats["Left"]+feats["Right"],dtype=np.float32) if (left_ok or right_ok) else None
    return left_ok, right_ok, hands_126

def extract_features_lazy(hands_proc, frame_bgr: np.ndarray, proc_scale: float)->FrameBundle:
    h,w=frame_bgr.shape[:2]
    frame_proc=frame_bgr if proc_scale==1.0 else cv2.resize(frame_bgr,(int(w*proc_scale), int(h*proc_scale)))
    left_ok,right_ok,hands_126=run_hands(frame_proc,hands_proc)
    return FrameBundle(ts=time.time(), frame_disp=frame_bgr, frame_proc=frame_proc,
                       left_ok=left_ok, right_ok=right_ok, hands_126=hands_126)

@dataclass
class WindowManager:
    mode_auto: bool=True; min_len:int=12; max_len:int=200; on_thresh:int=5; off_thresh:int=7
    pre_k:int=0; post_k:int=0
    state:int=WindowState.IDLE; on_count:int=0; off_count:int=0
    buffer: List[FrameBundle]=field(default_factory=list)
    prebuf: deque=field(default_factory=lambda: deque(maxlen=120))
    post_remaining:int=0
    def reset(self): self.state=WindowState.IDLE; self.on_count=self.off_count=0; self.buffer.clear(); self.post_remaining=0
    def manual_toggle(self):
        if self.state==WindowState.IDLE:
            self.state=WindowState.RECORDING
            self.buffer=list(self.prebuf)[-self.pre_k:] if (self.pre_k and len(self.prebuf)>0) else []
            self.on_count=self.off_count=0
        else:
            self.state=WindowState.IDLE
    def feed(self, fb: FrameBundle, active: bool)->Optional[List[FrameBundle]]:
        hands_present=(fb.left_ok or fb.right_ok); finished=None; self.prebuf.append(fb)
        if self.mode_auto:
            if not active: self.reset(); return None
            if self.state==WindowState.IDLE:
                if hands_present:
                    self.on_count+=1
                    if self.on_count>=self.on_thresh:
                        self.state=WindowState.RECORDING
                        self.buffer=list(self.prebuf)[-self.pre_k:] if (self.pre_k and len(self.prebuf)>0) else []
                        self.buffer.append(fb); self.off_count=0
                else: self.on_count=0
            else:
                self.buffer.append(fb)
                if not hands_present:
                    self.off_count+=1
                    if self.off_count>=self.off_thresh and self.post_remaining==0: self.post_remaining=self.post_k
                else: self.off_count=0
                if self.post_remaining>0:
                    self.post_remaining-=1
                    if self.post_remaining==0 and len(self.buffer)>=self.min_len:
                        finished=self.buffer.copy(); self.reset()
                if len(self.buffer)>=self.max_len and finished is None:
                    finished=self.buffer.copy(); self.reset()
        else:
            if self.state==WindowState.RECORDING:
                self.buffer.append(fb)
                if len(self.buffer)>=self.max_len:
                    finished=self.buffer.copy(); self.reset()
        return finished

FONT=cv2.FONT_HERSHEY_SIMPLEX
def draw_wrapped_tokens(img, tokens: List[str], origin: Tuple[int,int], max_width:int,
                        color=(255,255,255), scale=0.9, thickness=2, line_gap=10):
    x0,y0=origin; x,y=x0,y0
    space_w,_=cv2.getTextSize(" ",FONT,scale,thickness)[0]
    for tok in tokens:
        (w,h),_=cv2.getTextSize(tok,FONT,scale,thickness)
        if x+w > x0+max_width:
            x=x0; y+=h+line_gap
        cv2.putText(img, tok, (x,y), FONT, scale, color, thickness, cv2.LINE_AA)
        cv2.putText(img, tok, (x,y), FONT, scale, color, thickness, cv2.LINE_AA)
        x+=w+space_w
    return y

def draw_wrapped_text(img, text:str, origin: Tuple[int,int], max_width:int,
                      color=(255,255,255), scale=0.85, thickness=2, line_gap=8):
    words=text.split() if text else []
    return draw_wrapped_tokens(img, words, origin, max_width, color, scale, thickness, line_gap)


# ======================= Context & helpers =======================
def map_static_to_char(label: str, domain: str)->str:
    if domain=="Alphabet":
        if label in ("E1","E2"): return "E"
        return label[:1].upper()
    else:
        if label.lower() in ("9a","9b"): return "9"
        return label

@dataclass
class Context:
    static_domain: str="Alphabet"
    token_mode: str="dynamic"       # 'static' or 'dynamic'
    tokens: List[str]=field(default_factory=list)
    last_pred: str='' ; last_conf: float=0.0
    spell_buf: str=''               # live buffer for static letters/digits → forms one word
    spelled_idxs: set[int]=field(default_factory=set)  # indices in tokens that came from static spelling
    final_sentence: str=""          # set on 's'


# ======================= The App =======================
class App:
    def __init__(self, args: argparse.Namespace):
        self.args=args
        device=torch.device("cpu" if args.cpu or (not torch.cuda.is_available()) else "cuda")
        self.device=device
        self.ctx=Context(static_domain="Alphabet", token_mode=("dynamic" if args.default_dynamic else "static"))

        # Windowing config
        fps=max(1, args.fps)
        pre_k  = max(0, args.pre_frames,  int(round((args.pre_secs or 0.0)  * fps)))
        post_k = max(0, args.post_frames, int(round((args.post_secs or 0.0) * fps)))
        self.wm=WindowManager(mode_auto=(args.mode=='auto'), min_len=args.min_window, max_len=args.max_window,
                              on_thresh=args.on_thresh, off_thresh=args.off_thresh, pre_k=pre_k, post_k=post_k)
        self.flip=args.flip; self.active=args.active; self.font_white=(args.font_color=='white'); self.proc_scale=float(args.proc_scale)
        self.holistic_complexity=int(args.holistic_complexity)

        # Static models
        static_dir=Path(args.static_dir)
        self.static_alpha=StaticPredictor(static_dir,"Alphabet",device)
        self.static_num  =StaticPredictor(static_dir,"Numeral",device)

        # Dynamic
        aug_root=Path(args.aug_root); ctr_dir=Path(args.ctr_dir)
        self.dynamic=build_dynamic(aug_root,ctr_dir,force_cpu=args.cpu,use_half=args.half)

        # Gemini
        self.use_gemini = bool(args.use_gemini)
        if self.use_gemini:
            key = args.gemini_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            try:
                self.gemini = GeminiFormatter(model_name=args.gemini_model, api_key=key, temperature=args.temperature)
                print(f"[Gemini] Using model={args.gemini_model}")
            except Exception as e:
                print(f"[Gemini] Disabled ({e}).")
                self.use_gemini=False

        # TTS
        self.tts = TTS(enabled=args.tts, rate=args.tts_rate, volume=args.tts_volume, voice_query=args.tts_voice)

    def current_static(self)->StaticPredictor:
        return self.static_alpha if self.ctx.static_domain=='Alphabet' else self.static_num

    def static_majority(self, feats_126: List[np.ndarray])->Tuple[str,float]:
        be=self.current_static(); X=np.stack(feats_126,axis=0).astype(np.float32)
        probs=be.predict_probs(X); sums=probs.sum(axis=0); top_idx=int(np.argmax(sums))
        label=be.id_to_label(top_idx); conf=float(probs[:,top_idx].mean())
        return label, conf

    def _commit_spelling_buffer(self):
        if self.ctx.spell_buf:
            idx=len(self.ctx.tokens)
            word=self.ctx.spell_buf.strip()
            if word.isupper(): word=word.title()
            self.ctx.tokens.append(word)
            self.ctx.spelled_idxs.add(idx)
            self.ctx.spell_buf=''

    def _local_minimal_sentence(self, tokens: List[str])->str:
        if not tokens: return ""
        t=[t.strip() for t in tokens if t.strip()]
        if not t: return ""
        subj=t[0]
        cop = "am" if subj.lower()=="i" else ("are" if subj.lower() in ["we","they","you"] else "is")
        sent = f"{subj} {cop} {' '.join(t[1:])}."
        return sent[0:1].upper() + sent[1:]

    def _finalize_sentence(self):
        self._commit_spelling_buffer()
        if not self.ctx.tokens:
            self.ctx.final_sentence=""; return
        tokens=self.ctx.tokens
        if self.use_gemini:
            try:
                sentence, _ = self.gemini.format_tokens(tokens, spelled_indices=list(self.ctx.spelled_idxs), timeout_s=12.0)
                self.ctx.final_sentence = sentence
            except Exception as e:
                print(f"[Gemini] Fallback due to: {e}")
                self.ctx.final_sentence = self._local_minimal_sentence(tokens)
        else:
            self.ctx.final_sentence = self._local_minimal_sentence(tokens)

    def _speak_now(self):
    # Speak only the finalized sentence. If it's empty, tell the user to press 's' first.
        text = self.ctx.final_sentence.strip()
        if not text:
            print("[TTS] No finalized sentence. Press 's' to finalize first.")
            return
        print(f'[KEY] p -> Speaking FINAL: "{text}"')
        self.tts.speak(text)



    # ------------------ main loop ------------------
    def run(self):
        cap=cv2.VideoCapture(self.args.cam)
        if self.args.width and self.args.height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,self.args.width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,self.args.height)
        cap.set(cv2.CAP_PROP_FPS,self.args.fps)

        with mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                            min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands_proc:
            print("Press 'h' for help.")
            hol = mp_holistic.Holistic(model_complexity=self.holistic_complexity, refine_face_landmarks=False,
                                       enable_segmentation=False, min_detection_confidence=0.35, min_tracking_confidence=0.5)
            try:
                while True:
                    ok,frame=cap.read()
                    if not ok: print("[WARN] Camera frame not available."); break
                    if self.flip: frame=cv2.flip(frame,1)

                    fb=extract_features_lazy(hands_proc, frame, proc_scale=self.proc_scale)
                    finished=None

                    if self.ctx.token_mode=='dynamic' and self.wm.state==WindowState.RECORDING:
                        fb.feat_258=extract_258_from_holistic(fb.frame_proc, hol)

                    finished=self.wm.feed(fb, active=self.active)

                    if self.wm.state==WindowState.RECORDING and self.ctx.token_mode=='dynamic':
                        for x in self.wm.buffer:
                            if x.feat_258 is None:
                                x.feat_258=extract_258_from_holistic(x.frame_proc, hol)

                    # --------------- HUD ---------------
                    hud=fb.frame_disp.copy()
                    status='REC' if self.wm.state==WindowState.RECORDING else 'IDLE'
                    txt_color=(255,255,255) if self.font_white else (0,0,0)
                    cv2.putText(hud, f"Mode: {'AUTO' if self.wm.mode_auto else 'MANUAL'} | Window: {status} | Active: {'ON' if self.active else 'OFF'} | Flip: {'ON' if self.flip else 'OFF'}",
                                (10,24), FONT, 0.65, txt_color, 2, cv2.LINE_AA)
                    if self.wm.state==WindowState.RECORDING:
                        cv2.putText(hud, f"frames={len(self.wm.buffer)}", (10,50), FONT, 0.65, txt_color, 2, cv2.LINE_AA)
                    if self.ctx.last_pred:
                        cv2.putText(hud, f"Last: {self.ctx.last_pred} ({self.ctx.last_conf:.2f})", (10,74), FONT, 0.6, txt_color, 2, cv2.LINE_AA)

                    tokens_for_draw=self.ctx.tokens.copy()
                    if self.ctx.spell_buf:
                        preview=self.ctx.spell_buf if not self.ctx.spell_buf.isupper() else self.ctx.spell_buf.title()
                        tokens_for_draw.append(preview)
                    y_after_tokens=draw_wrapped_tokens(hud, tokens_for_draw, origin=(10,100),
                                                       max_width=hud.shape[1]-20, color=txt_color, scale=0.9, thickness=2, line_gap=10)
                    if self.ctx.final_sentence:
                        draw_wrapped_text(hud, "Final: " + self.ctx.final_sentence, origin=(10,y_after_tokens+18),
                                          max_width=hud.shape[1]-20, color=txt_color, scale=0.85, thickness=2, line_gap=8)

                    help_line=("q quit | h help | t auto/manual | a active | f flip | g font | "
                                "1 static | 2 dynamic | m A↔N (static) | Space start/stop (manual) or commit spelled word | "
                                "Enter accept | b back | c clear | s finalize | p speak (needs --tts)")
                    cv2.putText(hud, help_line, (10, hud.shape[0]-12), FONT, 0.50, txt_color, 1, cv2.LINE_AA)
                    cv2.imshow("ISL Unified Inference (Gemini + TTS)", hud)

                    # --------------- Keys ---------------
                    key=cv2.waitKey(1) & 0xFF
                    if key==ord('q'):
                        break
                    elif key==ord('h'):
                        print(help_line)
                    elif key==ord('t'):
                        self.wm.mode_auto=not self.wm.mode_auto; self.wm.reset()
                    elif key==ord('a'):
                        self.active=not self.active
                        if not self.active: self.wm.reset()
                    elif key==ord('f'):
                        self.flip=not self.flip
                    elif key==ord('g'):
                        self.font_white=not self.font_white
                    elif key==ord(' '):
                        if not self.wm.mode_auto:
                            if self.wm.state==WindowState.IDLE:
                                self.wm.manual_toggle()
                            else:
                                if len(self.wm.buffer)>=self.wm.min_len:
                                    finished=self.wm.buffer.copy()
                                self.wm.reset()
                        else:
                            self._commit_spelling_buffer(); self.ctx.final_sentence=""
                    elif key==ord('1'):
                        self.ctx.token_mode='static'
                    elif key==ord('2'):
                        self.ctx.token_mode='dynamic'
                        self._commit_spelling_buffer(); self.ctx.final_sentence=""
                    elif key==ord('m') and self.ctx.token_mode=='static':
                        self.ctx.static_domain='Numeral' if self.ctx.static_domain=='Alphabet' else 'Alphabet'
                    elif key==13:  # Enter
                        if self.ctx.last_pred:
                            if self.ctx.token_mode=='static':
                                ch=map_static_to_char(self.ctx.last_pred, self.ctx.static_domain)
                                self.ctx.spell_buf+=ch
                            else:
                                self._commit_spelling_buffer(); self.ctx.tokens.append(self.ctx.last_pred)
                            self.ctx.final_sentence=""
                        else:
                            self._commit_spelling_buffer(); self.ctx.final_sentence=""
                    elif key==ord('b'):
                        if self.ctx.spell_buf:
                            self.ctx.spell_buf=self.ctx.spell_buf[:-1]
                        elif self.ctx.tokens:
                            last_idx=len(self.ctx.tokens)-1
                            if last_idx in self.ctx.spelled_idxs: self.ctx.spelled_idxs.discard(last_idx)
                            self.ctx.tokens.pop()
                        self.ctx.final_sentence=""
                    elif key==ord('c'):
                        self.ctx.tokens.clear(); self.ctx.spell_buf=''; self.ctx.spelled_idxs.clear(); self.ctx.final_sentence=""
                    elif key==ord('s'):
                        self._finalize_sentence()
                    elif key==ord('p'):
                        self._speak_now()

                    # --------------- Window finished → token ---------------
                    if finished is not None:
                        if self.ctx.token_mode=='dynamic':
                            feats_258=[(x.feat_258 if x.feat_258 is not None else extract_258_from_holistic(x.frame_proc, hol)) for x in finished]
                            seq_258=np.stack(feats_258, axis=0).astype(np.float32)
                            label,conf=self.dynamic.predict(seq_258, seq_len=self.args.seq)
                            self._commit_spelling_buffer()
                            self.ctx.tokens.append(label)
                        else:
                            hands_feats=[x.hands_126 for x in finished if x.hands_126 is not None]
                            if hands_feats:
                                raw,conf=self.static_majority(hands_feats)
                                ch=map_static_to_char(raw, self.ctx.static_domain); self.ctx.spell_buf+=ch; label=raw
                            else:
                                label,conf=("<no-hands>",0.0)
                        self.ctx.last_pred,self.ctx.last_conf=label,conf
                        self.ctx.final_sentence=""

            finally:
                hol.close()
                try:
                    self.tts.shutdown()
                except Exception:
                    pass
            cap.release(); cv2.destroyAllWindows()


# ======================= CLI =======================
def parse_args()->argparse.Namespace:
    ap=argparse.ArgumentParser(description="Unified ISL inference with Gemini sentence formation; optional TTS on 'p'")
    # camera
    ap.add_argument('--cam',type=int,default=0); ap.add_argument('--fps',type=int,default=30)
    ap.add_argument('--width',type=int,default=1280); ap.add_argument('--height',type=int,default=720)
    # perf
    ap.add_argument('--proc_scale',type=float,default=1.0)
    ap.add_argument('--holistic_complexity',type=int,default=1,choices=[0,1,2])
    ap.add_argument('--half',action='store_true')
    # windowing
    ap.add_argument('--mode',choices=['auto','manual'],default='auto')
    ap.add_argument('--min_window',type=int,default=12); ap.add_argument('--max_window',type=int,default=200)
    ap.add_argument('--on_thresh',type=int,default=5); ap.add_argument('--off_thresh',type=int,default=7)
    ap.add_argument('--pre_frames',type=int,default=0); ap.add_argument('--post_frames',type=int,default=0)
    ap.add_argument('--pre_secs',type=float,default=0.0); ap.add_argument('--post_secs',type=float,default=0.0)
    ap.add_argument('--active',action='store_true')
    ap.add_argument('--flip',dest='flip',action='store_true'); ap.add_argument('--no-flip',dest='flip',action='store_false'); ap.set_defaults(flip=True)
    ap.add_argument('--font_color',choices=['white','black'],default='white')
    # static
    ap.add_argument('--static_dir',type=str,default='static')
    # dynamic
    ap.add_argument('--aug_root',type=str,default='dynamic/data/top_100/aug_keypoints')
    ap.add_argument('--ctr_dir',type=str,default='dynamic/data/top_100/ctr_gcn')
    ap.add_argument('--seq',type=int,default=200)
    ap.add_argument('--cpu',action='store_true')
    # Gemini
    ap.add_argument('--use_gemini',action='store_true')
    ap.add_argument('--gemini_model',type=str,default='gemini-1.5-flash')
    ap.add_argument('--gemini_key',type=str,default='')
    ap.add_argument('--temperature',type=float,default=0.2)
    # start mode
    ap.add_argument('--default_dynamic',action='store_true')
    # TTS
    ap.add_argument('--tts',action='store_true', help="Enable TTS (pyttsx3). Press 'p' to speak.")
    ap.add_argument('--tts_rate',type=int,default=180, help="TTS words-per-minute (pyttsx3 rate).")
    ap.add_argument('--tts_volume',type=float,default=1.0, help="TTS volume (0.0–1.0).")
    ap.add_argument('--tts_voice',type=str,default="", help="Substring to choose a voice (e.g., 'Zira' or 'David').")
    return ap.parse_args()

def main():
    os.environ.setdefault("OMP_NUM_THREADS","1")
    try: cv2.setUseOptimized(True)
    except: pass
    try: cv2.setNumThreads(1)
    except: pass
    args=parse_args()
    App(args).run()

if __name__=="__main__":
    main()
