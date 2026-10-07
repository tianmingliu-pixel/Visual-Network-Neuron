# NeuroCore · Visual Network Neuron

[中文](README.md) | [English](README.en.md) | **한국어** | [Deutsch](README.de.md)

확장 가능한 PyTorch 신경망 코어 라이브러리이며, 네트워크가 학습하는 과정을 눈으로 볼 수 있는 시각화 학습 UI를 함께 제공합니다.

1단계에서는 네 가지 기본 네트워크를 구현하고, **레지스트리 + 확장 폴더** 구조로 앞으로 모든 IT 분야의 네트워크(GNN, 오디오, 시계열, 추천, 보안, 강화학습 …)를 코어를 건드리지 않고 추가할 수 있게 했습니다.

| 모듈 | 파일 | 핵심 아이디어 |
|---|---|---|
| **Transformer** | `neurocore/models/transformer.py` | 어텐션만으로 이루어진 인코더–디코더: `softmax(QKᵀ/√d)V`, 인과 마스크 + 교차 어텐션 |
| **ViT** | `neurocore/models/vit.py` | 이미지를 패치로 잘라 토큰으로 → [CLS] + 위치 인코딩 → Transformer 인코더 |
| **U-Net** | `neurocore/models/unet.py` | 다운샘플링 / 업샘플링 + 같은 단계끼리 스킵 연결; 확산용 타임스텝 / 클래스 조건 선택 가능 |
| **DiT** | `neurocore/models/dit.py` | Transformer를 확산 디노이저로 사용, adaLN-Zero로 타임스텝과 클래스 주입 |
| DDPM | `neurocore/diffusion/ddpm.py` | 순방향 노이즈 추가 / 노이즈 ε 예측 / 조상 샘플링 + CFG, U-Net과 DiT 공용 |
| MLP | `neurocore/models/mlp.py` | 가장 기본적인 완전연결 신경망 — 그래프에서 모든 뉴런과 가중치를 볼 수 있음 |
| 공용 블록 | `neurocore/layers/` | 멀티헤드 어텐션, FFN, Pre-LN 블록, 사인파 / 타임스텝 / 패치 임베딩 |

## 빠른 시작 (Windows PowerShell)

```powershell
cd NeuroCore
# 명령 하나로: 환경 점검 → .venv 생성 → GPU에 맞는 PyTorch 설치 → 테스트 → 데모
powershell -ExecutionPolicy Bypass -File .\deploy.ps1
```

단계별 실행:

```powershell
.\scripts\check_env.ps1                 # 점검만, 설치하지 않음 (PASS / WARN / FAIL + 해결 명령)
.\scripts\check_env.ps1 -ReportPath env_report.json
.\scripts\setup_env.ps1                 # 배포 (GPU 자동 감지)
.\scripts\setup_env.ps1 -Cuda cpu       # CPU 버전 강제
.\scripts\setup_env.ps1 -Cuda cu130 -TorchVersion 2.14.0 -Recreate
.\scripts\run.ps1 -Task demo            # 네 가지 네트워크의 작은 학습 데모
.\scripts\run.ps1 -Task demo -Model dit -Steps 300
.\scripts\run.ps1 -Task test            # 테스트 실행 (pytest)
.\scripts\run.ps1 -Task list            # 등록된 모델
.\scripts\run.ps1 -Task info            # PyTorch / GPU 상태
```

### `check_env.ps1` 점검 항목
64비트 OS, PowerShell ≥ 5.1, Python 3.10–3.14 (`py` 런처 우선), pip, venv, Git, NVIDIA GPU와 드라이버가 지원하는 CUDA 버전, 메모리, 디스크, Windows 긴 경로, 실행 정책, 설치된 PyTorch가 GPU를 쓸 수 있는지. PASS가 아닌 항목마다 해결 명령이 함께 표시됩니다.

### PyTorch 설치 소스 자동 선택
`setup_env.ps1`은 `nvidia-smi`에서 드라이버의 CUDA 버전을 읽고 다음 순서로 시도합니다.

| 드라이버 CUDA | 시도 순서 |
|---|---|
| ≥ 13.2 | cu132 → cu130 → cu126 → cpu |
| 13.0–13.1 | cu130 → cu126 → cpu |
| 12.6–12.9 | cu126 → cpu (PyTorch 2.14가 CUDA 12.x를 제공하는 마지막 버전) |
| NVIDIA 없음 / 더 오래됨 | cpu |

**PyTorch가 이미 설치되어 있나요?** 스크립트가 기존 설치(예: 전역 `pip`로 설치한 2.14.1)를 찾아 `--system-site-packages`로 `.venv`를 만들어 그대로 재사용하므로 다시 내려받지 않습니다. CPU 버전인데 NVIDIA GPU가 있으면 안내가 나오며, `-FreshTorch -Recreate`로 CUDA 버전을 따로 설치할 수 있습니다.

설치 후 `torch.cuda.is_available()`을 실제로 호출해 확인하고, 한 소스가 실패하면 다음 소스로 넘어갑니다. 모든 출력은 `deploy.log`에 기록됩니다. macOS / Linux에서도 PowerShell 7(`pwsh`)로 같은 스크립트를 실행할 수 있습니다.

## 시각화 학습 UI

```powershell
.\scripts\run.ps1 -Task ui        # http://127.0.0.1:8765 열기 (빌드된 UI 포함, Node.js 불필요)
```

- **신경망 구조도**: 기본 작업 **MLP · 나선 분류**는 모든 뉴런과 가중치를 그립니다(주황 = 양수, 파랑 = 음수, 굵기 = |w|). 노드 색은 현재 샘플의 해당 층 활성값이며, 청록색 펄스 = 순전파, 자홍색 펄스 = 역전파(기울기가 가장 큰 연결을 따라)입니다. ViT / Transformer / U-Net / DiT는 실제 실행 순서대로 층마다 한 열씩 **층 흐름도**를 보여 주며, 마우스를 올리면 모양과 파라미터가, 클릭하면 해당 코드가 열립니다.
- **학습 반복 한 번의 전 과정**: ① 데이터 → ② 순전파 → ③ 손실 → ④ 역전파 → ⑤ 가중치 갱신. 단계마다 의미, 수식(교차 엔트로피 / MSE / 연쇄 법칙 / AdamW), 해당 코드 줄을 보여 줍니다.
- **왼쪽**: 실시간 loss / 정확도 / 기울기 노름 차트, 초당 반복 수, 학습 결과 미리보기(분류 결과, 시퀀스 뒤집기, 확산: 원본 → 노이즈 → 복원).
- **오른쪽**: **● 한 단계 설명**을 누르면 백엔드가 다음 반복을 한 줄씩 기록한 뒤 슬로모션으로 재생합니다. 코드 패널이 현재 줄과 주석을 강조하고, 아래에 그 줄이 만든 텐서(모양, 평균 ± 표준편차, 범위, 분포)와 호출 스택을 보여 줍니다. 스페이스 = 재생 / 일시정지, ← → = 한 단계.
- **버벅이지 않는 이유**: 학습은 백그라운드 스레드에서 실행되고, 지표는 100 ms마다 묶어서 전송(초당 ≤ 10회, 초과 시 자동 축소)하며, 차트는 Canvas로 그립니다. 줄 단위 추적은 설명하는 그 한 단계에서만 켜집니다.
- **언어와 레이아웃**: 오른쪽 위의 「中 / EN」으로 중국어 / 영어를 전환합니다(백엔드 메시지, 로그, 작업 설명도 함께 번역되며 코드 주석은 중국어로 유지). 두 열 사이, 코드와 아래 패널 사이의 구분선을 끌어 크기를 바꿀 수 있고, 왼쪽 열의 각 카드는 아래 가장자리를 끌어 높이를 조절합니다(더블클릭으로 복원). **기억 저장소**와 **한 줄씩 설명** 패널은 탭 전환, **나란히** 보기, 끌어서 옮기고 크기를 바꿀 수 있는 **떠 있는 창**, 별도 브라우저 창으로 **분리**(두 번째 모니터 등)가 가능합니다. 레이아웃은 자동으로 기억되며 오른쪽 위 ⟲로 기본값을 복원합니다.
- **ONNX 내보내기**: 오른쪽 위 버튼, `exports/<작업>.onnx`에 저장(`pip install -r requirements-export.txt` 필요).

프런트엔드 개발(React + Vite, Node.js 필요):

```powershell
.\scripts\run.ps1 -Task ui-dev     # 백엔드 :8765 + Vite 핫 리로드 :5173
.\scripts\run.ps1 -Task build-ui   # web\dist 다시 빌드
```

## 데이터 가져오기와 분석

상단에서 **数据导入与分析 (데이터 가져오기와 분석)** 으로 전환한 뒤, 로컬 경로를 입력하거나 *파일 업로드* / *폴더 업로드*를 누르거나, 파일·폴더를 상자로 끌어다 놓으세요.

| 데이터 | 형식 | 자동 처리 |
|---|---|---|
| 표 | CSV / TSV / TXT / JSON / JSONL / Excel(.xlsx, pandas 불필요) / .xls\* / Parquet\* | 열 유형 자동 인식: 수치(결측 → 평균), 범주(원-핫), 텍스트(해시 단어 가방), ID / 상수 열(제외). 기본 목표는 마지막 열이며 바꿀 수 있음 |
| 이미지 | 폴더 전체 또는 zip | 16–64 픽셀로 자르고 크기 조정, 흑백 / 컬러 판별 → ViT |
| 오디오 | `.wav`(PCM / 부동소수점); `.flac .ogg .mp3`는 `pip install soundfile` 필요 | 음향 특징 56개 추출 → MLP. 메타데이터의 수치 라벨로 회귀 가능 |
| 배열 | `.npz` / `.npy`: `X`+`y`, `x_train/y_train/x_test/y_test`, 폴더 안의 `X.npy`+`y.npy`, 또는 2차원 배열 하나(마지막 열 = 목표) | 2차원 → MLP, 이미지 모양 → ViT |

\* .xls / Parquet는 `pip install -r requirements-data.txt` 필요.

**이미지 / 오디오 라벨은 어디서 오나요** (다음 우선순위로 자동 인식되며 UI에 표시됨):

1. **메타데이터 표**: 폴더 안의 CSV / Excel / JSON 중 파일 이름 열(`img001.jpg`, `images/img001.jpg`, 확장자 없이도 가능)과 라벨 열이 있는 표. 예: `labels.csv`: `filename,breed,weight`. UI의 「라벨 열」에서 바꿀 수 있습니다.
2. **클래스 하위 폴더**: `데이터\고양이\*.jpg`, `데이터\개\*.jpg`. `train\ val\ test\` 아래에 클래스를 나눈 구조도 지원합니다.
3. **파일 이름 접두사**: `cat_001.jpg`, `dog.12.jpg` → `cat` / `dog`.

여러 번 업로드한 폴더는 「현재 데이터셋」으로 합쳐집니다. cat 폴더를 올린 뒤 dog 폴더를 올리면 = 2개 클래스. 업로드한 파일은 `data/uploads/<배치>/`에 저장됩니다.

**분석 내용**: 샘플 수와 학습 / 검증 분할, 목표 분포, 클래스 불균형 경고, 열별 유형 / 결측 / 통계 / 분포, 특징과 목표의 관련도(분류는 상관비 η², 회귀는 |피어슨 상관|), PCA 2차원 투영, 샘플 미리보기.

**학습**: 「이 데이터로 학습」을 누르면 *사용자 정의* 작업이 자동으로 만들어집니다(벡터 → MLP, 이미지 → ViT; 분류는 교차 엔트로피, 회귀는 MSE). 구조도, 한 단계 설명, 5단계 전 과정이 그대로 동작합니다.

**모델 성능 분석**: 정확도 / R²·MAE·RMSE, 혼동 행렬, 클래스별 정밀도 / 재현율 / F1, 순열 특징 중요도, 가장 크게 틀린 샘플.

> 페이지에 "백엔드가 구버전"이라는 안내가 뜨면: 백엔드를 실행 중인 PowerShell 창을 닫거나(또는 Ctrl+C) `.\scripts\run.ps1 -Task ui`를 다시 실행한 뒤 페이지를 새로고침하세요.

## 기계 기억 저장소 (Neural memory store)

학습 화면 오른쪽 아래의 **🧠 机器记忆库 (기억 저장소)**. 학습 중 N 스텝마다(기본값: 이번 학습 스텝 수의 1/40) 네트워크의 현재 상태를 로컬 데이터베이스 `data/memory/neuro_memory.db`(SQLite, Python 내장)에 기록합니다. 가중치 파일은 `data/memory/ckpt/<실행>/latest.pt`와 `best.pt`에 저장됩니다.

| 테이블 | 내용 |
|---|---|
| `runs` | 학습 1회당 1행: 작업, 데이터셋, 하이퍼파라미터, 어느 스냅샷에서 이어서 했는지, 최고 검증 성적 |
| `snapshots` | 스냅샷별: loss, 학습 / 검증 정확도, 기울기 노름, 학습률, 가중치 총 변화량, 체크포인트 |
| `layer_states` | 층별: 가중치 노름, 기울기 노름, 이전 스냅샷 대비 변화량 |
| `samples` | 샘플 메타데이터: 원본 파일, 라벨, 학습 / 검증 |
| `embeddings` | 샘플별 특징 벡터(끝에서 두 번째 층, float32) + 예측 + 신뢰도 + 정답 여부 |
| `sample_memory` | 여러 번의 학습에 걸쳐 쌓이는 "오답 노트": 평가 횟수, 틀린 횟수, 평활 손실 |

**다음 학습에서 이 기억을 쓰는 방법** (패널의 옵션):

- **시작점 = 최고 기억 / 특정 스냅샷에서 이어서**: 가중치 + AdamW 모멘텀 + 스텝 수를 불러와 처음부터가 아니라 이어서 학습;
- **어려운 예제 다시 학습(리플레이)**: 오답 노트로 학습 샘플에 가중치를 두어 뽑기 — 예전에 틀린 샘플을 더 많이 연습;
- **이미지 증강**: 무작위 좌우 반전 + 이동으로, 샘플이 적을 때 "암기"를 줄임(학습 100 %인데 검증이 낮은 것이 바로 이 문제);
- **검증 정확도가 가장 높은** 스냅샷은 best로 따로 저장되어 과적합 후에도 되돌릴 수 있음.

패널에서 볼 수 있는 것: 성장 곡선(학습 간 비교 + 기억 계보), 특징 벡터 공간(스냅샷별 PCA 투영 + 분리도 + 원본 레코드), 층별 변화 히트맵, 오답 노트, 테이블 구조와 읽기 전용 SQL 콘솔.

## 온라인 배포

UI는 **Vercel**, PyTorch 학습 백엔드는 **Hugging Face Spaces**(Docker)에서 실행되며, GitHub에 푸시하면 양쪽 모두 자동으로 다시 배포됩니다. 자세한 단계는 [DEPLOY.ko.md](DEPLOY.ko.md)를 보세요.

```powershell
.\scripts\publish_github.ps1 -User 내-GitHub-이름     # github.com/내-GitHub-이름/Visual-Network-Neuron 으로 푸시
```

## Python 사용법

```python
import torch, neurocore as nc
from neurocore.diffusion import GaussianDiffusion

vit = nc.build_model("vit_small", img_size=224, num_classes=10)
logits = vit(torch.randn(2, 3, 224, 224))

seq2seq = nc.build_model("transformer", src_vocab_size=8000, tgt_vocab_size=8000)

diffusion = GaussianDiffusion(timesteps=1000, schedule="cosine")
dit = nc.build_model("dit_s_2", img_size=32, in_channels=4, num_classes=1000)
loss = diffusion.training_loss(dit, torch.randn(8, 4, 32, 32), torch.randint(0, 1000, (8,)))
samples = diffusion.sample(dit, (4, 4, 32, 32), y=torch.tensor([1, 2, 3, 4]), cfg_scale=4.0)

unet = nc.build_model("unet", in_channels=3, base_channels=64)      # 확산 디노이징
seg = nc.build_model("unet_seg", in_channels=3, num_classes=21)     # 이미지 분할
```

등록된 모델: `transformer`, `transformer_encoder`, `vit`, `vit_tiny/small/base`, `unet`, `unet_seg`, `dit`, `dit_s_2/b_2/xl_2`, `mlp`.

어텐션은 학습하기 쉽도록 기본적으로 명시적인 행렬 연산으로 작성되어 있습니다. `use_sdpa=True`를 주면 PyTorch 융합 커널로 바뀝니다(두 방식의 결과가 같은지 테스트로 확인).

## 새 분야로 확장하기

1. `neurocore/extensions/_template.py`를 복사해 예를 들어 `gnn.py`로 이름을 바꿉니다(밑줄 제거).
2. `@register_model("my_net", domain="graph")`로 등록합니다.
3. `import neurocore` 시 자동으로 불러오며 `nc.build_model("my_net")`으로 사용합니다. 확장 하나에 오류가 있어도 경고만 나오고 코어에는 영향이 없습니다.

예약된 분야 태그: `nlp, vision, generative, graph, audio, multimodal, timeseries, recsys, security, rl, basic, other`.

## 디렉터리 구조

```
NeuroCore/
├─ deploy.ps1                  명령 하나로 점검 → 설치 → 데모
├─ DEPLOY.md                   온라인 배포 (Vercel + Hugging Face)
├─ Dockerfile · vercel.json    클라우드 배포 설정
├─ scripts/
│  ├─ common.ps1               공용 함수 (Python 찾기, GPU 읽기, 설치 소스 선택)
│  ├─ check_env.ps1            환경 점검
│  ├─ setup_env.ps1            설치 (.venv + PyTorch + 의존성 + 테스트)
│  ├─ run.ps1                  demo / test / list / info / ui / build-ui
│  ├─ publish_github.ps1       GitHub로 한 번에 푸시
│  └─ vercel_build.mjs         Vercel 빌드 단계
├─ neurocore/
│  ├─ registry.py              모델 레지스트리
│  ├─ layers/                  어텐션, 임베딩
│  ├─ models/                  transformer / vit / unet / dit / mlp
│  ├─ diffusion/               DDPM
│  ├─ export.py                ONNX 내보내기
│  └─ extensions/              향후 분야 모듈 (자동 탐색)
├─ server/                     시각화 백엔드 (Starlette + SSE)
│  ├─ tasks.py                 학습 작업 (한 줄씩 주석, UI에서 추적·설명)
│  ├─ trainer.py               백그라운드 학습 스레드, 일시정지 / 재개 / 추적 / 기억 기록
│  ├─ tracer.py                sys.settrace 줄 단위 추적기
│  ├─ graph.py · pipeline.py   구조도와 5단계 전 과정
│  ├─ datasets.py · evaluate.py 데이터 가져오기·분석, 모델 평가
│  ├─ memory.py                기계 기억 저장소 (SQLite)
│  ├─ config.py                클라우드 설정 (암호, CORS, 데이터 디렉터리)
│  ├─ hub.py                   전송 속도 제한 (초당 ≤ 10회)
│  └─ app.py                   HTTP API + UI
├─ web/                        React UI (web/dist는 빌드 결과)
├─ tests/                      모델 / 백엔드 / 데이터 / 기억 저장소 테스트
└─ demo.py
```
