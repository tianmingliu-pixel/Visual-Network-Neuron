# 입문 가이드: 내 컴퓨터에서 NeuroCore 실행하기

[中文](GETTING_STARTED.md) | [English](GETTING_STARTED.en.md) | **한국어** | [Deutsch](GETTING_STARTED.de.md)

이 가이드는 **이 저장소에 처음 온 사람**을 위한 것입니다. 순서대로 따라 하면 약 15–30분(대부분 PyTorch 다운로드 시간) 뒤에 내 컴퓨터에서 신경망을 학습시키고, 웹 페이지에서 학습 과정을 볼 수 있습니다.

> 아래 경로는 모두 `C:\NeuroCore`를 예로 듭니다. 다른 위치도 괜찮지만 **영문, 공백 없는 짧은 경로**가 가장 문제가 적습니다.

---

## 먼저: 웹 페이지와 백엔드의 관계

```
웹 페이지(화면)  https://visual-network-neuron.vercel.app   또는   http://127.0.0.1:8765
     │  표시만 담당: 버튼, 곡선, 네트워크 구조도
     ▼
백엔드(계산)  내 컴퓨터의 python -m server (127.0.0.1:8765)
        실제로 신경망 학습, 데이터 분석, "기계 기억 저장소" 저장
```

- **웹 페이지는 화면일 뿐**이며 스스로 계산하지 않습니다.
- **일은 백엔드가 합니다.** 이 가이드는 백엔드를 **내 컴퓨터**에 설치하는 방법입니다.
- `127.0.0.1`은 "이 컴퓨터 자신"이라는 뜻으로, 인터넷의 다른 사람은 접근할 수 없습니다.

---

## 0단계: 준비 (한 번만)

| 필요한 것 | 확인 방법 | 없으면 |
|---|---|---|
| Windows 10 / 11 | — | macOS / Linux는 맨 아래 「Windows가 아니라면?」 참고 |
| Python 3.10 – 3.14 | PowerShell에서 `python --version` 입력 | <https://www.python.org/downloads/>에서 설치하고 **「Add python.exe to PATH」에 체크**, 또는 `winget install Python.Python.3.12` |
| Git (선택) | `git --version` | 없어도 됩니다. 2단계에서 ZIP으로 받을 수 있습니다 |
| NVIDIA 그래픽카드 (선택) | — | 없어도 CPU로 학습할 수 있으며 조금 느릴 뿐입니다 |

**PowerShell 여는 법:** `Win` 키 → `PowerShell` 입력 → Enter.

---

## 1단계: 폴더 만들기

PowerShell에서 입력합니다(한 줄마다 Enter):

```powershell
mkdir C:\NeuroCore
cd C:\NeuroCore
```

이제 `C:\NeuroCore` 안에 있습니다. 이후 모든 명령은 **이 폴더 안에서** 실행합니다.

---

## 2단계: 코드를 이 폴더에 내려받기

**방법 A: Git (추천, 나중에 업데이트가 쉬움)**

```powershell
cd C:\
git clone https://github.com/tianmingliu-pixel/Visual-Network-Neuron.git NeuroCore
cd C:\NeuroCore
```

> 1단계에서 빈 `C:\NeuroCore`를 이미 만들었어도 `git clone`이 그대로 사용합니다(비어 있기만 하면 됩니다).

**방법 B: ZIP 다운로드**

1. <https://github.com/tianmingliu-pixel/Visual-Network-Neuron> 열기
2. 초록색 **Code** 버튼 → **Download ZIP**
3. 압축을 풀면 `Visual-Network-Neuron-main` 폴더가 생깁니다
4. **그 안의 모든 파일**을 `C:\NeuroCore`로 복사합니다

**확인:** `dir`을 실행하면 `README.md`, `scripts`, `server`, `web` 등이 보여야 합니다. `Visual-Network-Neuron-main` 폴더 하나만 보이면 한 단계가 더 들어간 것이니 안의 파일을 밖으로 옮기세요.

---

## 3단계: 스크립트 실행 허용 (새 PowerShell 창마다)

Windows는 기본적으로 `.ps1` 스크립트 실행을 막습니다. 아래 명령은 **현재 창에만** 적용되며, 창을 닫으면 원래대로 돌아오고 시스템 설정은 바뀌지 않습니다.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

ZIP으로 받았다면 한 번 더 실행합니다("인터넷에서 받은 파일" 표시 해제):

```powershell
Get-ChildItem C:\NeuroCore -Recurse | Unblock-File
```

---

## 4단계: 환경 한 번에 설치 (한 번만)

```powershell
cd C:\NeuroCore
.\scripts\check_env.ps1      # 먼저 점검: 항목마다 PASS / WARN / FAIL, FAIL 뒤에 해결 방법이 나옵니다
.\scripts\setup_env.ps1      # 설치: C:\NeuroCore\.venv를 만들고 그래픽카드에 맞는 PyTorch를 자동 설치
```

- PyTorch(수백 MB ~ 2 GB 이상)를 내려받으니 기다려 주세요.
- 모든 것이 `C:\NeuroCore\.venv`에 설치되어 **컴퓨터의 다른 Python에 영향을 주지 않습니다**.
- 마지막 자체 점검이 모두 통과하면 성공입니다.

CPU 버전을 강제로 쓰려면(그래픽 드라이버 문제 등): `.\scripts\setup_env.ps1 -Cuda cpu`

---

## 5단계: 백엔드 시작 (쓸 때마다)

```powershell
cd C:\NeuroCore
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\run.ps1 -Task ui
```

- `NeuroCore UI -> http://127.0.0.1:8765`가 보이면 백엔드가 시작된 것입니다.
- 브라우저가 `http://127.0.0.1:8765`를 자동으로 엽니다.
- **이 PowerShell 창을 닫지 마세요.** 닫으면 백엔드가 멈춥니다. 다 쓰면 `Ctrl + C`를 누르거나 창을 닫습니다.

---

## 6단계: 웹 페이지를 열고 내 컴퓨터를 쓰는지 확인

두 주소 모두 **똑같이** 동작합니다.

| 주소 | 설명 |
|---|---|
| `http://127.0.0.1:8765` | 내 백엔드가 직접 제공하는 페이지. 5단계에서 자동으로 열리며 **인터넷 없이도 동작** |
| `https://visual-network-neuron.vercel.app` | 온라인 페이지. 내 컴퓨터에서 실행 중인 백엔드를 **자동으로 찾습니다** |

열리면 **먼저 오른쪽 위를 보세요.**

| 오른쪽 위 표시 | 의미 | 할 일 |
|---|---|---|
| `● 后端 · 本页面` (this page) | 내 컴퓨터 사용 중 (127.0.0.1:8765에서 열었을 때) | ✅ 없음 |
| `● 后端 · 本机` (local) | 내 컴퓨터 사용 중 (온라인 페이지에서 열었을 때) | ✅ 없음 |
| `● 后端 · 网站后端 · 只读观看` (site backend · view only)와 「👀 观看模式」 | **작성자의 컴퓨터**에 연결됨, 보기만 가능 | ❌ 내 백엔드가 꺼져 있음 → 5단계로 |
| `○ … · 未连接` (disconnected) | 어떤 백엔드에도 연결 안 됨 | ❌ 5단계로, 창이 열려 있는지 확인 |

> 화면 언어는 오른쪽 위 **中 / EN**으로 바꿀 수 있습니다(현재 중국어 / 영어 지원). 아래 표의 괄호 안은 영어 화면의 표시입니다.

### 백엔드 패널의 각 옵션 뜻 (보통은 건드리지 않아도 됨)

오른쪽 위 **后端 (Backend)** 버튼을 누르면 「后端连接 (Backend connection)」 패널이 열립니다.

| 옵션 | 의미 | 언제 쓰나 |
|---|---|---|
| 后端地址 (Backend URL) | 페이지가 백엔드를 찾을 주소 | 보통 **비워 두고** 자동 선택에 맡깁니다 |
| 本页面同源 (Same as this page) | 이 페이지를 제공한 바로 그 컴퓨터 | `127.0.0.1:8765`에서 열었을 때 |
| 本机 (local) `http://127.0.0.1:8765` | **내 컴퓨터** | 온라인 페이지가 내 백엔드를 못 찾았을 때 직접 클릭 |
| 网站默认后端 (Site default backend) | 작성자의 컴퓨터(터널 경유) | 작성자의 학습을 **구경**만 할 때 |
| 口令 (Access token) | 작성자 백엔드의 암호 | **내 컴퓨터를 쓸 때는 비워 둡니다. 암호 필요 없음** |
| 自动选择 (Auto-select) | 수동 설정을 지우고 자동으로 되돌림 | 설정이 꼬였을 때 |
| 保存并重新连接 (Save & reconnect) | 위 설정을 저장하고 새로 고침 | 무언가 바꾼 뒤 |

**자동 선택 순서:** ① 직접 지정한 주소 → ② 페이지 자신의 주소 → ③ 내 컴퓨터의 `127.0.0.1:8765` → ④ 작성자의 컴퓨터(보기 전용). 따라서 5단계의 창이 열려 있는 한 페이지는 자동으로 내 컴퓨터를 사용합니다.

---

## 7단계: 첫 학습

1. 위쪽에서 **训练 (Training)** 탭을 선택합니다.
2. 드롭다운에서 작업을 고릅니다. 처음이라면 **MLP · 나선 분류(뉴런 단위)** 추천.
3. 단계 수, 학습률, 배치 크기는 기본값 그대로 두고 **▶ 开始训练 (Start training)** 을 누릅니다.
4. 네트워크 구조도에서 뉴런과 가중치가 바뀌고, 손실이 내려가고, 정확도가 오르고, 「逐行讲解 (Line-by-line)」이 실행 중인 코드 줄을 강조하고, 오른쪽 아래 「机器记忆库 (Memory store)」가 스냅숏을 기록하는 것을 볼 수 있습니다.

**내 데이터 사용:** 위쪽 **数据 (Data)** 탭 → 폴더 업로드(예: `cat\`, `dog\` 하위 폴더의 이미지, 또는 레이블 열이 있는 표) → 加载并分析 (Load & analyse) → 학습. 데이터는 `C:\NeuroCore\data\`에만 저장됩니다.

---

## 왜 내 컴퓨터를 백엔드로 쓰나요?

| | 내 컴퓨터 | 작성자의 컴퓨터(구경) |
|---|---|---|
| 암호 | **필요 없음** | 작성자의 암호가 없으면 보기만 가능 |
| 학습 / 업로드 / 내보내기 | ✅ 모두 가능 | ❌ 보기만 |
| 내 데이터의 위치 | **내 컴퓨터에만** (`C:\NeuroCore\data`) | 다른 사람의 컴퓨터로 전송 |
| 속도 | 내 CPU / GPU, 네트워크를 거치지 않음 | 작성자의 컴퓨터와 네트워크에 따라 다름 |
| 사용 가능 시간 | 언제든, 인터넷 없이도 | 작성자가 컴퓨터와 터널을 켜 둔 동안만 |
| 기계 기억 저장소 | 내 것, 학습할수록 쌓이고 이어서 학습 가능 | 볼 수 없음 |

---

## 다음부터 쓸 때: 세 줄이면 됩니다

```powershell
cd C:\NeuroCore
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\run.ps1 -Task ui
```

**최신 버전으로 업데이트**(Git으로 받은 경우): `cd C:\NeuroCore` → `git pull` → 다시 시작. ZIP이라면 다시 받아서 덮어쓰면 됩니다(`data\` 폴더에는 내 데이터와 기억 저장소가 있으니 지우지 마세요).

---

## 문제 해결

| 증상 | 해결 |
|---|---|
| `…ps1 파일을 로드할 수 없습니다. 이 시스템에서 스크립트를 실행할 수 없으므로…` | 3단계 실행 |
| `python`을 인식할 수 없음 | 「Add to PATH」에 체크하고 Python을 다시 설치한 뒤 PowerShell을 **다시 열기** |
| `setup_env.ps1`이 Python 버전 오류 | 3.10–3.14 필요: `winget install Python.Python.3.12` |
| 8765 포트 사용 중 | 이전 `run.ps1` 창을 닫거나, `.\scripts\run.ps1 -Task ui -Port 8766` 실행 후 백엔드 패널에 `http://127.0.0.1:8766` 입력 |
| 온라인 페이지가 계속 「보기 전용」 또는 「연결 안 됨」 | 5단계 창이 열려 있는지 확인 → 백엔드 패널에서 「本机 (local)」→「保存并重新连接」. 그래도 안 되면 `http://127.0.0.1:8765`를 직접 여세요(Safari 등 일부 브라우저는 온라인 페이지가 내 컴퓨터에 접근하는 것을 막습니다) |
| `UnicodeEncodeError: 'charmap' codec` | 이전 버전 문제이므로 최신 코드로 업데이트 |
| 「开始训练」을 눌러도 반응 없음 | 맨 아래 「日志 (Log)」의 빨간 글씨 확인. 「암호 필요」라면 작성자 컴퓨터에 연결된 것 → 6단계로 |

---

## Windows가 아니라면? (macOS / Linux)

스크립트는 PowerShell용이므로 macOS / Linux에서는 같은 단계를 직접 실행합니다.

```bash
mkdir -p ~/NeuroCore && cd ~/NeuroCore
git clone https://github.com/tianmingliu-pixel/Visual-Network-Neuron.git .
python3 -m venv .venv
source .venv/bin/activate
pip install torch            # NVIDIA 그래픽카드가 있으면 https://pytorch.org 의 명령을 사용
pip install -r requirements.txt
python -m server --open      # 백엔드를 시작하고 http://127.0.0.1:8765 를 엽니다
```

다음부터: `cd ~/NeuroCore && source .venv/bin/activate && python -m server --open`

---

더 보기: 프로젝트 소개는 [README.ko.md](README.ko.md), 저장소 주인이 자기 컴퓨터를 사이트 방문자에게 공개하는 방법은 [DEPLOY.ko.md](DEPLOY.ko.md)(일반 사용자는 볼 필요 없음).
