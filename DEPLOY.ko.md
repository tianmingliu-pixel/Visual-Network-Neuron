# 온라인 배포: Vercel(웹 UI) + 내 컴퓨터(학습 백엔드, 무료 Cloudflare 터널)

[中文](DEPLOY.md) | [English](DEPLOY.en.md) | **한국어** | [Deutsch](DEPLOY.de.md)

```
브라우저 ──► https://visual-network-neuron.vercel.app        웹 UI (Vercel, 정적 파일)
              │  /api/… (CORS + 암호)
              ▼
        https://xxxx.trycloudflare.com                       무료 Cloudflare 터널
              ▼
        내 컴퓨터의 python -m server (127.0.0.1:8765)          PyTorch 백엔드: 내 CPU / GPU, 데이터는 내 컴퓨터에
```

> 이 문서는 **저장소 주인**(자기 컴퓨터를 사이트 방문자에게 공개)을 위한 것입니다. 내 컴퓨터에서 쓰기만 하려면 [GETTING_STARTED.ko.md](GETTING_STARTED.ko.md)를 보세요. 아래 `C:\NeuroCore`는 내 프로젝트 폴더로 바꾸세요.

Vercel은 웹 페이지만 올릴 수 있고 PyTorch는 실행할 수 없으며, Hugging Face Docker Space는 이제 유료입니다. 그래서 학습 백엔드는 내 컴퓨터에서 실행하고, Cloudflare 무료 터널로 https 공개 주소를 붙입니다. 비용도 신용카드도 필요 없고 내 GPU를 쓸 수 있습니다.

---

## ① GitHub에 푸시 (업데이트할 때마다)

```powershell
cd C:\NeuroCore
.\scripts\publish_github.ps1 -User tianmingliu-pixel
```

`data\`(업로드, 기억 저장소, 암호), `exports\`, `.venv\`는 올라가지 않습니다. 프런트엔드 코드(`web\src`)를 바꿨다면 먼저 `.\scripts\run.ps1 -Task build-ui`를 실행하세요.

## ② 웹 UI: Vercel (한 번만 설정)

1. <https://vercel.com>에 GitHub로 로그인 → **Add New… → Project** → `Visual-Network-Neuron` 선택 → **Import**.
2. Framework Preset은 **Other**, 나머지는 그대로(`vercel.json`이 의존성 설치 없이 `web/dist`를 게시) → **Deploy**.
3. 예: `https://visual-network-neuron.vercel.app` 주소를 받습니다.

이후 GitHub에 푸시할 때마다 자동으로 다시 게시됩니다.

## ③ 백엔드: 내 컴퓨터에서 공개 (사이트를 쓸 때마다 실행)

```powershell
.\scripts\serve_public.ps1 -Publish
```

스크립트가 하는 일:

1. 처음 실행할 때 **암호**를 정하게 합니다(`data\.public_token`에 저장, 업로드되지 않음);
2. Cloudflare `cloudflared`를 자동 설치(winget);
3. 백엔드를 백그라운드로 시작하고 터널을 열어 `https://xxxx.trycloudflare.com` 주소를 얻습니다(클립보드에 복사);
4. `-Publish`: 그 주소를 `deploy/backend_url.txt`에 써서 푸시 — 약 1분 뒤 Vercel이 갱신되어 **다른 사람도 자동으로 연결**됩니다;
5. 사이트(고정 주소 `https://visual-network-neuron.vercel.app`)를 엽니다. 내 컴퓨터에서 열면 로컬 백엔드에 자동으로 연결됩니다. 오른쪽 위 **后端 (백엔드)** 에서 암호를 한 번 입력하면 학습·업로드할 수 있습니다(브라우저가 기억합니다).

PowerShell 창을 닫거나 Ctrl+C를 누르면 백엔드와 터널이 멈춥니다.

**주소가 바뀌나요?** 사이트 주소 `visual-network-neuron.vercel.app`은 절대 바뀌지 않습니다. 터널 주소 `xxxx.trycloudflare.com`은 시작할 때마다 무작위로 바뀌지만 뒤에서만 쓰이며, `-Publish`가 새 주소를 사이트에 알려 줍니다. 다른 사람에게는 Vercel 주소만 공유하고, 본인도 Vercel 주소로 여세요(터널 주소로 열지 마세요: 암호는 주소별로 저장되므로 주소가 바뀌면 다시 입력해야 합니다).

## 사이트를 연 뒤: 백엔드 자동 선택

웹 페이지는 화면일 뿐이고, 학습과 데이터 분석은 백엔드에서 실행됩니다. 페이지를 열면 다음 순서로 백엔드를 **자동으로** 찾습니다.

| 상황 | 연결 대상 | 오른쪽 위 표시 | 가능한 작업 |
|---|---|---|---|
| 방문자 자신의 컴퓨터에서 백엔드 실행 중(`run.ps1 -Task ui`) | 방문자 컴퓨터 `127.0.0.1:8765` | 后端 · 本机 (로컬) | 모든 기능, 방문자의 CPU/GPU 사용, 데이터는 그 컴퓨터에만 |
| 로컬 백엔드 없음, 내 터널 실행 중 | 내 컴퓨터(터널 경유) | 网站后端 · 只读观看 (보기 전용) | 보기만 가능, 암호 입력 후 학습·업로드 |
| 둘 다 없음 | — | 未连接 (연결 안 됨) | 화면만 보이고 실행 불가 |

직접 지정: 오른쪽 위 **后端** → 백엔드 주소 입력 → 「保存并重新连接 (저장 후 재연결)」. 「自动选择 (자동 선택)」을 누르면 자동으로 돌아갑니다.

### 암호 입력 위치

1. 맨 오른쪽 위 **后端** 버튼(中/EN 전환 오른쪽)을 눌러 「백엔드 연결」 패널을 엽니다.
2. **백엔드 주소**는 비워 둡니다(= 자동 / 이 페이지와 같은 주소).
3. **암호** 칸에 `serve_public.ps1`을 처음 실행할 때 정한 암호를 넣습니다(잊었다면 `Get-Content data\.public_token`).
4. 「저장 후 재연결」을 누릅니다. 「보기 전용」이 사라지면 학습할 수 있습니다.

> 「학습 시작」을 눌러도 반응이 없고 로그에 「암호 필요」가 보이면 = 암호를 아직 입력하지 않은 것이며, 파일이 빠진 것이 아닙니다.

### 다른 사람이 자기 컴퓨터로 학습하려면

오른쪽 위에 「👀 观看模式 · 自己训练？(보기 모드 · 직접 학습?)」이 표시되며, 누르면 단계가 나옵니다.

1. 코드 다운로드: <https://github.com/tianmingliu-pixel/Visual-Network-Neuron>
2. 코드 폴더에서 `.\scripts\setup_env.ps1` 실행 후 `.\scripts\run.ps1 -Task ui` 실행
3. 사이트 새로 고침: 자기 컴퓨터의 백엔드로 자동 전환(암호 불필요, 데이터가 내 컴퓨터로 오지 않음)

## 문제 해결

| 증상 | 원인 / 해결 |
|---|---|
| `UnicodeEncodeError: 'charmap' codec can't encode` | 이전 버전이 로그에 중국어를 Windows 기본 인코딩으로 썼기 때문입니다. 수정됨(백엔드 UTF-8 강제, 스크립트에서 `PYTHONUTF8=1`). 최신 코드를 받으세요 |
| 「백엔드 시작 실패」 | `data\backend.err.log` 확인. 8765 포트가 사용 중이면 다른 `run.ps1 -Task ui` 창을 먼저 닫으세요 |
| 사이트에 「연결 안 됨」 | `serve_public.ps1`이 실행 중이 아니거나, 방금 `-Publish`로 푸시해서 Vercel이 갱신 중(약 1분) |
| 학습 버튼 반응 없음 | 암호 미입력(위 「암호 입력 위치」 참고) |

## 보안

| 누구 | 할 수 있는 것 |
|---|---|
| 나(암호 있음) | 학습, 데이터 업로드, 기억 저장소 보기 / 삭제, 로컬 경로 사용 |
| 다른 사람(암호 없음) | 학습 과정 보기만 가능(차트, 구조도, 한 줄씩 설명). 내 데이터·업로드·기억 저장소는 볼 수 없음 |

백엔드는 127.0.0.1에서만 대기하며 외부에서는 터널로만 접근할 수 있습니다. `serve_public.ps1`을 실행하지 않으면 아무것도 공개되지 않습니다.

## 주소를 고정하고 싶다면

- **고정 주소**: Cloudflare에 내 도메인을 추가하고 named tunnel(`cloudflared tunnel create`)을 사용.
- **클라우드 서버**: 저장소의 `Dockerfile`은 어떤 Docker 호스트에서도 실행됩니다(Hugging Face PRO, Google Cloud Run, Render 유료 요금제 등, 메모리 2 GB 이상). Vercel에 `NEUROCORE_API_BASE`를 설정하면 됩니다.

## 관련 파일

| 파일 | 역할 |
|---|---|
| `scripts/publish_github.ps1` | 한 번에 커밋하고 GitHub로 푸시 |
| `scripts/serve_public.ps1` | 백엔드 + Cloudflare 터널을 시작하고 사이트에 주소를 알림 |
| `vercel.json`, `scripts/vercel_build.mjs` | Vercel이 `web/dist`를 게시. 백엔드 주소는 `NEUROCORE_API_BASE` 또는 `deploy/backend_url.txt` |
| `server/config.py` | 암호, CORS, 읽기 보호, 데이터 디렉터리 |
| `Dockerfile` | 나중에 클라우드 서버에 올릴 때 사용 |
