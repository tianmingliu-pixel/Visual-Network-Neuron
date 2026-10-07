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

Vercel은 웹 페이지만 올릴 수 있고 PyTorch는 실행할 수 없으며, Hugging Face Docker Space는 이제 유료입니다. 그래서 학습 백엔드는 내 컴퓨터에서 실행하고, Cloudflare 무료 터널로 https 공개 주소를 붙입니다. 비용도 신용카드도 필요 없고 내 GPU를 쓸 수 있습니다.

---

## ① GitHub에 푸시 (업데이트할 때마다)

```powershell
cd D:\网络神经测试\核心\NeuroCore
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
5. `내-사이트/?api=터널-주소`를 엽니다. 오른쪽 위 **后端 (백엔드)** 에서 암호를 입력하면 학습·업로드할 수 있습니다.

PowerShell 창을 닫거나 Ctrl+C를 누르면 백엔드와 터널이 멈춥니다. 무료 터널 주소는 매번 바뀌므로 매번 `-Publish`로 갱신하세요.

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
