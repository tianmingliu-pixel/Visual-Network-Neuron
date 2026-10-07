# 온라인 배포: GitHub → Vercel(웹 UI) + Hugging Face Spaces(학습 백엔드)

[中文](DEPLOY.md) | [English](DEPLOY.en.md) | **한국어** | [Deutsch](DEPLOY.de.md)

```
브라우저 ──► https://visual-network-neuron.vercel.app           웹 UI (Vercel, 정적 파일)
              │  /api/… (CORS + 암호)
              ▼
        https://<HF사용자>-visual-network-neuron.hf.space       PyTorch 학습 백엔드 (Docker, 무료 CPU)
```

Vercel은 웹 페이지만 올릴 수 있고 PyTorch는 실행할 수 없습니다(용량이 너무 크고, 오래 실행할 수 없고, 디스크가 없음). 그래서 학습 백엔드는 Hugging Face의 무료 Docker Space(2코어 CPU, 16 GB 메모리)에서 실행합니다. GitHub에 푸시하면 **양쪽 모두 자동으로 다시 배포**됩니다.

---

## ① GitHub에 푸시 (업데이트할 때마다 이 단계)

```powershell
cd D:\网络神经测试\核心\NeuroCore
.\scripts\publish_github.ps1 -User tianmingliu-pixel
```

- `https://github.com/tianmingliu-pixel/Visual-Network-Neuron`으로 푸시합니다. 처음에는 GitHub 로그인 창이 뜹니다.
- 스크립트가 `deploy/deploy-backend-hf.yml`을 `.github/workflows/`에 자동으로 넣습니다(백엔드 자동 배포용).
- 저장소에 이미 있는 README 등은 자동으로 병합됩니다. `data\`(업로드한 데이터, 기억 저장소), `exports\`, `.venv\`는 올라가지 않습니다.
- 프런트엔드 코드(`web\src`)를 바꿨다면 먼저 `.\scripts\run.ps1 -Task build-ui`(Node.js 필요)를 실행한 뒤 푸시하세요. Vercel은 저장소의 `web\dist`를 그대로 사용합니다.

## ② 백엔드: Hugging Face Space (한 번만 설정)

1. <https://huggingface.co>에 가입 / 로그인 → 오른쪽 위 아바타 → **New Space**:
   이름 `visual-network-neuron`, **SDK는 Docker → Blank**, 하드웨어 **CPU basic(무료)**, 공개 / 비공개 모두 가능.
2. Space 페이지 → **Settings → Variables and secrets**:
   - **New secret** `NEUROCORE_TOKEN` = 직접 정한 암호(암호가 있어야 학습 / 업로드 / 기억 삭제 가능, 없는 방문자는 보기만 가능)
   - (선택) **New variable** `NEUROCORE_ALLOWED_ORIGINS` = ③단계에서 받은 Vercel 주소, 예: `https://visual-network-neuron.vercel.app`
3. 아바타 → **Settings → Access Tokens → Create new token**, 유형 **Write**, 복사해 둡니다.
4. GitHub 저장소 → **Settings → Secrets and variables → Actions**:
   - **Secrets** 탭: New repository secret `HF_TOKEN` = 위에서 복사한 토큰
   - **Variables** 탭: New repository variable `HF_SPACE` = `내-HF-사용자/visual-network-neuron`
5. GitHub 저장소 → **Actions → Deploy backend to Hugging Face Space → Run workflow**.
   Space 빌드가 시작됩니다(처음에는 PyTorch를 받느라 약 5–10분). 끝나면
   `https://내-HF-사용자-visual-network-neuron.hf.space/api/version`을 열어 `{"version": 4, "cloud": true, …}`가 보이면 성공입니다.

이후 GitHub에 푸시할 때마다 백엔드가 자동으로 동기화되고 다시 빌드됩니다.

## ③ 웹 UI: Vercel (한 번만 설정)

1. <https://vercel.com>에 GitHub 계정으로 로그인 → **Add New… → Project** → `Visual-Network-Neuron` 선택 → **Import**.
2. Framework Preset은 **Other**, 나머지는 그대로 둡니다(저장소의 `vercel.json`이 의존성 설치 없이 `web/dist`를 게시하도록 설정되어 있음).
3. **Environment Variables**를 펼쳐 추가:
   `NEUROCORE_API_BASE` = `https://내-HF-사용자-visual-network-neuron.hf.space`
4. **Deploy**. 예를 들어 `https://visual-network-neuron.vercel.app` 같은 주소를 받습니다.
5. 주소를 열고 → 오른쪽 위 **后端 (백엔드)** → 암호 입력 → 저장. "클라우드 ☁"와 "암호 일치"가 보이면 학습과 데이터 업로드를 시작할 수 있습니다.

이후 GitHub에 푸시할 때마다 Vercel이 자동으로 다시 게시합니다.

## 같은 웹 페이지로 내 컴퓨터에도 연결 가능

오른쪽 위 **后端 (백엔드)** 에서 "本机 (로컬) http://127.0.0.1:8765"를 선택해 저장하면, 웹 페이지가 내 컴퓨터의 백엔드를 사용합니다(먼저 `.\scripts\run.ps1 -Task ui` 실행). 학습은 내 CPU / GPU로 하고 데이터는 컴퓨터 밖으로 나가지 않습니다. "기본값으로 복원"을 누르면 클라우드로 돌아갑니다.

## 무료 클라우드의 제한

| 제한 | 설명 |
|---|---|
| 속도 | 무료 CPU, GPU 없음. 데모 작업과 수백 장의 작은 이미지는 문제없음. 큰 데이터셋은 로컬 백엔드 권장 |
| 절전 | 48시간 동안 방문이 없으면 잠들며, 다시 열면 시작까지 약 1분 걸림 |
| 데이터 비영구 | Space가 재시작되면 업로드한 데이터와 기억 저장소가 지워짐. 보존하려면 Space Settings에서 Persistent storage를 구매하고 변수 `NEUROCORE_DATA_DIR=/data` 추가 |
| 공유 세션 | 모든 방문자가 같은 학습 과정을 봄. 암호가 쓰기 작업을 보호하며, 암호가 없으면 보기만 가능 |
| 보안 | 클라우드 모드에서는 서버에 업로드한 데이터만 분석하며 서버의 임의 경로는 읽을 수 없음. 파일당 업로드 한도 200 MB(`NEUROCORE_MAX_UPLOAD_MB`로 변경) |

## 관련 파일

| 파일 | 역할 |
|---|---|
| `scripts/publish_github.ps1` | 한 번에 커밋하고 GitHub로 푸시 |
| `vercel.json`, `scripts/vercel_build.mjs` | Vercel이 `web/dist`를 게시하고 백엔드 주소를 `config.js`에 기록 |
| `Dockerfile`, `.dockerignore` | 백엔드 이미지 (CPU 버전 PyTorch, 포트 7860) |
| `deploy/deploy-backend-hf.yml` | GitHub Actions 워크플로 템플릿. 푸시 전에 `.github/workflows/`로 복사되고, 푸시할 때마다 Space를 동기화 |
| `server/config.py` | 클라우드 설정: 암호, CORS, 데이터 디렉터리, 업로드 한도 |
