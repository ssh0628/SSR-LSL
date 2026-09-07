# SSR-LSL

- 루트: 사용자 이미지 데이터셋 학습
- `cifar/`: 기존 CIFAR-10 실험 독립 보관
- config: `setting/config.py`
- seed: `0`
- 기본값: pretrained ConvNeXtV2-Tiny, SSR + LSL, Label Wave 저장
- SSR/LSL 수식·sample selection 유지
- 입력: 원본 이미지 직접 읽기 → bbox crop → 224 resize → 증강
- 이미지 픽셀 복제·대용량 RGB 캐시 생성 없음
- multi-ROI offset·scale, 추가 noise, sqrt resampling 미사용

## 구조

```text
run.py         bbox 준비 → 학습 → checkpoint 평가
audit.py
benchmark.py
setting/       config, NPY loader, bbox 좌표·이미지 검사, augmentation, model 구성
models/        backbone
ssr/           relabel, k-NN selection, loss, 학습
lsl/           reverse k-NN, structural loss
label_wave/    prediction change, checkpoint 선택
log/           config, metric, checkpoint 저장
cifar/         기존 구조·실험 큐·설정 복사본
```

- 루트와 `cifar/`는 서로 import하지 않음
- `custom/` 폐기
- 기존 결과 파일은 이동·삭제하지 않음

## 데이터 설정

`setting/config.py` 상단에서 경로 지정:

```python
DATASET_ROOT = Path("/root/project/dataset/npy_path/modify_npy")
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
```

`DataConfig`에서 입력 정보 지정:

```python
class_names = ("A1", "A2", "A3", "A4", "A5", "A6", "A7")
train = SplitConfig("train_path.npy", "train_labels.npy")
validation = SplitConfig("val_path.npy", "val_labels.npy")
test = SplitConfig("test_path.npy", "test_labels.npy")
image_size = 224
crop_bbox = True
```

- paths: 이미지 경로 문자열 NPY, shape `(N,)`
- labels: 정수 class index NPY, shape `(N,)`
- 클래스 순서: `class_names[0]` → label `0`
- 1부터 시작하는 라벨: `label_offset=1`
- 상대 이미지 경로: `image_root` 기준. `None`이면 NPY root 기준
- 기본 파일명은 `*_path.npy`, `*_paths.npy` 둘 다 지원
- 다른 파일명: `SplitConfig`에서 직접 지정
- validation/test가 없으면 해당 필드를 `None`
- `split_config.json`, `classes.json` 필수 아님
- 클래스 수: `class_names`에서 자동 계산

Bbox:

- 학습 시작 시 train·val·test bbox 좌표 NPZ 준비. 기존 유효 cache 재사용
- 이미지 픽셀 미포함: 이미지당 좌표 4개만 저장
- 기본: `*_bboxes_cache_v1.npz`; `*_bbox_cache_v1.npz`도 지원
- 직접 지정: `SplitConfig(paths, labels, bboxes="파일명.npz")`
- 이미지 순서·원본 경로 fingerprint 검증; 다른 NPY와 잘못 연결 방지
- cache 없으면 JSON의 `labelingInfo → box → location[0]`에서 생성
- 기존 cache의 누락 항목: 시작 시 해당 JSON만 재확인·복구
- JSON 탐색: 이미지 옆 `.json`, `.JSON`, `이미지확장자.json`
- 별도 JSON 폴더: `annotation_root` 지정; `image_root` 기준 하위 경로 유지
- `missing_bbox="drop"` 기본: JSON에도 bbox가 없으면 train·val·test에서 제외
- 빈 JSON·깨진 JSON·UTF-8 해석 실패: `drop`에서 제외하고 원인 기록
- 제외: 원본 이미지·NPY 삭제 없이 경로·라벨·bbox를 같은 순서로 필터링
- `missing_bbox="error"`: 좌표 누락 시 목록 출력 후 중단
- `missing_bbox="full"`: 누락 이미지만 전체 입력. 명시적 선택 시에만 허용
- 잘못된 좌표·경로 설정, 이미지 밖 bbox, 제외 후 빈 split: 중단
- 원본 크기로 좌표 clamp. 이미 유효한 cache의 JSON 좌표 수정 시 재생성 필요

이미지 로딩:

- 원본은 디스크에 그대로 보존. worker에서 decode·crop·resize
- crop·resize는 sample당 1회; weak·strong 증강은 view별 독립 적용
- 전체 이미지 디스크 캐시 기능 제거. 기존 서버 캐시 자동 삭제 없음

이미지 검사:

- `verify_images=True`: bbox 필터링 후 사용 이미지 전체 decode 검사
- `allow_truncated_images=True`: 잘린 이미지의 decoder 재시도 허용
- 사전 검사 시 재시도 성공/실패 경로: `data_audit.jsonl`
- 완전히 읽을 수 없는 파일: 경로를 모아서 보고하고 중단
- 검정 이미지 대체 없음. bbox 누락 제외와 이미지 decode 실패는 별도 처리

## 실행

```bash
uv sync
uv run python run.py
```

서버:

```bash
cd /root/project/ssr
nohup /opt/conda/bin/python -u -m run > train.log 2>&1 &
tail -f train.log
```

- 실행 진입점: `run.py` 하나
- bbox 준비·입력 검증 실패 시 모델 생성 전 중단
- `crop_bbox=False`: 전체 이미지 학습
- 기본 `300 epoch`; cosine LR도 300 기준
- 새 실행마다 처음부터 학습. 기존 실행에 자동 resume·설정 반영 없음

검사만 실행:

```bash
uv run python audit.py
```

기존 CIFAR 실험:

```bash
uv run python -m cifar.run
uv run python -m cifar.cifar_ssr
```

- CIFAR 설정/실험표: [cifar/README.md](cifar/README.md)
- 모델 이름: `convnextv2_tiny`, `resnet18`, `resnet34` 등 timm backbone
- 증강: `AugmentationConfig`; 방향이 중요한 데이터는 반전·회전 조절
- LSL: `structural_labels.enabled`
- Label Wave 저장: `label_wave.enabled`
- Label Wave 조기 종료: `label_wave.stop_training`
- GPU별 batch 시작값: config 상단 주석

## H100 실행 설정

- 기준: H100 NVL 1장, 16 vCPU, RAM 200 GB
- 기본 학습: BF16 AMP, channels-last, CUDA fused AdamW
- 기본 batch: train `256`, evaluation `1024`
- train `512`: mixup forward CUDA OOM 확인으로 축소
- worker: 학습 loader당 `16` — 두 loader 동시 `32`; feature 추출·평가 `32`
- 평가 중 all-sample worker `16`개 유휴 상주; 상주 worker는 최대 `48`개
- bbox JSON worker: `16`; 학습과 순차 실행
- prefetch: worker당 `2` batch
- worker 수는 실행 후보값. 실제 속도는 `train_samples/s`, `data_wait_s`로 확인
- FP32 유지: loss 계산, feature 추출, k-NN, confidence, Label Wave·정확도 평가
- TF32 미사용; SSR/LSL 수식·selection 규칙 유지
- `runtime.deterministic=False`: cuDNN autotune; seed는 `0` 유지
- AMP·배치 변경: 기존 FP32 실험과 결과 차이 가능; LR 자동 확대 없음
- CPU/MPS: AMP·channels-last·fused AdamW 미적용

선택 사항 — 배치별 속도 비교:

```bash
python -u benchmark.py
```

- 후보: 파일 상단 `BATCH_SIZES`. 실제 학습과 동시 실행 금지
- scratch 모델·임시 target; 준비 3 step + 측정 40 step
- 출력: step 시간, sample/s, GPU 최대 메모리, OOM
- 전체 epoch·정확도 벤치마크 아님. config 자동 변경 없음
- CUDA AMP 참고: [PyTorch AMP](https://docs.pytorch.org/docs/stable/amp.html)

## 저장

`outputs/<설정 이름>/<실행 시각>/`

- 실행마다 새 폴더. 명시한 `runtime.run_id` 중복은 차단
- `config.json`: 실제 설정과 데이터 경로
- `data_audit.jsonl`: bbox 제외 목록·최종 클래스별 개수·사전 이미지 검사
- bbox 제외 로그의 `index`: 원본 NPY 위치. 학습 인덱스는 남은 순서대로 0부터 재부여
- validation/test 지표: bbox 필터링 후 실제 평가 표본 기준
- checkpoint 최대 4개:

| 선택 | 파일 |
|---|---|
| Best balanced accuracy | `best_balanced_accuracy.pt` |
| Best macro F1 | `best_macro_f1.pt` |
| Last | `last.pt` |
| Label Wave | `label_wave.pt` |

- Best: validation 지표별 독립 선택. 동점은 앞선 epoch 유지
- Last: 마지막 완료 epoch
- Label Wave: prediction change로만 선택. GT 지표 미사용
- 모든 checkpoint: balanced accuracy·macro F1 함께 기록
- validation 없음: Best 생략. LW 비활성·후보 없음: LW 생략
- `metrics.jsonl`: epoch별 상세 지표
  - validation: accuracy, balanced accuracy, macro/weighted/micro F1, precision·recall
  - 클래스별 precision·recall·F1·support, confusion matrix
  - CE, top-2 accuracy, MCC, kappa, confidence, entropy, Brier, ECE
  - loss별 값·가중 합, encoder/head LR, 선택·제외·라벨 변경 수·전이
  - 클래스별 예측·선택 분포, 제공된 train 라벨과의 일치 지표
  - 구간 시간, 처리량, 데이터 대기시간, GPU 최대 메모리
- train 라벨 비교: noisy label 기준 진단. clean GT 정확도 아님
- JSON 지표: 비율 `0~1`; 콘솔: `%`. MCC/kappa: `-1~1`
- balanced accuracy: GT가 있는 클래스 recall 평균
- macro 지표: 설정된 전체 클래스 평균; 분모 0은 0 처리
- 데이터 대기시간: CPU의 loader 대기 측정. GPU 유휴시간과 동일하지 않음
- `checkpoint_results.jsonl`: 4개 파일별 선택 epoch·validation·최종 test 지표
- 같은 epoch의 checkpoint: test forward 한 번만 수행
- `label_wave.jsonl`: PC, 이동평균, 선택 epoch, patience
- `stop_training=False`여도 Label Wave checkpoint 저장
- test는 학습 종료 후 저장된 모델 평가에만 사용. checkpoint 선택에 미사용
- 자동 resume는 미구현

## 참고

- SSR: [BMVC 2022 논문](https://bmvc2022.mpi-inf.mpg.de/0372.pdf) / [저자 코드](https://github.com/MrChenFeng/SSR_BMVC2022)
- LSL: [CVPR 2024 논문](https://openaccess.thecvf.com/content/CVPR2024/papers/Kim_Learning_with_Structural_Labels_for_Learning_with_Noisy_Labels_CVPR_2024_paper.pdf), Algorithm 1/2 기준
- Label Wave: [ICLR 2024 논문](https://proceedings.iclr.cc/paper_files/paper/2024/file/5edb57c05c81d04beb716ef1d542fe9e-Paper-Conference.pdf) / [저자 코드](https://github.com/tmllab/2024_ICLR_LabelWave)
