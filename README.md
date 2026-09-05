# SSR-LSL

- 루트: 사용자 이미지 데이터셋 학습
- `cifar/`: 기존 CIFAR-10 실험 독립 보관
- config: `setting/config.py`
- seed: `0`
- 기본값: pretrained ConvNeXtV2-Tiny, SSR + LSL, Label Wave 저장
- SSR/LSL 수식·sample selection 유지
- 입력: 전체 이미지. ROI·bbox cache·추가 noise 생성·sqrt resampling 없음

## 구조

```text
run.py
audit.py
setting/       config, NPY loader, 이미지 검사, augmentation, model 구성
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

`setting/config.py`의 `DataConfig`:

```python
root = Path("/root/project/dataset/npy_path/modify_npy")
class_names = ("A1", "A2", "A3", "A4", "A5", "A6", "A7")
train = SplitConfig("train_path.npy", "train_labels.npy")
validation = SplitConfig("val_path.npy", "val_labels.npy")
test = SplitConfig("test_path.npy", "test_labels.npy")
image_size = 224
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

이미지 검사:

- `verify_images=True`: 학습 전 train/val/test 전체 decode 검사
- `allow_truncated_images=True`: 잘린 이미지의 decoder 재시도 허용
- 재시도 성공/실패 경로: `data_audit.jsonl`
- 완전히 읽을 수 없는 파일: 경로를 모아서 보고하고 중단
- 검정 이미지 대체·sample 자동 제거 없음

## 실행

```bash
uv sync
uv run python run.py
```

서버:

```bash
cd /root/project/ssr
nohup /opt/conda/bin/python -u run.py > train.log 2>&1 &
```

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

## 저장

`outputs/<설정 이름>/<실행 시각>/`

- 실행마다 새 폴더. 명시한 `runtime.run_id` 중복은 차단
- `config.json`: 실제 설정과 데이터 경로
- `data_audit.jsonl`: 이미지 검사 결과
- `metrics.jsonl`: loss, LR, validation accuracy, selected, 실제 label 변경 수·전이
- `best.pt`: validation accuracy 최고점. validation이 없으면 생성 안 함
- `last.pt`: 매 epoch 저장. 마지막 test 평가는 이 모델 기준
- `label_wave.pt`: prediction-change 기준 선택 모델
- `label_wave.jsonl`: PC, 이동평균, 선택 epoch, patience
- `stop_training=False`여도 Label Wave checkpoint 저장
- test는 최종 평가에만 사용
- 자동 resume는 미구현

## 참고

- SSR: [BMVC 2022 논문](https://bmvc2022.mpi-inf.mpg.de/0372.pdf) / [저자 코드](https://github.com/MrChenFeng/SSR_BMVC2022)
- LSL: [CVPR 2024 논문](https://openaccess.thecvf.com/content/CVPR2024/papers/Kim_Learning_with_Structural_Labels_for_Learning_with_Noisy_Labels_CVPR_2024_paper.pdf), Algorithm 1/2 기준
- Label Wave: [ICLR 2024 논문](https://proceedings.iclr.cc/paper_files/paper/2024/file/5edb57c05c81d04beb716ef1d542fe9e-Paper-Conference.pdf) / [저자 코드](https://github.com/tmllab/2024_ICLR_LabelWave)
