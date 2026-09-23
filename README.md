# SSR-LSL

노이즈 라벨 학습 논문 구현·확장 프로젝트.

- SSR: confidence 기반 라벨 수정, k-NN sample selection, mixup CE, feature consistency
- LSL: reverse k-NN으로 structural target 생성, soft-label loss 추가
- Label Wave: train prediction 변화량으로 checkpoint 선택
- 각 알고리즘은 원 논문·저자 코드 기반. 새로운 알고리즘 제안이 아닌 구현·실험 목적

## 구현 범위

- SSR 학습 루프에 LSL·Label Wave를 독립 옵션으로 통합
- timm backbone 교체, 사용자 이미지 데이터셋 지원
- CUDA BF16 학습·channels-last·fused AdamW 옵션
- selection·k-NN·평가는 FP32 유지
- 라벨 변경 수·클래스별 지표·checkpoint 선택 기록
- 핵심 수식, 학습 경로, checkpoint 저장 회귀 테스트

## 구조

```text
run.py          학습 → checkpoint 평가
audit.py        데이터 검사
setting/        config, NPY loader, 증강, 모델 구성
models/         timm backbone
ssr/            라벨 수정, sample selection, loss, 학습
lsl/            reverse k-NN, structural loss
label_wave/     prediction change, checkpoint 선택
log/            설정·지표·모델 저장
tests/          핵심 구현·실행 검증
```

- 기존 CIFAR·Multi-ROI 실험: 로컬 `experiment_folder/`에 분리, Git 제외
- 공개 코드에서 실험 폴더 import 없음
- 이미지·학습 결과·가중치 미포함

## 실행

Python 3.10 이상. 프로젝트 루트에서 실행.

```bash
uv sync
uv run python audit.py
uv run python run.py
```

먼저 `setting/config.py` 수정:

- `DATASET_ROOT`, `OUTPUT_ROOT`: 데이터·결과 저장 경로
- `DataConfig.class_names`: 라벨 번호순 클래스 이름
- `SplitConfig`: train / validation / test NPY 파일명
- `ModelConfig`: backbone·pretrained 여부
- `TrainingConfig`: batch·LR·epoch·worker
- pretrained 사용 시 첫 실행에서 가중치 다운로드

입력 형식:

- `*_path.npy`: 이미지 경로 문자열 배열 `(N,)`, pickle 미사용
- `*_labels.npy`: 정수 라벨 배열 `(N,)`, 기본 `0 … C-1`
- 상대 이미지 경로: `image_root` 기준. `None`이면 `DATASET_ROOT` 기준
- validation/test가 없으면 해당 설정을 `None`
- train / validation / test 경로 중복 차단
- 원본 RGB → 정사각 bicubic resize → weak/strong 증강 → 정규화
- 데이터 split·라벨·원본 파일 수정 없음. 이미지 디스크 캐시 생성 없음

알고리즘 설정:

| 설정 | 항목 |
| --- | --- |
| SSR | `relabel_threshold`, `selection_threshold`, `neighbors`, `mixup_alpha`, `feature_consistency_weight` |
| LSL | `structural_labels.enabled`, `neighbors`, `loss_weight` |
| Label Wave | `label_wave.enabled`, `moving_average_window`, `patience`, `stop_training` |

- LSL OFF여도 SSR weak/strong feature consistency는 유지
- Label Wave `stop_training=False`: checkpoint 선택·저장만 수행
- 기본 config는 사용자 데이터용 예시. 원 논문의 실험 조건·성능 재현 설정과 구분
- seed `0`. 새 실행은 처음부터 학습하며 자동 resume는 미구현

## 결과 저장

`OUTPUT_ROOT/<설정 이름>/<실행 시각>/`

- `config.json`: 실행 설정
- `metrics.jsonl`: loss·LR·validation 지표·라벨 변경 통계
- `label_wave.jsonl`: prediction change·선택 epoch
- `checkpoint_results.jsonl`: 저장 모델별 최종 test 지표
- `best_balanced_accuracy.pt`, `best_macro_f1.pt`: validation 기준 best
- `last.pt`: 마지막 완료 epoch
- `label_wave.pt`: Label Wave 선택 모델

Test는 종료 후 평가에만 사용. Validation이 없으면 best 저장 생략.
라벨 변경 수는 수정의 정확도를 뜻하지 않으며, noisy 평가 라벨 기준 점수는 실제 정답률과 구분.

## 테스트

```bash
uv run python -m unittest discover -s tests -t .
```

- 임시 이미지·작은 모델 사용. 실제 데이터나 pretrained 다운로드 불필요
- SSR/LSL ON·OFF, Label Wave 선택, 지표 계산, 저장·실행 경로 검증
- CUDA 전용 검증은 CUDA가 없는 환경에서 생략

## 참고

- SSR — [BMVC 2022 논문](https://bmvc2022.mpi-inf.mpg.de/0372.pdf) · [저자 코드](https://github.com/MrChenFeng/SSR_BMVC2022)
- LSL — [Learning with Structural Labels, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Kim_Learning_with_Structural_Labels_for_Learning_with_Noisy_Labels_CVPR_2024_paper.pdf)
- Label Wave — [ICLR 2024 논문](https://proceedings.iclr.cc/paper_files/paper/2024/file/5edb57c05c81d04beb716ef1d542fe9e-Paper-Conference.pdf) · [저자 코드](https://github.com/tmllab/2024_ICLR_LabelWave)
- SSR 원 저작권·MIT 라이선스: `LICENSE` 유지
