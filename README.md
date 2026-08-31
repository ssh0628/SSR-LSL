# SSR-LSL

CIFAR-10 noisy-label 학습 코드.

- SSR baseline
- Learning with Structural Labels
- Label Wave checkpoint selection
- warm-up 없음
- seed `0`
- CIFAR-10 전용

## 지원 설정

- Noise
  - symmetric
  - asymmetric
  - IDN
- Backbone
  - `preact_resnet18`
  - `cifar_resnet18`
  - `cifar_resnet34`
- LSL
  - `True`: SSR + LSL
  - `False`: SSR only
- Label Wave
  - `enabled=True`: prediction change 기록 + `label_wave.pt` 저장
  - `stop_training=False`: 끝까지 학습, 선택 지점만 관찰
  - `stop_training=True`: patience 충족 시 실제 조기 종료

## 학습 순서

1. 전체 feature 추출
2. 전체 prediction 추출
3. confidence 기반 relabel
4. SSR k-NN sample selection
5. reverse k-NN structural label 생성
6. loss 계산 및 update

## Loss

```text
L = L_ce + lambda_fc * L_fc + lambda_st * L_st
```

- `L_ce`: selected sample mixup CE
- `L_fc`: weak/strong feature consistency
- `L_st`: structural-label mixup CE
- LSL off: `L_st` 제외

## Config

파일: `setting/config.py`

- 프로젝트 경로: `PROJECT_ROOT`
- LSL on/off: `structural_labels.enabled`
- Label Wave on/off: `label_wave.enabled`
- 실제 early stopping: `label_wave.stop_training`
- Backbone: `model.name`
- Noise: `data.noise_kind`, `data.noise_rate`
- Seed: `seed`

논문 기본값:

- epoch: `300`
- batch size: `128`
- learning rate: `0.02`
- `k_st`: `20`
- `lambda_fc`: `1.0`
- `lambda_st`: `1.0`
- mixup alpha: `4.0`

Label Wave 시작값:

- moving-average window: `3`
- patience: `10`
- 첫 실험: `enabled=True`, `stop_training=False`
- 확인 후: `stop_training=True`

주의:

- window `3`: 논문 Appendix E에서 가장 강한 PC/test accuracy 상관
- patience: 논문에 고정 기본값이 없어 config에서 실험값으로 관리
- PC 입력: corrected label이 아닌 전체 train sample의 raw model prediction
- `label_wave.jsonl`의 epoch: 완료된 학습 epoch 수 (`0`은 초기 모델 기준선)
- `label_wave.pt`: clean validation/test GT를 선택 기준으로 사용하지 않음
- low/no-noise 또는 매우 강한 regularization: 명확한 turning point가 없을 수 있음

## GPU 설정

논문 재현:

- batch size: `128`
- learning rate: `0.02`
- workers: `4`
- SSR k-NN chunks: `10`
- LSL k-NN chunks: `10`

RTX 5080 16GB:

- batch size: `256`
- learning rate: `0.04`
- workers: `8`
- prefetch factor: `4`
- SSR k-NN chunks: `8`
- LSL k-NN chunks: `8`

H100 NVL 94GB:

- batch size: `512`
- learning rate: `0.08`
- workers: `16`
- prefetch factor: `4`
- SSR k-NN chunks: `2`
- LSL k-NN chunks: `2`

주의:

- GPU 설정: 처리량 기준 시작값
- 논문 비교: 논문 재현값 사용
- worker 수: CPU core와 storage에 맞춰 조절
- H100 NVL 2장: 현재 코드는 자동 병렬화하지 않음

## 실행

```bash
uv sync
uv run python cifar_ssr.py
```

## 출력

경로: `outputs/<run_name>/`

- `config.json`
- `metrics.jsonl`
- `best.pt`
- `last.pt`
- `label_wave.jsonl` (`label_wave.enabled=True`)
- `label_wave.pt` (`label_wave.enabled=True`, Label Wave 선택 checkpoint)

Checkpoint 기준:

- `best.pt`: test accuracy 최고점. 연구용 oracle 비교
- `last.pt`: 마지막 실제 학습 epoch
- `label_wave.pt`: prediction-change 이동평균 기준

## 폴더

```text
setting/    config, data, augmentation
models/     backbone
ssr/        relabel, selection, training
lsl/        structural label, structural loss
label_wave/ validation-free checkpoint selection
log/        metric, checkpoint
```

## 참고

- clean train label: metric 계산에만 사용
- 학습 supervision: noisy/relabelled/structural label만 사용
- SSR 논문: <https://bmvc2022.mpi-inf.mpg.de/0372.pdf>
- SSR 저자 코드: <https://github.com/MrChenFeng/SSR_BMVC2022>
- LSL 논문: <https://openaccess.thecvf.com/content/CVPR2024/papers/Kim_Learning_with_Structural_Labels_for_Learning_with_Noisy_Labels_CVPR_2024_paper.pdf>
- LSL 저자 코드: 공식 공개 저장소 확인되지 않음. 논문 Algorithm 1/2 기준 구현
- Label Wave 논문: <https://proceedings.iclr.cc/paper_files/paper/2024/file/5edb57c05c81d04beb716ef1d542fe9e-Paper-Conference.pdf>
- Label Wave 저자 코드: <https://github.com/tmllab/2024_ICLR_LabelWave>
