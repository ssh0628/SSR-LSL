# SSR-LSL

CIFAR-10 noisy-label 학습 코드.

- SSR baseline
- Learning with Structural Labels
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
- Backbone: `model.name`
- Noise: `data.noise_kind`, `data.noise_rate`
- Seed: `seed`
- GPU별 권장값: config 상단 주석 참고

논문 기본값:

- epoch: `300`
- batch size: `128`
- learning rate: `0.02`
- `k_st`: `20`
- `lambda_fc`: `1.0`
- `lambda_st`: `1.0`
- mixup alpha: `4.0`

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

## 폴더

```text
setting/    config, data, augmentation
models/     backbone
ssr/        relabel, selection, training
lsl/        structural label, structural loss
log/        metric, checkpoint
```

## 참고

- clean train label: metric 계산에만 사용
- 학습 supervision: noisy/relabelled/structural label만 사용
