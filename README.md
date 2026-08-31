# CIFAR-10 SSR baseline

이 저장소는 Learning with Structural Labels 구현의 기반으로 사용할
**CIFAR-10 전용 SSR** 코드입니다. 공식
[SSR_BMVC2022](https://github.com/MrChenFeng/SSR_BMVC2022) 저장소의 commit
`849530dda0cb8fe1a6af5f11ecaf45858cbe8285`을 기준으로 알고리즘을 보존하고,
RLNLC 프로젝트처럼 설정·데이터·모델·학습 엔진을 분리했습니다.

## 보존한 SSR 메커니즘

warm-up은 없습니다. 매 epoch마다 처음부터 아래 순서를 반복합니다.

1. weak augmentation으로 전체 학습 데이터의 feature와 classifier 확률을 계산합니다.
2. `max probability > theta_r`인 샘플만 classifier 예측으로 relabel합니다. 나머지는
   **원래 noisy label**로 되돌립니다.
3. 정규화된 feature에서 cosine k-NN을 구합니다. 자기 자신을 neighbor에서 제외하지
   않으며, similarity 가중치 없이 hard vote를 합니다.
4. vote를 현재 relabelled label의 전역 class prior로 나누고 정규화합니다.
5. 현재 label의 score가 최대 class score와 이루는 비율이 `theta_s` 이상인 샘플만
   선택합니다. 기본 `theta_s=1`이므로 동률을 포함해 hard-vote 최댓값과 일치해야 합니다.
6. 선택 샘플을 class-balanced oversampling하고, 두 strong view에 Beta(4, 4) mixup CE를
   적용합니다.
7. 전체 데이터의 `[weak, strong]` view에 projector/predictor를 적용하고
   `-cosine(p_strong, stopgrad(z_weak))` feature-consistency loss를 더합니다.

공식 코드의 `p_weak`는 loss에 직접 쓰이지 않지만 predictor BatchNorm 상태를 갱신하므로
forward 계산을 그대로 유지합니다. k-NN은 `K=200`, 10개 chunk, 자기 자신 포함,
uniform voting, global-prior correction이라는 원본 동작을 유지합니다.

## 프로젝트 구조

```text
cifar_ssr.py              # 얇은 실행 진입점; argparse 없음
setting/config.py         # immutable dataclass 설정
setting/data.py           # CIFAR-10 + symmetric/asymmetric/IDN noise
setting/augmentation.py   # none/weak/strong view 정의
setting/model.py          # PreActResNet-18와 SSR heads 구성
models/preresnet.py       # 공식 SSR의 CIFAR encoder
ssr/selection.py          # relabelling + structural selection
ssr/knn.py                # hard-vote k-NN + global class balance
ssr/losses.py             # mixup CE + feature consistency
ssr/sampler.py            # class-balanced oversampling
ssr/engine.py             # epoch orchestration
log/common.py             # config, JSONL metric, atomic checkpoint
```

Animal-10N, Clothing1M, WebVision, CIFAR-100, open-set noise loader와 관련 모델·진입점은
제거했습니다. upstream README는 `SSR_UPSTREAM_README.md`, MIT license는 `LICENSE`에
보존되어 있습니다.

## 실행

기본 설정은 이후 Structural Labels 실험을 고려한 CIFAR-10 IDN 50%와
`theta_r=0.55`입니다. 모든 설정은 `setting/config.py`의 `CONFIG` 한 곳에서 바꿉니다.
SSR 논문의 CIFAR-10 symmetric 50% 설정을 재현할 때는 `noise_kind="symmetric"`와
`relabel_threshold=0.8`을 사용합니다.

```bash
uv sync
uv run python cifar_ssr.py
```

noise label은 `data/noise/`에 재사용 가능한 artifact로 저장되고, 실행 설정·epoch metric·
best/last checkpoint는 `outputs/<run_name>/`에 저장됩니다. clean train label은 selection과
relabel 성능을 관찰하는 metric에만 사용하며 학습 target에는 사용하지 않습니다.
