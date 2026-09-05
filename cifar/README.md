# CIFAR-10 실험

- 기존 SSR / LSL / Label Wave 비교 실험 보관
- 코드와 설정: `cifar/` 내부에서 독립 실행
- 데이터: train 50,000장 / test 10,000장
- Noise: symmetric / asymmetric / IDN
- Backbone: PreActResNet18 / CIFAR-ResNet18 / CIFAR-ResNet34
- seed `0`, 별도 warm-up 없음
- CIFAR-ResNet18/34: 최초 BN 통계 보정 유지

## 실행

프로젝트 루트에서:

```bash
uv run python -m cifar.run          # 실험표 순차 실행
uv run python -m cifar.cifar_ssr    # config 단일 실행
```

- 직접 실행도 지원: `python cifar/run.py`, `python cifar/cifar_ssr.py`
- 단일 설정: `cifar/setting/config.py`
- 실험 목록: `cifar/run.py`의 `RUN_IDS_TO_RUN` (기존 기본값: Run 4)
- 데이터 / 출력: `cifar/data/`, `cifar/outputs/`
- 경로 변경: config의 `PROJECT_ROOT`
- 기존 root 출력은 이동하지 않음. 설정 분리로 run hash도 새로 생성

## 비교 실험

공통: CIFAR-ResNet34, IDN 50%, 300 epochs, batch 128, SGD LR 0.02.

| Run | 알고리즘 | τr | Label Wave |
| --- | --- | --- | --- |
| 1 | SSR | 0.90 | OFF |
| 2 | SSR | 0.55 | OFF |
| 3 | SSR + LSL | 0.55 | OFF |
| 4 | SSR + LSL | 0.55 | 저장 ON / 조기 종료 OFF |

- SSR: K 200, τs 1.0, mixup α 4.0, feature-consistency weight 1.0
- LSL: reverse K 20, loss weight 1.0
- Label Wave: window 3, patience 20
- Cosine minimum LR ratio: 1/50
- 완료 실험: `last.pt` 존재 시 건너뜀
- 중단 실험: 기본값에서 자동 덮어쓰기 금지

## 저장

- `config.json`: 실행 설정
- `metrics.jsonl`: accuracy / loss / LR / selection / 실제 label 변경 수
- `best.pt`: test accuracy 최고점 (CIFAR 비교용 oracle)
- `last.pt`: 마지막 학습 epoch
- `label_wave.pt`: prediction-change 이동평균으로 선택한 checkpoint
- `label_wave.jsonl`: prediction change / 선택 epoch / patience
- clean train GT: 분석 metric에만 사용

## 참고

- SSR: [BMVC 2022 논문](https://bmvc2022.mpi-inf.mpg.de/0372.pdf) / [저자 코드](https://github.com/MrChenFeng/SSR_BMVC2022)
- LSL: [CVPR 2024 논문](https://openaccess.thecvf.com/content/CVPR2024/papers/Kim_Learning_with_Structural_Labels_for_Learning_with_Noisy_Labels_CVPR_2024_paper.pdf), Algorithm 1/2 기준 구현
- Label Wave: [ICLR 2024 논문](https://proceedings.iclr.cc/paper_files/paper/2024/file/5edb57c05c81d04beb716ef1d542fe9e-Paper-Conference.pdf) / [저자 코드](https://github.com/tmllab/2024_ICLR_LabelWave)
