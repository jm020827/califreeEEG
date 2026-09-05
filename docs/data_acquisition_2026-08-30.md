# 공개 SSVEP 데이터 획득 기록 — 2026-08-30

> **Cohort 역할:** wearable 전체 asset은 N=102·24,480행으로 감사했다. DEC-20260830-001에 따라 S1–S3는 development-only이고, DEC-20260830-002에 따라 S4–S102는 39명 training·60명 independent lockbox다.

> **2026-09-06 실행 갱신:** 39명 source는 metadata-calibration v1에 한 번 사용했고
> `development_no_go`였다. Held 60명은 seal/claim/result 없이 미개봉 보존한다.

상태: 4개 P0 데이터셋 전체 raw 다운로드·checksum·전처리 완료. Wearable source
39명 개발 성능은 v1에서 공개했고, held 60명 확증 성능은 열지 않음.

## 확보 결과

| 데이터셋 | 고정 배포본 | 이용조건 | 고정 배포 구성 | 처리 결과 |
|---|---|---|---:|---:|
| Wearable | Figshare `13560281` v4, DOI `10.6084/m9.figshare.13560281.v4` | CC BY 4.0 | 106 files, 973,933,622 bytes | 102명, 24,480 trials, 12 classes |
| BETA | Figshare `12264401` v3, DOI `10.6084/m9.figshare.12264401.v3` | CC BY 4.0 | 72 files, 5,276,556,984 bytes | 70명, 11,200 trials, 40 classes |
| Dong2023 | Zenodo public mirror record `18847318`; NEMAR `nm000128` v1.0.2 | mirror metadata: CC BY-NC 4.0 | 61 files, 700,939,903 bytes | 59명, 9,440 trials, 40 classes |
| Wang2016 | Zenodo MOABB re-upload record `14865172` | mirror metadata: CC BY 4.0; upstream 권리 확인 필요 | 38 files, 3,708,484,359 bytes | 35명, 8,400 trials, 40 classes |

합계는 고정 배포 파일 `277`개·`10,659,914,868` bytes, `266` subject-dataset units와 `53,520` processed trials다. 사람 수를 서로 다른 데이터셋 사이에서 동일인 여부 없이 합산한 값이므로 통계적 `N=266`으로 사용하지 않는다.

로컬 경로:

- raw: `/home/whwovy/eeg-data/raw/{wearable,beta,dong2023,wang}`
- processed: `/home/whwovy/eeg-data/processed/{wearable_v3,beta_v1,dong2023_v1,wang_v1}`

## 무결성과 provenance

`scripts/fetch_dataset.py`는 다음 고정 배포 API 응답 snapshot을 저장하고, 각 파일에 대해 API가 제시한 size와 MD5를 확인한 뒤 `.part`를 최종 이름으로 원자적으로 바꾼다.

- Wearable: `figshare_article.json` SHA-256 `db97d40090e8c0651eb5085591bba5dad7144a65beb6f625927e13574d652bb8`
- BETA: `figshare_article.json` SHA-256 `8d9c948366512183556cd4082f871b72c80e9d2972fa8596795fde56dd5eadc1`
- Dong2023: `zenodo_record.json` SHA-256 `dd4bb5abb2bcd80fbe0f32f814a809bbcb39f2732a28cfbebaab6d56e2569617`
- Wang2016: `zenodo_record.json` SHA-256 `19c0b595f066d3d8be5470ed390321da908cd26cf860bd088d84c61a098b6735`

이 SHA-256은 2026-08-30에 저장한 **API 응답 snapshot 자체**를 식별한다. Zenodo의 동적 조회 통계처럼 재조회 때 달라질 수 있는 필드가 포함되므로 장기 source lock은 이 값 하나가 아니라 record/article ID·명시적 version과 응답 안의 정렬 가능한 파일명·size·MD5 inventory를 함께 사용한다.

동적 필드를 제외하고 Figshare는 `{name,size,md5}`, Zenodo는 `{name,size,checksum}`만 남긴 뒤 `name`으로 정렬하고 `jq -cS`로 canonical JSON화한 stable inventory SHA-256은 다음과 같다.

- Wearable: `0675575c856e3bdcf423131045582602ff8cf35d3a8a1e1bf08d6b7aa9ab3d54`
- BETA: `0086ed67f3413d9df4532e102f2854e3ffee3f410f255c9301233cd3b536a716`
- Dong2023: `75349a4389fbb023f62e4a3777994a6a0ba7fb52a7260ae7ca28e54838db4bff`
- Wang2016: `19d92a6340ce1feee162b19c596c2ae771f2f47c10fedead4c7297701079f12b`

완료 후 `downloaded_paths.txt`는 Wearable/BETA/Dong/Wang 순서로 정확히 `106/72/61/38`행이며 모든 경로가 존재한다. 네 raw directory 어디에도 `.part` 파일이 남지 않았다. Processed verifier는 manifest/HDF5 길이, label 일치, subject·target·sample·원 채널 수와 class map을 검사했고 네 데이터셋 모두 통과했다.

Wearable primary의 full deep audit도 통과했다. 102명×wet/dry×10 blocks×12 targets의 `24,480`행 전체 grid, dry/wet 각 `12,240`행, dry-first/wet-first subject `53/49`, raw EEG→filtered/cropped signal, query-QC, per-channel impedance, headband order, channel alignment를 전수 대조했다. Signal/query-QC/impedance 최대 절대오차는 모두 `0.0`이었다. 감사 receipt는 `wearable_v3_audit_receipt.json`, SHA-256은 `1e2c10d7922c329d8d28cd8bd3cb8ed94e0b22f6a2f3432e95c77deab3f0bc9b`다.

## 획득 중 발견해 수정한 문제

1. Figshare latest endpoint를 그대로 따르면 미래 version drift가 생긴다. BETA는 v3, Wearable은 v4 endpoint로 고정했다.
2. 현재 MOABB Wang adapter는 35명 데이터라고 설명하면서 `range(1, 35)`를 사용해 S35를 누락한다. MOABB fetch를 제거하고 공개 Zenodo MOABB re-upload record `14865172`의 S1–S35와 지원파일을 직접 받도록 바꿨다.
3. 전체 기본 fetch는 정확한 subject ID 집합을 요구한다. BETA `{1..70}`, Wearable `{1..102}`, Dong `{1..59}`, Wang `{1..35}`에서 하나라도 빠지거나 추가되면 다운로드 전에 중단한다.
4. Wang Zenodo mirror에는 원 배포의 `Freq_Phase.mat`이 없다. 현재 frequency/phase는 논문·원 배포표와 대조한 versioned config에서 읽으며, 이 사실을 provenance 제한으로 유지한다.
5. Dong과 Wang Zenodo record는 raw artifact 전용 DOI 대신 각각 논문 DOI `10.26599/BSA.2023.9050020`, `10.1109/TNSRE.2016.2627556`을 재사용하고 `conceptdoi`도 없다. 따라서 Zenodo record ID·파일별 MD5를 우선 raw 식별자로 쓰고, Dong에는 NEMAR version DOI `10.82901/nemar.nm000128.v1.0.2`도 함께 기록한다.

## 접근 요청이 필요한 경우

현재 네 데이터셋은 로그인·토큰·접근신청 없이 비상업 연구 목적으로 내려받을 수 있었다.

- Dong2023은 CC BY-NC 4.0이므로 상업적 이용에는 권리자 허가가 필요하다.
- BETA는 Figshare상 CC BY 4.0이지만 논문의 participant-consent 설명은 non-commercial scientific research를 언급한다. 상업적 활용 전에는 저자·기관 확인이 필요하다.
- Wang2016의 CC BY 4.0 표시는 Zenodo MOABB re-upload metadata의 선언이다. 원 배포 페이지에는 명확한 license가 없으므로 상업적 활용이나 제3자 재배포 전에는 원 저자·권리자 확인이 필요하다.
- Nakanishi 12JFPM은 현재 배포 license가 명확하지 않다. exact-12 외부 확증 데이터로 사용하거나 derivative를 배포하기 전에 저자에게 서면 재사용 허가를 받아야 한다.
- 데이터셋은 아니지만 secondary REVE backbone을 실행하려면 Hugging Face gated access 승인과 `HF_TOKEN`이 필요하다. 외부 pretraining이 없는 compact scratch primary에는 필요하지 않다.

따라서 현재 연구 실행을 위해 사용자에게 필요한 계정·토큰은 없다. 추가 exact-12 외부 복제를 Nakanishi로 진행할 때에는 저자 연락이 필요하다.

## 다음 gate

1. [완료·no-go] S1–S3 physical assay와 후속 metadata-calibration v1 source 39명 gate를
   각각 사전 규칙으로 판정했다.
2. [완료] BETA/Wang/Dong은 paired impedance가 없고 Ke2025는 impedance `n/a`이므로
   M 효과 자료가 아닌 signal/OOD 역할로 제한했다.
3. [필요] Current source 39명과 독립인 randomized wet/dry×repeated-session×block-impedance
   development cohort를 요청·수집한다.
4. [보존] 기존 held 60명은 새 v2가 독립 gate를 통과하기 전까지 미개봉이다.
