# Closet AI (가칭)

갤러리 셀카와 OOTD로 옷장이 자동으로 채워지고, 날씨·일정에 맞춘 코디 추천과 연예인·화보 룩 "따라 입기"를 제공하는 스타일 앱.

- 왜·원칙: [INTENT.md](INTENT.md)
- 기획·스펙·일정: [PLAN.md](PLAN.md)

## 주요 기능
- 스타일 심리테스트와 공유용 결과 카드
- 자동 옷 등록(갤러리 스캔, 옷장 한 컷, OOTD)과 AI 분류·태깅
- 오늘의 코디 (날씨·일정 기반)
- 따라 입기 (참고 사진 → 내 옷으로 재현 + 부족 아이템 1개)
- 친구 투표 링크
- 구독 (7일 무료 체험, 갤러리 스캔 뒤 페이월)

## 기술 스택
- 앱: React Native + Expo
- 백엔드: Supabase (Auth, Postgres, Storage, Edge Functions)
- AI: 비전 LLM API (태깅·추천·매칭), 이미지 생성 API (v0.5)
- 결제: RevenueCat
- 외부: 날씨 API

## 폴더 구조
```
app/          모바일 앱 화면·컴포넌트
server/       Edge Functions
server/ai/    AI 호출 모음 (모델 교체 지점)
web-vote/     앱 없이 접속하는 투표 페이지
scanner/      갤러리 스캐너 엔진 (Python, M0 검증용 기준 구현)
m0/           M0 정답 라벨 페이지와 검증 절차
docs/         DECISIONS.md, SCANNER.md 등
INTENT.md
PLAN.md
```

## 갤러리 스캐너 (지금 동작하는 부분)

사진 폴더를 넣으면 옷을 찾아 상의·하의·신발로 나누고, 같은 옷끼리 묶어 옷장(closet.json)과 확인용 보고서(report.html)를 만든다. Mac(Apple Silicon)과 Linux에서 동작하고, 모델은 처음 실행할 때 `scanner/.models/`에 내려받는다.

```
cd scanner
uv sync
uv run closet-scan scan ~/Pictures/m0 --out ../out
open ../out/report.html
```

- 같은 옷인지 애매한 쌍은 보고서에서 "같은 옷 / 다른 옷"으로 답한다. 내려받은 `decisions.json`을 `out/`에 넣고 `uv run closet-scan organize ../out`을 다시 실행하면 반영된다.
- Claude 태깅(분류·소재·이름)과 애매한 쌍 자동 검증: `uv sync --extra claude` 후 `ANTHROPIC_API_KEY`를 설정하고 `--tagger claude --verify`.
- 옷 한 벌 단위 검출(실험): `--detector vlm`. 원피스를 한 벌로, 아우터와 안쪽 상의를 따로 잡는다. API 키 필요. bbox 평가는 `closet-scan eval-bbox`.
- M0 정답 비교: [m0/README.md](m0/README.md)
- 설계와 한계: [docs/SCANNER.md](docs/SCANNER.md)
- 사용자가 할 일(M0 절차 요약): [intend.md](intend.md)

## 앱 시작하기 (M1부터)
1. 저장소 클론
2. `.env` 설정
   ```
   SUPABASE_URL=
   SUPABASE_ANON_KEY=
   AI_API_KEY=
   WEATHER_API_KEY=
   REVENUECAT_API_KEY=
   ```
3. `npm install`
4. `npx expo start` 후 기기 또는 에뮬레이터에서 실행
5. Edge Functions: `supabase functions serve`

## 로드맵
- M0 인식 정확도 검증
- M1 옷 등록·태깅
- M2 오늘의 코디
- M3 스타일 테스트·페이월
- M4 따라 입기·투표
- M5 베타
- v0.5 잡지 표지 합성, 액세서리 인식

## 라이선스
미정
