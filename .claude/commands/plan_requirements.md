---
description: requirements.yaml（Loop B用要件定義）を作成・修正する
---

# 要件定義ファイル作成・修正（/plan_requirements）

requirements.yaml を作成または修正します。

## requirements.yaml とは

Loop B で自動検証するための **受け入れ基準（Acceptance Criteria）** を定義するファイル。
一般的な要件定義ドキュメントとは異なり、LLM が検証可能な条件に特化している。

## 詳細設計ドキュメントが必要なケース

以下のケースでは requirements.yaml に加えて `doc/design/{feature-name}-design.md` を作成する:

- **アーキテクチャ判断**: 技術選択の根拠（WebSocket vs SSE 等）
- **複雑なデータ構造**: 状態遷移図、ER図、スキーマ定義
- **外部システム連携**: シーケンス図、API仕様、認証フロー
- **UI/UX設計**: ワイヤーフレーム、画面遷移、コンポーネント構成

詳細ドキュメントを作成した場合、requirements.yaml の `notes` や `acceptance_criteria` から参照する:

```yaml
- id: req_payment
  name: 決済処理
  notes: |
    詳細設計: doc/design/payment-design.md を参照
  acceptance_criteria:
    - "doc/design/payment-design.md のシーケンス図通りに動作する"
```

## モード判定

- **引数なし**: 新規作成モード
- **引数あり**: 既存ファイルの修正モード

---

## 新規作成モード

### 手順

1. **目的・背景のヒアリング**
   - 何を実現したいか
   - 現状の課題は何か
   - 成功条件は何か

2. **コードベース調査**（必要に応じて）
   - 関連する既存コード
   - 制約となる技術的要素
   - 変更が必要な箇所の特定

3. **要件の構造化**
   - 要件を独立した単位に分解
   - 各要件に明確な受け入れ基準を定義
   - 依存関係・優先度の整理

4. **出力**
   - `doc/design/{feature-name}-requirements.yaml` に保存

### 出力フォーマット

```yaml
# 出力先: doc/design/{feature-name}-requirements.yaml

requirements:
  - id: req_1
    name: {要件名}
    notes: |
      {補足情報・背景・制約など}
    acceptance_criteria:
      - "{検証可能な受け入れ基準1}"
      - "{検証可能な受け入れ基準2}"

  - id: req_2
    name: {要件名}
    acceptance_criteria:
      - "{受け入れ基準}"
```

---

## 修正モード

### 手順

1. **既存ファイルの読み込み**

2. **問題点の分析**
   - 曖昧な記述の特定
   - 矛盾・重複の検出
   - 漏れている要件の発見
   - 受け入れ基準の検証可能性チェック

3. **改善提案の提示**
   - 問題点と改善案を明示
   - ユーザーに確認

4. **更新**
   - 承認後、ファイルを更新

---

## 要件定義の原則

1. **検証可能性**: 受け入れ基準は機械的または明確に検証可能であること
   - 良い例: 「`aggregate_by_category(data)` が `dict[str, float]` を返す」
   - 悪い例: 「使いやすいUIを提供する」

2. **独立性**: 各要件は可能な限り独立して検証可能であること

3. **具体性**: 曖昧さを排除し、実装者が迷わない記述

4. **完全性**: 成功条件を満たすために必要な要件が網羅されていること

5. **notes の活用**: 背景情報、技術的制約、参考リンクなどを notes に記載

---

## スコープ

このコマンドの責務は **requirements.yaml ファイルの作成・更新のみ**。

後続の実行（`task-orchestrator run` や `/plan_tasks`）はユーザーが判断する。
