---
description: requirements.yaml（Loop B用要件定義）を作成・修正する
---

# 要件定義ファイル作成・修正（/plan_requirements）

requirements.yaml を作成または修正します。

## requirements.yaml とは

Loop B で自動検証するための **要件** と **設計判断** を定義するファイル。
人がレビューし、実装後に LLM が検証する。

## 要件（acceptance_criteria）と設計判断（design_decisions）の違い

| 項目 | acceptance_criteria | design_decisions |
|------|---------------------|------------------|
| 観点 | ユーザー視点（何が欲しいか） | 技術視点（どう実現するか） |
| 例 | 「カテゴリ別に売上を集計できる」 | 「集計処理は Pandas を使用する」 |
| レビュー | ビジネス要件として妥当か | 技術選択として妥当か |

両方とも実装後に LLM が検証する。`design_decisions` はオプション（技術判断が不要な場合は省略可）。

## acceptance_criteria の構造

各 acceptance_criteria は以下の2要素で構成される：

| フィールド | 説明 | 必須 |
|------------|------|------|
| `criterion` | ユーザー視点の検証可能な基準 | はい |
| `verify` | 検証方法（方法 + 対象 + 期待状態） | はい |

### verify の記述パターン

verify は以下の5パターンのいずれかで記述する（pytest で検証可能な振る舞いに限定）:

| パターン | 目的 | チェック対象 |
|----------|------|--------------|
| `[metrics_check]` | 最終/中間指標の確認 | レポートファイル or オブジェクトのプロパティ・値 |
| `[cli_test]` | コマンドUIテスト | コマンド実行の成功、引数による出力差分 |
| `[flow_order]` | データフロー順序の確認 | spy/mock による呼び出し順序 |
| `[intermediate]` | 中間生成物の検証 | 中間ファイルのフォーマット・プロパティ |
| `[regression]` | リグレッションテスト | 数値メトリクスの許容誤差内一致 |

### verify の書き方例

| criterion | verify |
|-----------|--------|
| validation期間のメトリクス悲観的推定値が計算される | `[metrics_check]` レポートにSharpe/OptMetricの5%/20%/50%パーセンタイル値が出力されている |
| CLIオプションで足切り基準を指定できる | `[cli_test]` --cutoff-pct オプションで実行成功。異なる値で選択結果が変わる |
| bootstrap計算後に選択処理が実行される | `[flow_order]` compute_bootstrap → select_method の順で呼び出される |
| ACFプロファイルが中間出力される | `[intermediate]` 中間ファイルにlag 1-20のACF値が出力されている |
| 同じseedで同一結果が得られる | `[regression]` seed=42で2回実行し、全メトリクス値が一致する |

**注意**:
- `[regression]` は実装完了後に追加するもの。要件定義時点では他の4パターンを使用
- 技術的な具体値（引数名、エラーメッセージ文言等）は書かない。実装前に書ける自然な言葉で記述する

## 可視化資料（要件判断の材料）

テキストで記述するより、図表やサンプル出力を見せた方が acceptance_criteria / design_decisions を明確に判断できるケースがある。
**詳細設計ではなく、ユーザー要件（acceptance_criteria）と技術判断（design_decisions）の判断材料**として使う点に注意。


### 可視化が有効なケース

| ケース | 可視化方法 | 判断できること |
|--------|------------|----------------|
| 出力レポートのフォーマット | サンプル出力（実データ or モック） | 「この形式で出力される」が妥当か |
| アーキテクチャ構成 | コンポーネント図（Mermaid等） | モジュール分割・責務配置が妥当か |
| データフロー/パイプライン | フロー図・パイプライン図 | 処理ステップ・入出力の流れが妥当か |
| UI/画面設計 | ワイヤーフレーム・モックアップ | ユーザー操作フローが妥当か |

### 作成場所と参照方法

`doc/design/{feature-name}-visual.md` に作成し、`notes` から参照する:

```yaml
- id: req_report
  name: 売上レポート出力
  notes: |
    出力サンプル: doc/design/sales-report-visual.md
  acceptance_criteria:
    - "カテゴリ別に売上を集計できる"
    - "CSV形式でエクスポートできる"
  design_decisions:
    - "集計処理は Pandas を使用する"
```

### 可視化資料の原則

- **目的**: acceptance_criteria / design_decisions のレビュー判断を助ける
- **範囲**: 詳細設計ではない。実装詳細は含めない
- **粒度**: 要件の妥当性を判断できる最小限の情報

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
      - criterion: "{ユーザー観点の検証可能な基準1}"
        verify: "[metrics_check] {対象}に{期待するプロパティ・値}が存在する"
      - criterion: "{ユーザー観点の検証可能な基準2}"
        verify: "[cli_test] {コマンド}で実行成功。{条件}で出力が変わる"
    design_decisions:  # オプション
      - "{技術選択の検証可能な基準}"

  - id: req_2
    name: {要件名}
    acceptance_criteria:
      - criterion: "{受け入れ基準}"
        verify: "[intermediate] {中間生成物}に{期待するフォーマット・プロパティ}がある"
    # design_decisions は省略可
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

1. **検証可能性**: 受け入れ基準・設計判断は LLM が検証可能であること
   - acceptance_criteria 良い例: 「カテゴリ別に売上を集計できる」
   - acceptance_criteria 悪い例: 「使いやすいUIを提供する」
   - design_decisions 良い例: 「集計処理は Pandas を使用する」

2. **観点の分離**: ユーザー要件（acceptance_criteria）と技術判断（design_decisions）を分ける

3. **独立性**: 各要件は可能な限り独立して検証可能であること

4. **具体性**: 曖昧さを排除し、実装者が迷わない記述

5. **完全性**: 成功条件を満たすために必要な要件が網羅されていること

6. **notes の活用**: 背景情報、技術的制約、参考リンクなどを notes に記載

---

## スコープ

このコマンドの責務は **requirements.yaml ファイルの作成・更新のみ**。

後続の実行（`task-orchestrator run` や `/plan_tasks`）はユーザーが判断する。
