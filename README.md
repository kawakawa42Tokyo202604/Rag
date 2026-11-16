# RAG Against the Machine

## English

A tool that answers questions about source code by finding relevant passages first. It splits Python and Markdown files into chunks, searches with BM25, and passes retrieved text to a local Qwen model.

### Layout

| Path | Contents |
| --- | --- |
| `src/mandatory/` | Chunking, indexing, search, generation, and evaluation |
| `src/bonus/` | Semantic/hybrid search, incremental indexes, and HTTP API |
| `tests/` | Ranges, ranking, JSON models, batches, and HTTP behavior |
| `data/` | Local corpus, datasets, indexes, and output |

### Run and test

Requires Python 3.10+, uv, and Make. Run from the repository root.

```sh
make test
make install
make run FILE_PATH=path/to/source
make search QUERY="Where is the request handler?"
make answer QUERY="What does the request handler do?"
make test-model
```

`make test` prepares a small cached environment and runs 123 tests without model weights or API keys. It uses the real retrieval pipeline and test doubles at the model boundary. `make test-model` selects two additional checks using runtime dependencies and real model weights; it may download those weights. Keep corpus files and generated data locally.

See [the detailed guide](docs/GUIDE.md) for the original setup and implementation notes.

## 日本語

ソースコードから関連箇所を探し、その本文を根拠に質問へ回答するツールです。Python・Markdownを小さな単位に分け、BM25で検索した結果をローカルのQwenモデルへ渡します。

### 構成

| パス | 内容 |
| --- | --- |
| `src/mandatory/` | 分割・索引作成・検索・回答生成・評価 |
| `src/bonus/` | 意味検索・混合検索・差分索引・HTTP API |
| `tests/` | 参照範囲・順位・JSON・一括処理・HTTPの検証 |
| `data/` | ローカルのコード・質問データ・索引・出力 |

### 実行とテスト

Python 3.10以降・uv・Makeが必要です。 リポジトリ直下で実行します。

```sh
make test
make install
make run FILE_PATH=path/to/source
make search QUERY="Where is the request handler?"
make answer QUERY="What does the request handler do?"
make test-model
```

`make test` は小さな環境をキャッシュ上に用意し、モデルやAPIキーなしで123テストを実行します。検索処理は実装をそのまま使い、モデルとの境界にテスト用の代替実装を渡します。実モデルを使う追加2件は `make test-model` で実行し、実行時の依存関係とモデルを取得します。コードデータと生成物はローカルで管理します。

元の構築手順と設計の説明は[詳細ガイド](docs/GUIDE.md)にまとめています。
