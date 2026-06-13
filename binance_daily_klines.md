# 币安日K线获取程序 (Binance Daily Klines Fetcher)

## 概述

自动获取币安(Binance) **全部 USDT 现货交易对** 和 **全部 USDT 永续合约** 的日K线数据，增量存入 PostgreSQL，支持每日定时运行。

---

## 运行环境

```
/data/anaconda3/envs/okx_api/bin/python3
```

**依赖包** (已预装在 conda 环境):
- `psycopg2` — PostgreSQL 连接
- `requests` — HTTP 请求
- `python-dotenv` — 环境变量读取

---

## 功能特性

| 特性 | 说明 |
|------|------|
| 现货市场 | 自动获取币安全部 `USDT` 现货交易对 (约 430 个) |
| 合约市场 | 自动获取币安全部 `USDT` 永续合约 (约 529 个) |
| 增量更新 | 自动判断数据库最新日期，只获取缺失的数据 |
| 防重复 | `ON CONFLICT DO UPDATE`，重复运行不重复插入 |
| 完整字段 | 开盘/最高/最低/收盘/成交量/成交额/成交笔数/吃单量 |
| 代理支持 | 自动读取 `.env` 中的 `HTTP_PROXY` / `HTTPS_PROXY` |
| 限速保护 | 请求间隔 0.15 秒，触发限流自动等待 60 秒 |

---

## 数据库表结构

**表名**: `binance_daily_klines`

| 字段 | 类型 | 说明 |
|------|------|------|
| `symbol` | VARCHAR(30) | 交易对，如 `BTCUSDT` |
| `market_type` | VARCHAR(10) | `spot`(现货) / `swap`(合约) |
| `open_time` | TIMESTAMP | K线开盘时间 |
| `open` | NUMERIC(30,12) | 开盘价 |
| `high` | NUMERIC(30,12) | 最高价 |
| `low` | NUMERIC(30,12) | 最低价 |
| `close` | NUMERIC(30,12) | 收盘价 |
| `volume` | NUMERIC(30,12) | 成交量 |
| `quote_volume` | NUMERIC(30,12) | 成交额 |
| `trades_count` | INTEGER | 成交笔数 |
| `taker_buy_base` | NUMERIC(30,12) | 吃单买入量 |
| `taker_buy_quote` | NUMERIC(30,12) | 吃单买入额 |
| `close_time` | TIMESTAMP | K线收盘时间 |
| `created_at` | TIMESTAMP | 记录创建时间 |
| `updated_at` | TIMESTAMP | 记录更新时间 |

**唯一约束**: `(symbol, market_type, open_time)`

---

## 快速开始

### 1. 初始化数据库表

```bash
cd /data/okx
/data/anaconda3/envs/okx_api/bin/python3 binance_daily_klines_pg.py --init-only
```

### 2. 全量获取（首次运行）

```bash
# 获取现货 + 合约全部交易对
/data/anaconda3/envs/okx_api/bin/python3 binance_daily_klines_pg.py --market all

# 只获取现货
/data/anaconda3/envs/okx_api/bin/python3 binance_daily_klines_pg.py --market spot

# 只获取合约
/data/anaconda3/envs/okx_api/bin/python3 binance_daily_klines_pg.py --market swap
```

> 首次全量运行约需 **20~30 分钟**（430 现货 + 529 合约），写入约 **95 万条**记录。

### 3. 增量更新（后续每日运行）

```bash
# 相同命令，程序会自动识别已有数据，只拉取新增日K线
/data/anaconda3/envs/okx_api/bin/python3 binance_daily_klines_pg.py --market all
```

增量更新通常在 **1 分钟内**完成。

---

## 命令行参数

```
--market {spot,swap,all}   要获取的市场类型 (默认: all)
--symbols [SYM1 SYM2 ...]  指定交易对列表，如 BTCUSDT ETHUSDT
--init-only                仅初始化数据库表，不获取数据
--proxy URL                临时指定 HTTP 代理，如 http://127.0.0.1:7890
--daemon                   守护进程模式，每日自动执行
--run-at HH:MM             守护进程每日执行时间 (默认: 00:05)
```

---

## 定时任务配置

### 方式一：crontab（推荐）

```bash
crontab -e
```

添加以下内容：

```cron
# 币安日K线每日增量更新 (00:05 执行)
5 0 * * * cd /data/okx && /data/anaconda3/envs/okx_api/bin/python3 binance_daily_klines_pg.py --market all >> /data/okx/cron.log 2>&1
```

### 方式二：systemd 服务

```bash
# 复制服务文件
sudo cp /data/okx/binance-daily-klines.service /etc/systemd/system/

# 启动并设为开机自启
sudo systemctl daemon-reload
sudo systemctl enable binance-daily-klines
sudo systemctl start binance-daily-klines

# 查看状态
sudo systemctl status binance-daily-klines
```

---

## 数据来源

| 市场 | API 地址 | 接口 |
|------|---------|------|
| 现货 | `https://api.binance.com` | `/api/v3/klines` |
| 合约 | `https://fapi.binance.com` | `/fapi/v1/klines` |

日K线参数: `interval=1d`, `limit=1000`（单次最大 1000 条，约 3 年数据）

---

## 配套扫描器

### 超跌反弹扫描器 (`oversold_reversal_scanner.py`)

基于日K线数据，筛选深度超跌后均线突破的币种。

```bash
# 扫描现货，回撤85%以上，弱势250天
/data/anaconda3/envs/okx_api/bin/python3 oversold_reversal_scanner.py --market spot --drawdown 0.85 --weak-days 250

# 扫描合约，默认参数
/data/anaconda3/envs/okx_api/bin/python3 oversold_reversal_scanner.py --market swap
```

**筛选逻辑**:
1. 从历史最高价回撤 ≥ `--drawdown` (默认 80%)
2. 过去 `--weak-days` 天中，≥80% 天数收盘价低于 **60周均线** (≈300日均线)
3. 当日收盘价同时超过 **日线MA6** 和 **日线MA24**

---

## 日志文件

- **运行日志**: `binance_daily_klines.log`
- **crontab 日志**: `cron.log`
- **全量同步日志**: `full_sync.log`

---

## 注意事项

1. **运行环境**: 必须使用 `/data/anaconda3/envs/okx_api/bin/python3`，不要直接使用系统 Python。
2. **网络代理**: 程序自动读取 `.env` 中的 `HTTP_PROXY` / `HTTPS_PROXY`。如需临时指定，用 `--proxy` 参数。
3. **数据库连接**: 从 `.env` 中读取 `DB_DSN`（默认 `postgresql://postgres:12@127.0.0.1:5432/market_data`）。
4. **首次运行**: 建议直接执行全量获取，不要同时开多个实例，避免触发币安 API 频率限制。
5. **数据时区**: 币安 K 线时间戳为 UTC+8（北京时间），`open_time` 存储为 UTC 时间。
