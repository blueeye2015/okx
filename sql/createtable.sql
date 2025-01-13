CREATE TABLE IF NOT EXISTS public.trade_orders (
    id bigint ,
    symbol VARCHAR(20) NOT NULL,
    order_type VARCHAR(10) NOT NULL, -- 'BUY' or 'SELL'
    price DOUBLE PRECISION NOT NULL,
    quantity DOUBLE PRECISION NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    strategy_name VARCHAR(50), -- 策略名称
    status VARCHAR(20) NOT NULL, -- 'OPEN', 'CLOSED', '"CANCEL"'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS public.trade_pairs (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    profit_amount DOUBLE PRECISION, -- 盈利金额
    profit_percentage DOUBLE PRECISION, -- 盈利百分比
    hold_duration INTERVAL, -- 持仓时长
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS public.trade_data
(
    id SERIAL PRIMARY KEY,  -- 自增主键
    symbol VARCHAR(255) NOT NULL,  -- 产品ID
    tradeId VARCHAR(255) NOT NULL,  -- 成交ID
    px NUMERIC(20, 8) NOT NULL,  -- 成交价格
    sz NUMERIC(20, 8) NOT NULL,  -- 成交数量
    side VARCHAR(4) NOT NULL CHECK (side IN ('buy', 'sell')),  -- 成交方向
    timestamp timestamp with time zone NOT NULL -- 成交时间
	
);
CREATE TABLE IF NOT EXISTS public.trading_symbols
(
    symbol character varying(20) COLLATE pg_catalog."default" NOT NULL,
    market_value numeric(20,8),
    status character varying(10) COLLATE pg_catalog."default",
    group_type character varying(10) COLLATE pg_catalog."default",
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT trading_symbols_pkey PRIMARY KEY (symbol)
)


CREATE TABLE IF NOT EXISTS public.trend_records_5m_tz
(
    id integer NOT NULL DEFAULT nextval('trend_records_id_seq'::regclass),
    "timestamp" timestamp with time zone,
    symbol character varying(20) COLLATE pg_catalog."default",
    ma60 double precision,
    ma60_slope double precision,
    ma60_r2 double precision,
    close double precision,
    consecutive_count integer,
    is_above_20_periods boolean,
    created_at timestamp with time zone,
    CONSTRAINT trend_records_5m_tz_pkey PRIMARY KEY (id),
    CONSTRAINT trend_records_5m_tz_symbol_date_unique UNIQUE (symbol, "timestamp")
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.trend_records_5m_tz
    OWNER to postgres;
-- Index: idx_trendrecords_5m_symbol_timestamp

-- DROP INDEX IF EXISTS public.idx_trendrecords_5m_symbol_timestamp;

CREATE INDEX IF NOT EXISTS idx_trend_records_5m_tz_symbol_timestamp
    ON public.trend_records_5m_tz USING btree
    (symbol COLLATE pg_catalog."default" ASC NULLS LAST, "timestamp" ASC NULLS LAST)
    WITH (deduplicate_items=True)
    TABLESPACE pg_default;


    -- Table: public.trades

-- DROP TABLE IF EXISTS public.trades;

CREATE TABLE IF NOT EXISTS public.trades_tz
(
    event_type character varying(50) COLLATE pg_catalog."default",
    event_time timestamp with time zone,
    symbol character varying(20) COLLATE pg_catalog."default" NOT NULL,
    trade_id bigint NOT NULL,
    price numeric,
    quantity numeric,
    buyer_order_maker boolean,
    trade_time timestamp with time zone,
    CONSTRAINT trades_tz_pkey PRIMARY KEY (symbol, trade_id),
    CONSTRAINT uk_symbol_tradeid UNIQUE (symbol, trade_id)
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.trades_tz
    OWNER to postgres;
-- Index: idx_symbol_tradetime

-- DROP INDEX IF EXISTS public.idx_symbol_tradetime;

CREATE INDEX IF NOT EXISTS idx_trades_tz_symbol_tradetime
    ON public.trades_tz USING btree
    (symbol COLLATE pg_catalog."default" ASC NULLS LAST, trade_time ASC NULLS LAST)
    WITH (deduplicate_items=True)
    TABLESPACE pg_default;


CREATE TABLE IF NOT EXISTS public.klines_5m
(
    symbol character varying(20) COLLATE pg_catalog."default" NOT NULL,
    "timestamp" timestamp with time zone NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume double precision,
    CONSTRAINT klines_5m_pkey PRIMARY KEY (symbol, "timestamp")
)


#虚拟交易记录表
CREATE TABLE virtual_trade_records (
            id SERIAL  PRIMARY KEY ,
            symbol VARCHAR(50),
            entry_price FLOAT,
            entry_time timestamp with time zone,
            exit_price FLOAT,
            exit_time timestamp with time zone,
            position_size FLOAT,
            leverage INTEGER,
            pnl FLOAT,
            roi FLOAT,
            status VARCHAR(50),
            close_reason VARCHAR(50),
            success INTEGER
        )
		
#虚拟活跃订单表		
CREATE TABLE  virtual_active_positions (
            id SERIAL PRIMARY KEY ,
            symbol VARCHAR(50),
            entry_price FLOAT,
            entry_time timestamp with time zone,
            position_size FLOAT,
            leverage INTEGER,
            last_price FLOAT,
            last_update_time timestamp with time zone,
            unrealized_pnl FLOAT,
            unrealized_roi FLOAT
        )
# 策略性能记录表
CREATE TABLE  strategy_performance (
			strategy_name varchar(50),
            date DATE ,
            balance FLOAT,
            daily_pnl FLOAT,
            total_positions INTEGER,
            winning_positions INTEGER,
            win_rate FLOAT,
			CONSTRAINT date_name_pkey PRIMARY KEY (strategy_name, date)
        )