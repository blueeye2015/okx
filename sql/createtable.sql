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