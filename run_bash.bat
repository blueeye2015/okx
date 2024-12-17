@echo off
REM 设置 PostgreSQL 的连接信息
SET PGPASSWORD=12
psql -U postgres -d market_data -c "INSERT INTO public.trend_records_5m (timestamp, symbol, ma60, ma60_slope, ma60_r2, close, consecutive_count, is_above_20_periods, created_at) SELECT *, NOW() FROM get_moving_average_slopes_1() ON CONFLICT (symbol, timestamp) DO UPDATE SET ma60 = EXCLUDED.ma60, ma60_slope = EXCLUDED.ma60_slope, ma60_r2 = EXCLUDED.ma60_r2, close = EXCLUDED.close, consecutive_count = EXCLUDED.consecutive_count, is_above_20_periods = EXCLUDED.is_above_20_periods, created_at = NOW();"