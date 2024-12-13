update 
public.trend_records as a set close=b.close from 
 public.klines b where a.symbol=b.symbol 
and a.timestamp =b.timestamp
alter table public.trend_records add close float null