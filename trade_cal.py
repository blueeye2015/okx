import pandas as pd
import psycopg2

# --------------  本地交易日历工具  --------------
_CAL_CACHE = None          # 内存缓存

def load_trade_cal(conn, start: str, end: str) -> pd.DataFrame:
    """
    从本地 trade_cal 表拉取[start, end]区间交易日历，含去年同一交易日
    返回：DataFrame  index=trade_date  col=prev_trade_date
    """
    global _CAL_CACHE
    if _CAL_CACHE is not None:
        # 内存缓存命中
        return _CAL_CACHE.loc[start:end]

    sql = """
        SELECT cal_date::date      AS trade_date,
               prev_year_date::date AS prev_year_date
        FROM   public.v_trade_cal_ly
        WHERE  is_open = 1
          AND  cal_date BETWEEN %s AND %s
        ORDER  BY cal_date
    """
    cal = pd.read_sql(sql, conn, params=(start, end),
                      parse_dates=['trade_date', 'prev_year_date'])
    cal.set_index('trade_date', inplace=True)
    _CAL_CACHE = cal           # 全局缓存
    return cal