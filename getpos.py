import okex.Account_api as Account
import okex.Trade_api as Trade
import json
from api_parser import parse_positions,parse_orderlist
import logging

if __name__ == '__main__':
    api_key = "ba7f444f-e83e-4dd1-8507-bf8dd9033cbc"
    secret_key = "D5474EF76B0A7397BFD26B9656006480"
    passphrase = "TgTB+pJoM!d20F"

    # 设置日志配置
    logging.basicConfig(filename='output.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('脚本开始执行')

    # 设置代理
    proxies = {
    'http': 'http://127.0.0.1:7890',
    'https': 'http://127.0.0.1:7890'
    }
    # flag是实盘与模拟盘的切换参数
    # flag = '1'  # 模拟盘
    flag = '0'  # 实盘

    # account api
    accountAPI = Account.AccountAPI(api_key, secret_key, passphrase, False, flag, proxies)
    # 查看账户余额  Get Balance
    # result = accountAPI.get_account('BTC')
    # 查看持仓信息  Get Positions
    result = accountAPI.get_positions('SWAP', '')
    
    # trade api
    tradeAPI = Trade.TradeAPI(api_key, secret_key, passphrase, False, flag, proxies=proxies)
    # 使用函数
    api_response = json.dumps(result)  # 这里应该是完整的API响应
    try:
        positions = parse_positions(api_response)
        for pos in positions:
            logging.info(f"Symbol: {pos['symbol']}")
            logging.info(f"Position Size: {pos['size']}")
            logging.info(f"Side: {pos['side']}")
            logging.info(f"Liquidation Price: {pos['liquidation_price']}")
            logging.info(f"uplRatio: {pos['uplRatio']}")
            logging.info(f"Leverage: {pos['leverage']}")
            logging.info(f"margin: {pos['margin']}")
            logging.info("------------------------")

            #如果收益率>=-30%,补一次，但要判断是否已经存在一样的委托
            if float(pos['uplRatio'])<=-0.4:
                #获取该合约未完成订单
                result1 = tradeAPI.get_order_list(instType='SWAP',instId=pos['symbol'])
                orderlist = parse_orderlist(json.dumps(result1))
                if not orderlist:
                    logging.info (f"{pos['symbol']} 没有{pos['side']}订单,收益率 :{pos['uplRatio']} ")
                    logging.info ("----下同币种单子-----")
                    #计算强平价格，如果是buy则*1.2 ，如果是sell则*0.8
                    price = float(pos['liquidation_price'])*(1-0.02) if pos['side'] == 'sell' else float(pos['liquidation_price'])*(1+0.02)
                    #计算止盈价格，如果保证金低于50，止盈25% 5倍杠杆就是挂单价格*1.05，如果保证金低于100 止盈20% 挂单价格*1.04 如果是低于200 止盈15% 即挂单价格*1.03 如果是低于300 止盈10% 即挂单价格*1.02 其余挂单价格*1.01
                    if pos['side'] == 'buy':
                        #tpprice = price*0.99 if pos['side'] == 'sell' else price*1.01
                        tpprice = price*1.05 if float(pos['margin']) <= 50 \
                            else price*1.04 if float(pos['margin']) <= 100 \
                            else price*1.03 if float(pos['margin']) <= 200 \
                            else price*1.02 if float(pos['margin']) <= 300 else price*1.01
                    else:
                        tpprice = price*0.95 if float(pos['margin']) <= 50 \
                            else price*0.96 if float(pos['margin']) <= 100 \
                            else price*0.97 if float(pos['margin']) <= 200 \
                            else price*0.98 if float(pos['margin']) <= 300 else price*0.99
                    logging.info (f"price: {price} tpprice: {tpprice}")
                    order_reslut = tradeAPI.place_order(instId=pos['symbol'], tdMode='isolated', side=pos['side'],
                                   ordType='limit', sz=abs(pos['size']), px = price, tpTriggerPx=tpprice,tpOrdPx=-1)
                    logging.info(json.dumps(order_reslut))
            elif float(pos['uplRatio'])>=-0.4 and float(pos['uplRatio'])<=0:
                #判断收益率如果大于-30%，则取消未完成订单
                #获取该合约未完成订单
                result1 = tradeAPI.get_order_list(instType='SWAP',instId=pos['symbol'])
                orderlist = parse_orderlist(json.dumps(result1))
                logging.info (f"{pos['symbol']} 的{pos['side']}订单,收益率 :{pos['uplRatio']} ")
                for order in orderlist:
                    order_reslut = tradeAPI.cancel_order(instId=pos['symbol'], ordId=order['ordId'])
                    logging.info (f"{order['symbol']} 的{order['side']}订单 ordid :{order['ordId']} 撤销")
    except Exception as e:
        logging.info(f"Error: {e}")

    
