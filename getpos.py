import okex.Account_api as Account
import okex.Trade_api as Trade
import json
from api_parser import parse_positions,parse_orderlist


if __name__ == '__main__':
    api_key = "ba7f444f-e83e-4dd1-8507-bf8dd9033cbc"
    secret_key = "D5474EF76B0A7397BFD26B9656006480"
    passphrase = "TgTB+pJoM!d20F"

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
            print(f"Symbol: {pos['symbol']}")
            print(f"Position Size: {pos['size']}")
            print(f"Side: {pos['side']}")
            print(f"Liquidation Price: {pos['liquidation_price']}")
            print(f"uplRatio: {pos['uplRatio']}")
            print(f"Leverage: {pos['leverage']}")
            print("------------------------")

            #如果收益率>=-30%,补一次，但要判断是否已经存在一样的委托
            if float(pos['uplRatio'])<=-0.4:
                #获取该合约未完成订单
                result1 = tradeAPI.get_order_list(instType='SWAP',instId=pos['symbol'])
                orderlist = parse_orderlist(json.dumps(result1))
                if not orderlist:
                    print (f"{pos['symbol']} 没有{pos['side']}订单,收益率 :{pos['uplRatio']} ")
                    print ("----下同币种单子-----")
                    #计算强平价格，如果是buy则*1.2 ，如果是sell则*0.8
                    price = float(pos['liquidation_price'])*(1-0.02) if pos['side'] == 'sell' else float(pos['liquidation_price'])*(1+0.02)
                    #计算止盈价格，价格等于挂单价格*1.05
                    tpprice = price*0.99 if pos['side'] == 'sell' else price*1.01
                    print (f"price: {price} tpprice: {tpprice}")
                    order_reslut = tradeAPI.place_order(instId=pos['symbol'], tdMode='isolated', side=pos['side'],
                                   ordType='limit', sz=abs(pos['size']), px = price, tpTriggerPx=tpprice,tpOrdPx=-1)
                    print(json.dumps(order_reslut))
            elif float(pos['uplRatio'])>=-0.4 and float(pos['uplRatio'])<=0:
                #判断收益率如果大于-30%，则取消未完成订单
                #获取该合约未完成订单
                result1 = tradeAPI.get_order_list(instType='SWAP',instId=pos['symbol'])
                orderlist = parse_orderlist(json.dumps(result1))
                print (f"{pos['symbol']} 的{pos['side']}订单,收益率 :{pos['uplRatio']} ")
                for order in orderlist:
                    order_reslut = tradeAPI.cancel_order(instId=pos['symbol'], ordId=order['ordId'])
                    print (f"{order['symbol']} 的{order['side']}订单 ordid :{order['ordId']} 撤销")
    except Exception as e:
        print(f"Error: {e}")

    
