from account import OkexAccountManager

if __name__ == '__main__':
    # API凭证
    api_key = "ba7f444f-e83e-4dd1-8507-bf8dd9033cbc"
    secret_key = "D5474EF76B0A7397BFD26B9656006480"
    passphrase = "TgTB+pJoM!d20F"

    # 创建账户管理器实例
    account_manager = OkexAccountManager(
        api_key=api_key,
        secret_key=secret_key,
        passphrase=passphrase,
        is_simulated=False  # False表示实盘
    )

    try:
        # 获取持仓信息
        positions = account_manager.get_positions()
        print("持仓信息:", positions)

        # 获取余额
        balance = account_manager.get_balance()
        print("账户余额:", balance)

        # 获取完整账户信息
        account_info = account_manager.get_account_info()
        print("完整账户信息:", account_info)

    except Exception as e:
        print(f"发生错误: {str(e)}")
