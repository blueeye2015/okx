import okex.Account_api as Account
import json
from api_parser import parse_positions, parse_orderlist, parse_balance
import logging
from regular_err import SpecialJumpException

class OkexAccountManager:
    def __init__(self, api_key, secret_key, passphrase, is_simulated=False):
        """
        初始化OkexAccountManager
        
        Args:
            api_key (str): API密钥
            secret_key (str): 密钥
            passphrase (str): API密码
            is_simulated (bool): 是否使用模拟盘，默认False表示实盘
        """
        # 设置API凭证
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        
        # 设置是否为模拟盘
        self.flag = '1' if is_simulated else '0'
        
        # 设置代理
        self.proxies = {
            'http': 'http://127.0.0.1:7890',
            'https': 'http://127.0.0.1:7890'
        }
        
        # 初始化API
        self.init_api()
        
        # 设置日志
        self.setup_logging()

    def setup_logging(self):
        """设置日志配置"""
        logging.basicConfig(
            filename='output.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info('OkexAccountManager initialized')

    def init_api(self):
        """初始化API连接"""
        try:
            self.account_api = Account.AccountAPI(
                self.api_key,
                self.secret_key,
                self.passphrase,
                False,
                self.flag,
                self.proxies
            )
        except Exception as e:
            self.logger.error(f"API初始化失败: {str(e)}")
            raise

    def get_positions(self, instrument_type='SWAP', instrument_id=''):
        """
        获取持仓信息
        
        Args:
            instrument_type (str): 产品类型，默认'SWAP'
            instrument_id (str): 产品ID，默认为空
            
        Returns:
            dict: 解析后的持仓信息
        """
        try:
            result = self.account_api.get_positions(instrument_type, instrument_id)
            self.logger.info("Successfully retrieved positions")
            return result
        except Exception as e:
            self.logger.error(f"获取持仓信息失败: {str(e)}")
            raise

    def get_balance(self, currency='USDT'):
        """
        获取保证金余额
        
        Args:
            currency (str): 货币类型，默认'USDT'
            
        Returns:
            float: 解析后的余额信息
        """
        try:
            result = self.account_api.get_account(currency)
            balance = parse_balance(json.dumps(result))
            self.logger.info(f"Successfully retrieved balance for {currency}")
            return balance
        except Exception as e:
            self.logger.error(f"获取余额失败: {str(e)}")
            raise

    def get_account_info(self):
        """
        获取账户完整信息（包括持仓和余额）
        
        Returns:
            dict: 包含持仓和余额信息的字典
        """
        try:
            positions = self.get_positions()
            balance = self.get_balance()
            
            return {
                'positions': positions,
                'balance': balance
            }
        except Exception as e:
            self.logger.error(f"获取账户信息失败: {str(e)}")
            raise

