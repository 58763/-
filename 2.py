"""缠论第一买点选股系统"""
import argparse
import time
import akshare as ak
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from tqdm.auto import tqdm

# ========== 系统配置 ==========
PUSH_TOKEN = 'feef5ded80954b2c891a889d683420a6'
MIN_MV = 80  
MAX_MV = 300  
REQUEST_INTERVAL = 0.5  
MAX_WORKERS = 3  
MACD_FAST = 6  
MACD_SLOW = 13
MACD_SIGNAL = 5
VOLUME_THRESHOLD = 5e7  

# ========== 核心类定义 ==========
class StockAnalyzer:
    _main_board_cache = None  

    @classmethod
    def get_main_board(cls):
        """主板列表获取"""
        if cls._main_board_cache is None:
            df = ak.stock_zh_a_spot_em()
            df['总市值'] = pd.to_numeric(df['总市值'], errors='coerce').fillna(0) / 1e8
            df['成交量'] = pd.to_numeric(df['成交量'], errors='coerce').fillna(0) * 100
            df['代码'] = df['代码'].astype(str).str.zfill(6)
            
            filtered_df = df[
                (df['代码'].str[:2].isin(['60', '00'])) &
                (df['总市值'].between(MIN_MV, MAX_MV)) &
                (df['成交量'] > VOLUME_THRESHOLD)
            ].copy()
            
            cls._main_board_cache = filtered_df[['代码', '名称']].drop_duplicates('代码')
        return cls._main_board_cache

    @classmethod
    def get_stock_name(cls, code):
        """名称查询（修正）"""
        try:
            df = ak.stock_zh_a_spot_em()
            df['代码'] = df['代码'].astype(str).str.zfill(6)
            return df[df['代码'] == code.zfill(6)].iloc[0]['名称']
        except:
            return "未知股票"

    @staticmethod
    def calculate_macd(df):
        """MACD计算"""
        df = df.sort_values('date')
        df['EMA_Fast'] = df['close'].ewm(span=MACD_FAST, adjust=False).mean()
        df['EMA_Slow'] = df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
        df['macd'] = df['EMA_Fast'] - df['EMA_Slow']
        df['signal'] = df['macd'].ewm(span=MACD_SIGNAL, adjust=False).mean()
        return df.drop(columns=['EMA_Fast', 'EMA_Slow']).fillna(0)

    @staticmethod
    @lru_cache(maxsize=300)
    def get_enhanced_kline(code, period='daily'):
        """K线数据获取"""
        try:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
            df = ak.stock_zh_a_hist(symbol=code, period=period, adjust="qfq", start_date=start_date)
            df = df.rename(columns={'日期':'date','开盘':'open','收盘':'close','最高':'high','最低':'low'})
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            return StockAnalyzer.calculate_macd(df.sort_values('date', ascending=False))
        except:
            return pd.DataFrame()

# ========== 缠论引擎 ==========
class TurboChanEngine:
    """第一买点检测引擎"""
    def __init__(self, data):
        self.data = data.sort_values('date', ascending=True).reset_index(drop=True)
        self.bi_list = []

    def detect_fenxing(self):
        """分型检测（修正后）"""
        fx_list = []
        for i in range(1, len(self.data)-1):
            # 底分型
            current_low = self.data.iloc[i]['low']
            prev_low = self.data.iloc[i-1]['low']
            next_low = self.data.iloc[i+1]['low']
            if current_low < prev_low and current_low < next_low:
                fx_list.append({'type': 'bottom', 'pos': i, 'price': current_low})
            # 顶分型
            current_high = self.data.iloc[i]['high']
            prev_high = self.data.iloc[i-1]['high']
            next_high = self.data.iloc[i+1]['high']
            if current_high > prev_high and current_high > next_high:
                fx_list.append({'type': 'top', 'pos': i, 'price': current_high})
        return fx_list

    def fast_detect_bi(self):
        """笔识别（修正索引字段）"""
        bi_list = []
        prev_fx = None
        for curr_fx in self.detect_fenxing():
            if prev_fx and (curr_fx['type'] != prev_fx['type']):
                bi_list.append({
                    'type': '上升笔' if curr_fx['price'] > prev_fx['price'] else '下降笔',
                    'start': str(self.data.iloc[prev_fx['pos']]['date']),
                    'start_price': prev_fx['price'],
                    'end': str(self.data.iloc[curr_fx['pos']]['date']),
                    'end_price': curr_fx['price'],
                    'start_idx': prev_fx['pos'],
                    'end_idx': curr_fx['pos']
                })
            prev_fx = curr_fx
        self.bi_list = bi_list
        return bi_list

    def detect_first_buy(self):
        """第一买点检测（修正后）"""
        if len(self.bi_list) < 3:
            return []

        buy_points = []
        for i in range(len(self.bi_list)-2):
            bi1, bi2, bi3 = self.bi_list[i], self.bi_list[i+1], self.bi_list[i+2]
            if bi1['type'] == '下降笔' and bi2['type'] == '上升笔' and bi3['type'] == '下降笔':
                macd_area1 = self.data['macd'].iloc[bi1['start_idx']:bi1['end_idx']].sum()
                macd_area3 = self.data['macd'].iloc[bi3['start_idx']:bi3['end_idx']].sum()
                if bi3['end_price'] < bi1['end_price'] and macd_area3 > macd_area1:
                    buy_points.append({
                        'type': '第一买点',
                        'price': bi3['end_price'],
                        'date': self.data.iloc[bi3['end_idx']]['date']
                    })
        return buy_points

# ========== 执行引擎 ==========
def analyze_stock(code):
    """分析流程（新增涨幅获取）"""
    try:
        code = str(code).zfill(6)
        daily = StockAnalyzer.get_enhanced_kline(code)
        if daily.empty:
            return None
        
        # 获取实时数据
        spot_df = ak.stock_zh_a_spot_em()
        spot_df['代码'] = spot_df['代码'].astype(str).str.zfill(6)
        stock_data = spot_df[spot_df['代码'] == code]
        
        latest_price = stock_data['最新价'].values[0]
        change_percent = stock_data['涨跌幅'].values[0]  # 新增涨跌幅字段

        engine = TurboChanEngine(daily)
        engine.fast_detect_bi()
        buy_points = engine.detect_first_buy()

        return {
            '代码': code,
            '名称': StockAnalyzer.get_stock_name(code),
            '最新价': latest_price,
            '涨跌幅': change_percent,  # 返回涨跌幅
            '买点': buy_points[-1] if buy_points else None
        }
    except Exception as e:
        print(f"股票 {code} 分析异常: {str(e)}")
        return None

# ========== 主控制系统 ==========
def main_controller(code=None):
    """主控制（新增涨幅过滤）"""
    base_df = StockAnalyzer.get_main_board()
    stock_list = [str(code).zfill(6)] if code else base_df['代码'].tolist()
    
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(analyze_stock, code): code for code in stock_list}
        for future in tqdm(as_completed(futures), total=len(stock_list), desc="分析进度"):
            if result := future.result():
                print(f"股票 {result['代码']} 分析结果：买点存在？{bool(result['买点'])} | 涨幅：{result['涨跌幅']:.2f}%")
                # 过滤条件：存在买点且涨幅≤5%
                if result['买点'] and result['涨跌幅'] <= 5.0:
                    results.append(result)
            time.sleep(REQUEST_INTERVAL)

    if results:
        # 按买点时间排序
        results.sort(key=lambda x: x['买点']['date'], reverse=True)
        
        full_report = f"""
📈 <strong>【大柚子选股报告】</strong>  
⏰ 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}  
═══════════════════════════════
        """
        
        for idx, stock in enumerate(results[:5], 1):
            change_sign = '+' if stock['涨跌幅'] >= 0 else ''
            full_report += f"""
🔍 <strong>第{idx}个信号</strong>  
▸ 代码：{stock['代码']}  
▸ 名称：{stock['名称']}  
▸ 最新价：<span style="color: #FF4500;">{stock['最新价']:.2f}</span>  
▸ 当前涨幅：<span style="color: {'#32CD32' if stock['涨跌幅'] >=0 else '#FF0000'};">{change_sign}{stock['涨跌幅']:.2f}%</span>  
▸ 买点价：{stock['买点']['price']:.2f}  
▸ 买点时间：📅 {stock['买点']['date']}  
──────────────
            """
        
        full_report += """
<em>⚠️ 提示：涨幅≤5%，买点需结合其他指标验证。</em>  
✨ 数据来源：AKShare | 缠论引擎V2.2
        """
        
        print("检测到的买点结果:", results)
        if PUSH_TOKEN.strip():
            try:
                response = requests.post(
                    "https://www.pushplus.plus/send",
                    json={
                        "token": PUSH_TOKEN,
                        "title": f"🔔 柚子日志 {datetime.now().strftime('%m-%d')}",
                        "content": full_report.replace('\n', '<br>'),
                        "template": "html"
                    }
                )
                response.raise_for_status()
                print("推送成功")
            except Exception as e:
                print(f"推送失败: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="缠论第一买点选股系统")
    parser.add_argument("--code", type=str, help="指定股票代码（如：600000）")
    args = parser.parse_args()
    main_controller(code=args.code)
