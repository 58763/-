"""缠论选股系统 v5.5（完整稳定版）"""
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
PUSH_TOKEN = 'f084c45f55ca4d658565498255db384b'  # 需替换为实际值（若无需推送可留空）
MIN_MV = 20  # 最小市值（亿）30
MAX_MV = 500  # 最大市值（亿）500
PE_THRESHOLD = 50  # 市盈率警戒值
PB_THRESHOLD = 5  # 市净率警戒值
REQUEST_INTERVAL = 0.5  # 请求间隔（秒）
MAX_WORKERS = 3  # 并行线程数
MACD_FAST = 8  # 优化MACD参数
MACD_SLOW = 20
MACD_SIGNAL = 7
VOLUME_THRESHOLD = 5e7  # 成交量阈值（万股）5e7

# ========== 核心类定义 ==========
class StockAnalyzer:
    _main_board_cache = None  # 主板股票缓存
    
    @classmethod
    def get_main_board(cls):
        """增强版主板列表获取（带自动重试和条件调节）"""
        if cls._main_board_cache is None:
            retry_count = 0
            dynamic_volume = VOLUME_THRESHOLD
            dynamic_mv_min = MIN_MV
            
            while retry_count < 3:
                try:
                    df = ak.stock_zh_a_spot_em()
                    print(f"原始数据量：{len(df)}条")
                    
                    # ==== 数据清洗和类型转换 ====
                    df['总市值'] = pd.to_numeric(df['总市值'], errors='coerce').fillna(0) / 1e8  # 转换为亿单位
                    df['成交量'] = pd.to_numeric(df['成交量'], errors='coerce').fillna(0) * 100  # 手转股数
                    df['代码'] = df['代码'].astype(str).str.zfill(6)
                    
                    # ==== 动态筛选条件 ====
                    filtered_df = df[
                        (df['代码'].str[:2].isin(['60', '00'])) &
                        (df['总市值'].between(
                            dynamic_mv_min,  # 直接使用亿为单位
                            MAX_MV,
                            inclusive='both'
                        )) &
                        (df['成交量'] > dynamic_volume)
                    ].copy()
                    
                    print(f"动态参数：市值>{dynamic_mv_min}亿 成交量>{dynamic_volume/1e4:.0f}万")
                    print(f"筛选后数据量：{len(filtered_df)}条")

                    if not filtered_df.empty:
                        cls._main_board_cache = filtered_df[['代码', '名称']].drop_duplicates('代码')
                        print(f"主板列表已缓存，共{len(cls._main_board_cache)}只股票")
                        return cls._main_board_cache
                    else:
                        print("⚠️ 筛选条件过严，自动放宽条件...")
                        dynamic_volume = max(dynamic_volume * 0.7, 1e7)  # 最低1000万成交量
                        dynamic_mv_min = max(dynamic_mv_min - 5, 10)  # 最低10亿市值
                        retry_count += 1
                        time.sleep(2)
                        
                except Exception as e:
                    print(f"获取股票列表失败: {str(e)}")
                    time.sleep(3)
                    retry_count += 1
                    
            cls._main_board_cache = pd.DataFrame()
            print("❌ 股票列表获取失败，请检查网络或参数")
           
        return cls._main_board_cache

    @classmethod
    def get_stock_name(cls, code):
        """带缓存的名称查询"""
        try:
            df = cls.get_main_board()
            return df[df['代码'] == code.zfill(6)].iloc[0]['名称']
        except:
            return "未知股票"

    @staticmethod
    @lru_cache(maxsize=300)
    def get_enhanced_kline(code, period='daily'):
        """带容错的K线获取"""
        try:
            start_date = (datetime.now() - timedelta(days=730)).strftime("%Y%m%d")
            df = ak.stock_zh_a_hist(
                symbol=code, 
                period=period, 
                adjust="qfq",
                start_date=start_date
            )
            if df.empty: return df
            
            # 数据标准化
            df = df.rename(columns={
                '日期':'date','开盘':'open','收盘':'close',
                '最高':'high','最低':'low','成交量':'volume'
            })
            df = df.sort_values('date', ascending=False).reset_index(drop=True)
            
            # 指标计算
            exp_fast = df['close'].ewm(span=MACD_FAST, adjust=False).mean()
            exp_slow = df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
            df['macd'] = exp_fast - exp_slow
            df['signal'] = df['macd'].ewm(span=MACD_SIGNAL, adjust=False).mean()
            
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            df['tr'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = df['tr'].rolling(14).mean()
            
            return df
        except Exception as e:
            print(f"K线获取失败({code}): {str(e)}")
            return pd.DataFrame()

    @staticmethod
    @lru_cache(maxsize=300)
    def get_30m_kline(code):
        """带容错的30分钟K线获取"""
        try:
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
            df = ak.stock_zh_a_hist_min_em(
                symbol=code, 
                period="30", 
                adjust="qfq",
                start_date=start_date
            )
            if df.empty: return df
            
            # 数据标准化
            df = df.rename(columns={
                '时间':'date','开盘':'open','收盘':'close',
                '最高':'high','最低':'low','成交量':'volume'
            })
            df = df.sort_values('date', ascending=False).reset_index(drop=True)
            
            # 指标计算
            exp_fast = df['close'].ewm(span=MACD_FAST, adjust=False).mean()
            exp_slow = df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
            df['macd'] = exp_fast - exp_slow
            df['signal'] = df['macd'].ewm(span=MACD_SIGNAL, adjust=False).mean()
            
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            df['tr'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = df['tr'].rolling(14).mean()
            
            return df
        except Exception as e:
            print(f"30分钟K线获取失败({code}): {str(e)}")
            return pd.DataFrame()

    @staticmethod
    @lru_cache(maxsize=500)
    def get_fundamental(code):
        """带重试的基本面数据获取"""
        retry = 0
        while retry < 2:
            try:
                df = ak.stock_financial_analysis_indicator(symbol=code)
                if not df.empty:
                    return {
                        'pe': df.iloc[0]['市盈率'],
                        'pb': df.iloc[0]['市净率'],
                        'roe': df.iloc[0]['净资产收益率']
                    }
                retry += 1
                time.sleep(1)
            except:
                retry += 1
                time.sleep(1)
        return {}

class TurboChanEngine:
    """极速缠论引擎（优化分型过滤）"""
    def __init__(self, data):
        self.data = data.sort_values('date', ascending=False).reset_index(drop=True)
        self.bi_list = []
        self.risks = []

    def fast_detect_bi(self):
        """改进的笔识别逻辑（增加分型有效性验证）"""
        if len(self.data) < 7: return []
        
        # 分型检测（需满足至少3根K线验证）
        fx_list = []
        for i in range(2, len(self.data)-2):
            prev2, prev1, curr, next1, next2 = self.data.iloc[i-2:i+3].values
            # 顶分型有效性验证
            if (curr[3] > prev1[3] and curr[3] > next1[3] and 
                curr[3] > prev2[3] and curr[3] > next2[3]):
                fx_list.append({'type': 'top', 'pos': i, 'price': curr[3]})
            # 底分型有效性验证
            elif (curr[4] < prev1[4] and curr[4] < next1[4] and 
                  curr[4] < prev2[4] and curr[4] < next2[4]):
                fx_list.append({'type': 'bottom', 'pos': i, 'price': curr[4]})
        
        # 笔生成（排除相邻同向分型）
        bi_list = []
        prev_fx = None
        for curr_fx in fx_list:
            if prev_fx and (curr_fx['type'] != prev_fx['type']):
                price_diff = curr_fx['price'] - prev_fx['price']
                if abs(price_diff)/prev_fx['price'] > 0.03:  # 过滤幅度<3%的无效笔
                    bi_type = '上升笔' if price_diff > 0 else '下降笔'
                    bi_list.append({
                        'type': bi_type,
                        'start': {'date': self.data.iloc[prev_fx['pos']]['date'], 'price': prev_fx['price']},
                        'end': {'date': self.data.iloc[curr_fx['pos']]['date'], 'price': curr_fx['price']}
                    })
            prev_fx = curr_fx
        
        self.bi_list = bi_list[-3:]  # 取最近3笔
        return self.bi_list

    def evaluate_risks(self):
        """快速风险评估"""
        if self.bi_list:
            last_bi = self.bi_list[-1]
            if abs(last_bi['end']['price'] - last_bi['start']['price']) / last_bi['start']['price'] < 0.05:
                self.risks.append("波动不足5%")
                
        if len(self.data) > 20:
            current_macd = self.data.iloc[0]['macd']
            prev_macd = self.data.iloc[5]['macd']
            if (self.bi_list[-1]['type'] == '上升笔' and current_macd < prev_macd) or \
               (self.bi_list[-1]['type'] == '下降笔' and current_macd > prev_macd):
                self.risks.append("MACD背离")
        
        return list(set(self.risks))

    def detect_buy_sell_points(self):
        """检测缠论买卖点"""
        if len(self.data) < 7: return {}
        
        # 分型检测
        fx_list = []
        for i in range(2, len(self.data)-2):
            prev2, prev1, curr, next1, next2 = self.data.iloc[i-2:i+3].values
            if (curr[3] > prev1[3] and curr[3] > next1[3] and 
                curr[3] > prev2[3] and curr[3] > next2[3]):
                fx_list.append({'type': 'top', 'pos': i, 'price': curr[3]})
            elif (curr[4] < prev1[4] and curr[4] < next1[4] and 
                  curr[4] < prev2[4] and curr[4] < next2[4]):
                fx_list.append({'type': 'bottom', 'pos': i, 'price': curr[4]})
        
        # 笔生成
        bi_list = []
        prev_fx = None
        for curr_fx in fx_list:
            if prev_fx and (curr_fx['type'] != prev_fx['type']):
                price_diff = curr_fx['price'] - prev_fx['price']
                if abs(price_diff)/prev_fx['price'] > 0.03:
                    bi_type = '上升笔' if price_diff > 0 else '下降笔'
                    bi_list.append({
                        'type': bi_type,
                        'start': {'date': self.data.iloc[prev_fx['pos']]['date'], 'price': prev_fx['price']},
                        'end': {'date': self.data.iloc[curr_fx['pos']]['date'], 'price': curr_fx['price']}
                    })
            prev_fx = curr_fx
        
        # 买卖点检测
        buy_points = []
        sell_points = []
        if len(bi_list) >= 3:
            # 第一买点
            if bi_list[0]['type'] == '下降笔' and bi_list[1]['type'] == '上升笔' and bi_list[2]['type'] == '下降笔':
                buy_points.append({
                    'type': '第一买点',
                    'price': bi_list[1]['end']['price']
                })
            # 第二买点
            if bi_list[0]['type'] == '上升笔' and bi_list[1]['type'] == '下降笔' and bi_list[2]['type'] == '上升笔':
                buy_points.append({
                    'type': '第二买点',
                    'price': bi_list[2]['end']['price']
                })
            # 第三买点
            if bi_list[0]['type'] == '下降笔' and bi_list[1]['type'] == '上升笔' and bi_list[2]['type'] == '下降笔':
                if bi_list[2]['end']['price'] < bi_list[0]['end']['price']:
                    buy_points.append({
                        'type': '第三买点',
                        'price': bi_list[2]['end']['price']
                    })
            # 第一卖点
            if bi_list[0]['type'] == '上升笔' and bi_list[1]['type'] == '下降笔' and bi_list[2]['type'] == '上升笔':
                sell_points.append({
                    'type': '第一卖点',
                    'price': bi_list[1]['end']['price']
                })
            # 第二卖点
            if bi_list[0]['type'] == '下降笔' and bi_list[1]['type'] == '上升笔' and bi_list[2]['type'] == '下降笔':
                sell_points.append({
                    'type': '第二卖点',
                    'price': bi_list[2]['end']['price']
                })
            # 第三卖点
            if bi_list[0]['type'] == '上升笔' and bi_list[1]['type'] == '下降笔' and bi_list[2]['type'] == '上升笔':
                if bi_list[2]['end']['price'] > bi_list[0]['end']['price']:
                    sell_points.append({
                        'type': '第三卖点',
                        'price': bi_list[2]['end']['price']
                    })
        
        return {
            'buy_points': buy_points,
            'sell_points': sell_points
        }

# ========== 辅助系统 ==========
class QuantKit:
    """极速分析工具包（评分系统优化）"""
    @staticmethod
    def analyze_volume(data):
        """量能分析"""
        if len(data) < 5: return "数据不足"
        return "放量" if data['volume'].iloc[0] > data['volume'].iloc[1:5].mean() * 1.5 else "平量"

    @staticmethod
    def calculate_score(daily_bi, weekly_bi, daily_data, weekly_data):
        """增强型评分系统（增加趋势力度评估）"""
        score = 0
        
        # 趋势评分（50%）
        trend_score = 0
        if daily_bi and daily_bi[-1]['type'] == '上升笔': 
            trend_score += 5
            # 评估趋势力度
            price_change = (daily_bi[-1]['end']['price'] - daily_bi[-1]['start']['price'])/daily_bi[-1]['start']['price']
            if price_change > 0.1: trend_score += 2
        if weekly_bi and weekly_bi[-1]['type'] == '上升笔': 
            trend_score += 4
        score += trend_score * 0.5

        # MACD动能（25%）
        macd_status = 0
        if daily_data.iloc[0]['macd'] > daily_data.iloc[0]['signal']: macd_status += 3
        if weekly_data.iloc[0]['macd'] > weekly_data.iloc[0]['signal']: macd_status += 2
        score += macd_status * 0.25

        # 量价配合（15%）
        volume_score = 0
        if QuantKit.analyze_volume(daily_data) == "放量":
            volume_score += 3 if daily_data.iloc[0]['close'] > daily_data.iloc[1]['close'] else 1
        score += volume_score * 0.15

        # 波动率评估（10%）
        atr_score = 2 if daily_data.iloc[0]['atr'] > daily_data['atr'].mean() else 0
        score += atr_score * 0.1

        return min(score + np.random.randint(0,2), 10)  # 减小随机扰动
class RiskManagerPro:
    """风控增强版（动态ATR止损）"""
    @staticmethod
    def get_position(rsi_value):
        """智能仓位"""
        if rsi_value < 30: return "30%-40%"
        elif rsi_value < 70: return "15%-25%"
        else: return "5%-10%"

    @staticmethod
    def get_stop_loss(data):
        """基于ATR的动态止损（增加数据检查）"""
        try:
            # 检查数据是否有效
            if data.empty or 'close' not in data.columns or 'low' not in data.columns:
                return "数据无效"
            
            # 优先使用 ATR 计算止损
            if 'atr' in data.columns and not pd.isna(data.iloc[0]['atr']):
                atr_value = data.iloc[0]['atr']
                if atr_value > 0:  # 确保 ATR 有效
                    stop_loss = data.iloc[0]['close'] - 2 * atr_value
                    return f"{stop_loss:.2f}"
            
            # 如果 ATR 无效，使用最近 5 日最低价的 97% 作为止损
            recent_low = data['low'].iloc[:5].min()
            if not pd.isna(recent_low) and recent_low > 0:
                return f"{recent_low * 0.97:.2f}"
            
            # 如果以上方法都失败，返回默认值
            return "无有效止损"
        except Exception as e:
            print(f"止损计算失败: {str(e)}")
            return "计算错误"

# ========== 执行引擎 ==========
def turbo_analyze(code):
    """带健康检查的分析流程"""
    try:
        code = str(code).zfill(6)
        
        # 并行获取数据
        with ThreadPoolExecutor(max_workers=3) as executor:
            daily_future = executor.submit(StockAnalyzer.get_enhanced_kline, code, 'daily')
            weekly_future = executor.submit(StockAnalyzer.get_enhanced_kline, code, 'weekly')
            thirty_min_future = executor.submit(StockAnalyzer.get_30m_kline, code)
            daily, weekly, thirty_min = daily_future.result(), weekly_future.result(), thirty_min_future.result()
        
        if daily.empty or weekly.empty or thirty_min.empty:
            print(f"⚠️ {code} 数据不完整")
            return None
        
        # 并行分析
        with ThreadPoolExecutor(max_workers=3) as executor:
            daily_engine = TurboChanEngine(daily)
            weekly_engine = TurboChanEngine(weekly)
            thirty_min_engine = TurboChanEngine(thirty_min)
            future_d = executor.submit(daily_engine.fast_detect_bi)
            future_w = executor.submit(weekly_engine.fast_detect_bi)
            future_30m = executor.submit(thirty_min_engine.detect_buy_sell_points)
            bi_daily, bi_weekly, buy_sell_points = future_d.result(), future_w.result(), future_30m.result()
        
        # 风险检测
        risks = []
        daily_engine.evaluate_risks()
        weekly_engine.evaluate_risks()
        risks += daily_engine.risks + weekly_engine.risks
        
        # 基本面检查
        fundamental = StockAnalyzer.get_fundamental(code)
        if fundamental.get('pe', 0) > PE_THRESHOLD:
            risks.append(f"PE({fundamental['pe']})")
        if fundamental.get('pb', 0) > PB_THRESHOLD:
            risks.append(f"PB({fundamental['pb']})")
        
        return {
            '代码': code,
            '名称': StockAnalyzer.get_stock_name(code),
            '日线': f"{bi_daily[-1]['type']}" if bi_daily else "N/A",
            '周线': f"{bi_weekly[-1]['type']}" if bi_weekly else "N/A",
            '量能': QuantKit.analyze_volume(daily),
            '评分': QuantKit.calculate_score(bi_daily, bi_weekly, daily, weekly),
            '风险': list(set(risks))[:3],
            '建议': [
                f"仓位: {RiskManagerPro.get_position(daily.iloc[0].get('rsi', 50))}",
                f"止损: {RiskManagerPro.get_stop_loss(daily)}"
            ],
            '30分钟缠论': {
                '买入点': [f"{point['type']}: {point['price']:.2f}" for point in buy_sell_points['buy_points']],
                '卖出点': [f"{point['type']}: {point['price']:.2f}" for point in buy_sell_points['sell_points']]
            }
        }
    except Exception as e:
        print(f"分析失败({code}): {str(e)}")
        return None

# ========== 主控制系统 ==========
def main_controller(code=None):
    """带预检的任务调度"""
    print("\n=== 系统初始化检查 ===")
    print(f"AKShare版本: {ak.__version__}")
    print(f"Pandas版本: {pd.__version__}")
    print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("====================\n")
    
    # 获取股票列表
    if code:
        stock_list = [str(code).zfill(6)]
        print(f"🔍 开始分析指定股票 {code}")
    else:
        base_df = StockAnalyzer.get_main_board()
        if base_df.empty:
            print(f"""
            ❗ 无法获取股票列表，请检查：
            1. 网络连接是否正常
            2. 是否安装最新版AKShare（pip install --upgrade akshare）
            3. 尝试修改以下参数：
               - 调低 MIN_MV（当前值：{MIN_MV}亿）
               - 调低 VOLUME_THRESHOLD（当前值：{VOLUME_THRESHOLD/1e4:.0f}万）
            """)
            return
        stock_list = base_df['代码'].tolist()
        print(f"🚀 启动全市场扫描，共{len(stock_list)}只股票")
    
    results = []
    
    # 并行处理
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(turbo_analyze, code): code for code in stock_list}
        
        # 进度监控
        progress = tqdm(as_completed(futures), total=len(stock_list), 
                       desc="极速分析", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}")
        
        for future in progress:
            if result := future.result():
                results.append(result)
            time.sleep(REQUEST_INTERVAL)
            progress.set_postfix({'完成数': len(results), '最新代码': list(futures.values())[0]})
    
    # 结果处理
    if results:
        df = pd.DataFrame(results).sort_values('评分', ascending=False)
        print(f"\n📊 分析完成，有效结果：{len(df)}条")
        
        # 生成报告
        report = f"""
📈 缠论选股报告（{datetime.now().strftime('%Y-%m-%d %H:%M')}）
============================================
{df.head(5).to_markdown(index=False)}

► 30分钟缠论买卖点：
{df[['代码', '名称', '30分钟缠论']].head(10).to_markdown(index=False)}

► 风险提示：
- 评分系统含随机扰动因子（±1分）
- 止损价基于最近5日最低价或ATR动态计算
- 需结合大盘环境综合判断

► 系统参数：
- 市值范围：{MIN_MV}~{MAX_MV}亿 | PE警戒：{PE_THRESHOLD} | PB警戒：{PB_THRESHOLD}
- MACD参数：{MACD_FAST}-{MACD_SLOW}-{MACD_SIGNAL}
- 数据源：AKShare {ak.__version__}
        """
        print(report)
        
        # 微信推送
        if PUSH_TOKEN and PUSH_TOKEN.strip():  # 修改判断条件
            max_retries = 2
            backoff_factor = 1.5
            
            # 调试输出
            print(f"调试信息 | Token长度: {len(PUSH_TOKEN)} 内容: |{PUSH_TOKEN}|")
            
            for attempt in range(max_retries + 1):
                try:
                    response = requests.post(
                        "https://www.pushplus.plus/send",
                        json={
                            "token": PUSH_TOKEN,
                            "title": f"📈 最终进阶版选股报告 {datetime.now().strftime('%m-%d')}",
                            "content": report.replace('\n', '<br>'),
                            "template": "html",
                            "channel": "wechat",
                            "timestamp": int(time.time()*2000)
                        },
                        timeout=10
                    )
                    # 添加响应处理（可选）
                    if response.status_code == 200:
                        print("推送成功")
                        break
                except Exception as e:
                    print(f"推送失败（尝试 {attempt+1}/{max_retries}）: {str(e)}")
                    time.sleep(backoff_factor ** attempt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="缠论选股系统")
    parser.add_argument("--code", type=str, help="指定股票代码（如：600000）")
    args = parser.parse_args()
    
    main_controller(code=args.code)
