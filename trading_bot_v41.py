import gymnasium as gym
import numpy as np
import pandas as pd
import ta  # Technical Analysis Library
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from binance.client import Client
from binance.exceptions import BinanceAPIException
from urllib3.exceptions import ReadTimeoutError
from flask import Flask, jsonify
from threading import Thread
import datetime
import os
import time
import json
import requests
import torch
import torch.nn as nn
from decimal import Decimal, ROUND_DOWN
import logging
import traceback
import shutil 
from ta.momentum import RSIIndicator

# Binance API Keys (Testnet)
API_KEY = 'RBHNQbMZTLCdemkAsR61AcH15Kc6YF5zGQVaWAF4nWh1MyPVe8MnbSc72DtJasMB'
API_SECRET = 'U3RUCw2xp6YDoXHDwNcEsfAAyi3Z0OdgR1ZFjzkrj8e4y4gKRiFy8QBOjEX0aBrP'

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def safe_binance_request(request_func, max_retries=7, delay=10):
    """Выполняет безопасный запрос к Binance API с повторными попытками при ошибках."""
    for attempt in range(max_retries):
        try:
            response = request_func()
            # logger.info(f"Успешный запрос к Binance API с попытки {attempt + 1}")
            return response
        except (requests.exceptions.ReadTimeout, ReadTimeoutError, BinanceAPIException) as e:
            logger.warning(f"Ошибка при запросе к Binance API: {e}. Попытка {attempt + 1}/{max_retries}.")
            time.sleep(delay)
        except Exception as e:
            logger.error(f"Неизвестная ошибка: {e}\n{traceback.format_exc()}")
            if "502 Bad Gateway" in str(e) or "Invalid JSON error message" in str(e):
                logger.warning(f"502 Bad Gateway или ошибка сети: {e}. Попытка {attempt + 1}/{max_retries}.")
                time.sleep(delay)
            else:
                break
    logger.error(f"Ошибка: превышено количество попыток {max_retries}. Запрос не выполнен.")
    return None

def initialize_binance_client(api_key, api_secret):
    """Инициализация клиента Binance с повторными попытками при ошибках."""
    def create_client():
        return Client(api_key, api_secret, testnet=True, requests_params={'timeout': 40})  # Увеличиваем тайм-аут

    client = safe_binance_request(create_client, max_retries=7, delay=10)
    if client is None:
        raise Exception("Не удалось инициализировать клиента Binance после нескольких попыток.")
    return client

# Инициализация клиента Binance через безопасную функцию
client = initialize_binance_client(API_KEY, API_SECRET)

# Глобальная переменная для среды торговли
env = None

class CustomCNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(CustomCNNFeatureExtractor, self).__init__(observation_space, features_dim)

        # Заменяем количество входных каналов на 1 для правильного входа данных
        self.depth_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Заменяем количество входных каналов на 1 для свечных данных
        self.candle_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Полносвязные слои для обработки дополнительных данных
        self.fc = nn.Sequential(
            nn.Linear(self._get_conv_output_shape(observation_space) + observation_space.shape[0] - 333, 256), # Обработка несвёрточных данных
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )

    def _get_conv_output_shape(self, observation_space):
        with torch.no_grad():
            # Определяем размер выхода после свёрток
            sample_input = torch.zeros(1, 1, 64, 64)  # Используем только один канал
            depth_out = self.depth_conv(sample_input)
            candle_out = self.candle_conv(sample_input)
            return depth_out.size(1) + candle_out.size(1)

    def forward(self, observations):
        # Разделение входов
        depth_data = observations[:, :1, :, :]  # Данные глубины (1 канал)
        candle_data = observations[:, 1:2, :, :]  # Данные свечей (1 канал)
        other_data = observations[:, 2:].view(observations.size(0), -1)  # Несвёрточные данные
        
        # Применение свёрток к глубине и свечам
        depth_features = self.depth_conv(depth_data)
        candle_features = self.candle_conv(candle_data)
        
        # Объединение признаков и применение полносвязных слоев
        combined_features = torch.cat((depth_features, candle_features, other_data), dim=1)
        return self.fc(combined_features)

# Flask API
app = Flask(__name__)

@app.route('/status', methods=['GET'])
def get_status():
    global env  # Используем глобальную переменную env
    return jsonify({
        'step': env.current_step,
        'balance_usdt': env.get_current_usdt_balance(),
        'portfolio_value_usdt': env.current_portfolio_value,
        'traded_assets': env.traded_assets,
    })

# Функция для получения часовых свечей за последнюю неделю
def get_hourly_candles(symbol, limit=168):
    candles = safe_binance_request(lambda: client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1HOUR, limit=limit))
    if candles is None or len(candles) == 0:
        print(f"Нет данных свечей для {symbol}")
        return []
    return candles

# Дополнительная функция для получения дневных свечей за год
def get_daily_candles(symbol, limit=365):
    candles = safe_binance_request(lambda: client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1DAY, limit=limit))
    if candles is None or len(candles) == 0:
        print(f"Нет данных дневных свечей для {symbol}")
        return []
    return candles

def get_15min_candles(symbol, limit=96):
    """Получает 15-минутные свечи за указанный период."""
    candles = safe_binance_request(lambda: client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_15MINUTE, limit=limit))
    if candles is None or len(candles) == 0:
        print(f"Нет данных 15-минутных свечей для {symbol}")
        return []
    return candles

def get_current_price(symbol):
    """Получает текущую цену указанного символа."""
    ticker = safe_binance_request(lambda: client.get_symbol_ticker(symbol=symbol))
    return float(ticker['price']) if ticker else 0.0

def get_market_depth(self, symbol, limit=5):
    depth = safe_binance_request(lambda: self.client.get_order_book(symbol=symbol, limit=limit))
    if depth:
        bids = [float(bid[0]) for bid in depth['bids'][:limit]]  # Цены и объемы на покупку
        asks = [float(ask[0]) for ask in depth['asks'][:limit]]  # Цены и объемы на продажу
        return bids + asks
    else:
        return [0.0] * (limit * 2)  # Заполняем нулями, если данных нет

# Функция для получения всех доступных символов на Testnet
def get_available_symbols():
    try:
        exchange_info = client.get_exchange_info()
        symbols = [s['symbol'] for s in exchange_info['symbols']]
        return symbols
    except BinanceAPIException as e:
        print(f"Ошибка при получении списка символов: {e}")
        return []

# Функция для получения точности торговой пары
def get_symbol_precision(symbol):
    symbol_info = safe_binance_request(lambda: client.get_symbol_info(symbol))
    if symbol_info is None:
        return 0.01  # Возвращаем шаг по умолчанию в случае ошибки
    for filt in symbol_info['filters']:
        if filt['filterType'] == 'LOT_SIZE':
            step_size = float(filt['stepSize'])
            return step_size

# Округление количества актива в соответствии с шагом
def round_step_size(quantity, step_size):
    return round(quantity - (quantity % step_size), 8)  # Округляем до 8 знаков

# Функция для получения балансов с Testnet
def get_binance_balances():
    try:
        account_info = safe_binance_request(lambda: client.get_account())
        if account_info is None:
            print("Не удалось получить балансы. Возвращено пустое значение.")
            return {}
        balances = {asset['asset']: float(asset['free']) for asset in account_info['balances'] if float(asset['free']) > 0}
        return balances
    except BinanceAPIException as e:
        print(f"Ошибка при запросе балансов: {e}")
        return {}

# Путь к файлу состояния
STATE_FILE = "environment_state.json"
MODEL_DIR = "./models/"
LOG_DIR = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

class TradingEnv(gym.Env):
    def __init__(self, client, traded_assets=None, max_step_duration=300, max_steps=100):
        super(TradingEnv, self).__init__()
        print("Initializing trading environment...")

        # Инициализация атрибутов
        self.client = client
        self.available_symbols = get_available_symbols()  # Все доступные символы на бирже
        self.traded_assets = traded_assets or self.available_symbols[:10]
        if not self.traded_assets:  # Если список не передан, выбираем первые 10 активов
            self.traded_assets = [symbol.replace("USDT", "") for symbol in self.available_symbols[:10]]
        assert self.traded_assets, "Список активов для торговли не инициализирован."
        self.usdt_balance = 1000.0  # Начальный баланс в USDT
        self.positions = {asset: {'open_price': 0.0, 'quantity': 0.0, 'fees': 0.0} for asset in self.traded_assets}
        # Инициализация портфеля
        self.portfolio = {asset: 0.0 for asset in self.traded_assets}
        self.portfolio['USDT'] = self.usdt_balance
        self.asset_balances = {}
        self.prices = {}  # Цены активов будут обновлены при инициализации

        # Ставка комиссии для Binance
        self.commission_rate = 0.001  # комиссия 0.1%

        self.num_indicators = 37

        # Задаем переменные, которые будут заполнены после динамического выбора активов
        self.observation_space = None
        self.action_space = None
        
        self.base_buy_fraction = 0.1
        self.base_sell_fraction = 0.25
        self.high_volatility_multiplier = 1.5

        self.current_step = self.load_last_step()
        self.max_steps = max_steps
        self.max_step_duration = max_step_duration
        
        # Кэш для индикаторов и цен активов
        self.indicator_cache = {}
        self.cached_prices = {}  # Кэш цен активов
        
        # Параметры для управления обновлениями и шагами
        self.price_update_interval = 5  
        self.balance_update_interval = 5  
        self.last_price_update_step = -self.price_update_interval
        self.last_balance_update_step = -self.balance_update_interval
        
        # Переменные для учета наград и комиссий
        self.Rwd1_previous = 0.0  
        self.commission_previous = {}  
        self.full_report = pd.DataFrame()  
        self.report_history = []
        self.previous_a2 = {}
        self.initial_asset_values = {}

        self.last_action = None
        self.last_reward = 0.0

        self.reward_history = []  
        self.report_interval = 10  
        self.max_report_length = 1000  
        
        self.report_data = {
            "Step": [],
            "Asset": [],
            "Realized Reward": [],
            "Floating Reward": [],
            "Total Reward": [],
            "Portfolio Value": [],
        }

        # Обновляем баланс и цены перед началом
        self.update_balances()  
        self.update_prices()
        self.current_portfolio_value = self.calculate_portfolio_value() 
        print(f"Инициализация цен завершена. Доступные пары: {list(self.prices.keys())}")
        print("Traded Assets (at init):", self.traded_assets)
        print("Portfolio Keys (at init):", list(self.portfolio.keys()))
    
    def get_step_size(self, symbol):
        """
        Получает минимальный шаг изменения количества актива для указанной торговой пары.

        :param symbol: Торговая пара (например, 'BTCUSDT').
        :return: Минимальный шаг изменения количества актива.
        """
        try:
            # Получаем информацию о торговой паре
            exchange_info = self.client.get_symbol_info(symbol)
            if exchange_info:
                filters = exchange_info.get('filters', [])
                for f in filters:
                    if f['filterType'] == 'LOT_SIZE':
                        return float(f['stepSize'])
            raise ValueError(f"Информация о шаге для {symbol} недоступна.")
        except Exception as e:
            logger.error(f"Ошибка при получении step size для {symbol}: {e}")
            raise e

    def update_balances(self):
        """Обновляет балансы всех активов, включая USDT, используя Binance Testnet API."""
        try:
            account_info = safe_binance_request(lambda: self.client.get_account())
            if account_info:
                for balance_info in account_info['balances']:
                    asset = balance_info['asset']
                    if asset in self.traded_assets or asset == "USDT":
                        self.asset_balances[asset] = float(balance_info['free']) + float(balance_info['locked'])
                # Устанавливаем баланс USDT из полученных данных
                self.usdt_balance = self.asset_balances.get("USDT", 0.0)
            else:
                self.usdt_balance = 0.0
                logger.warning("Не удалось обновить балансы. Значение USDT установлено в 0.")
        except Exception as e:
            logger.error(f"Ошибка при обновлении балансов: {e}\n{traceback.format_exc()}")
            self.usdt_balance = 0.0

    def update_prices(self):
        """Метод для обновления цен активов с использованием Binance API."""
        try:
            # Проверяем, инициализирована ли переменная last_price_update_step
            if not hasattr(self, 'last_price_update_step'):
                self.last_price_update_step = 0  # Устанавливаем значение по умолчанию

            # Проверяем, нужно ли обновить цены
            if (self.current_step - self.last_price_update_step) >= self.price_update_interval:
                new_prices = {}
                for asset in self.traded_assets:
                    # Пропускаем USDT, так как его цена фиксирована
                    if asset == "USDT":
                        continue

                    # Формируем символ пары, например BTCUSDT
                    asset_symbol = asset + 'USDT'

                    try:
                        # Получаем цену через Binance API
                        price_data = self.client.get_symbol_ticker(symbol=asset_symbol)
                        if price_data and 'price' in price_data:
                            new_prices[asset_symbol] = float(price_data['price'])
                        else:
                            logger.warning(f"Цена для пары {asset_symbol} не найдена.")
                    except Exception as e:
                        logger.error(f"Ошибка при запросе цены для {asset_symbol}: {e}")

                # Если успешно обновлены цены, сохраняем их и обновляем шаг
                if new_prices:
                    self.prices.update(new_prices)
                    self.last_price_update_step = self.current_step
                    logger.info(f"Цены успешно обновлены: {self.prices}")
                else:
                    logger.warning("Цены не обновлены. Проверьте подключение к Binance API.")
        except Exception as e:
            logger.error(f"Ошибка в методе update_prices: {e}")

    def calculate_portfolio_value(self):
        """Рассчитывает текущую стоимость портфеля, учитывая только активы с ненулевым балансом."""
        total_value = self.usdt_balance
        for asset, quantity in self.asset_balances.items():
            if quantity > 0:
                asset_price_in_usdt = self.prices.get(asset + "USDT", 0.0)
                total_value += quantity * asset_price_in_usdt
        return total_value

    def load_last_step(self):
        """Загружает последний шаг из файла состояния."""
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
                return state.get('step', 0)
        return 0  # Начинаем с шага 0, если файл отсутствует

    def save_state(self):
        """Сохраняет текущее состояние среды."""
        state = {
            'step': self.current_step,
            'usdt_balance': self.usdt_balance,
            'asset_balances': self.asset_balances,
            'prices': self.prices,
        }
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f)

    def calculate_atr(self, data, window):
        """Вспомогательная функция для расчета ATR с проверкой на длину данных и конвертацией в числовой формат."""
        # Конвертация столбцов в числовой формат
        data['high'] = pd.to_numeric(data['high'], errors='coerce')
        data['low'] = pd.to_numeric(data['low'], errors='coerce')
        data['close'] = pd.to_numeric(data['close'], errors='coerce')

        if len(data) < window:
            print(f"Недостаточно данных для расчета ATR. Требуется минимум {window} данных, получено {len(data)}.")
            return np.zeros(len(data))  # Возвращаем массив нулей соответствующей длины
        
        true_range = ta.volatility.AverageTrueRange(
            high=data['high'], 
            low=data['low'], 
            close=data['close'], 
            window=window
        ).average_true_range()
        
        return true_range

    def reset(self, seed=None, options=None):
        # Добавляем seed для совместимости
        super().reset(seed=seed)
        
        logger.info(f"Баланс USDT перед сбросом среды: {self.usdt_balance}")
        print("Resetting environment...")

        # Устанавливаем начальный баланс в USDT и очищаем портфель активов   
        self.usdt_balance = 1000.0
        self.traded_assets = []  # Очищаем список активов
        self.asset_balances = {"USDT": self.usdt_balance}  # Изначально только USDT
        
        # Динамически выбираем активы на основе определенных критериев
        self.select_traded_assets()
            
        # Обновляем пространство действий и наблюдений на основе выбранных активов
        self.num_assets = len(self.traded_assets)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_assets * (self.num_indicators + 10),), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(3 ** self.num_assets)

        # Инициализация `cached_prices` для всех выбранных активов
        self.cached_prices = {asset + "USDT": None for asset in self.traded_assets}

        # Сбрасываем цены и другие параметры
        self.update_balances()
        self.update_prices()
        
        # Конвертация всех данных в числовой формат
        for asset in self.traded_assets:
            if asset in self.prices:
                self.prices[asset + "USDT"] = float(self.prices[asset + "USDT"])

        # Инициализация начальных значений активов и балансов
        self.initial_asset_values = {asset: self.prices.get(asset + "USDT", 0) for asset in self.traded_assets}
        self.previous_a2 = {asset: self.initial_asset_values.get(asset, 0) for asset in self.traded_assets}
        self.previous_quantities = {asset: 0 for asset in self.traded_assets}
        self.commission_previous = {asset: 0.0 for asset in self.traded_assets}
        self.current_a_changes = {asset: 0.0 for asset in self.traded_assets}

        # Рассчитываем стоимость портфеля с обновленным балансом
        self.current_portfolio_value = self.calculate_portfolio_value()

        # Устанавливаем A0 = A1 для первого шага, включая USDT
        for asset in self.traded_assets + ["USDT"]:
            if asset == "USDT":
                self.previous_a2[asset] = self.usdt_balance
            else:
                self.previous_a2[asset] = self.prices.get(asset + "USDT", 0.0)

        obs = self._next_observation()
        return obs, {}

    def select_traded_assets(self):
        """Выбирает активы для торговли на основе заданных показателей, например, объема и капитализации."""
        all_assets = get_available_symbols()
        # Оставляем только пары с "USDT"
        usdt_assets = [symbol for symbol in all_assets if symbol.endswith("USDT")]
    
        # Добавляем логику для отбора активов на основе критериев
        selected_assets = usdt_assets[:10]  # Пример: берем первые 10 активов с "USDT"
        self.traded_assets = [symbol.replace("USDT", "") for symbol in selected_assets]
        logger.info(f"Выбранные активы для торговли: {self.traded_assets}")

    def _calculate_indicators(self, df, asset_symbol, interval='hourly'):
        """Рассчитывает индикаторы для заданного интервала и сохраняет их в кэш."""
        try:
            # Рассчет индикаторов только если данных достаточно
            if len(df) >= 14:
                suffix = interval.capitalize()  # Добавляем суффикс для уникальности
                df['RSI'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
                macd = ta.trend.MACD(close=df['close'])
                df['MACD'] = macd.macd()
                df['Signal_Line'] = macd.macd_signal()
                df['SMA_20'] = ta.trend.SMAIndicator(close=df['close'], window=20).sma_indicator()
                df['EMA_20'] = ta.trend.EMAIndicator(close=df['close'], window=20).ema_indicator()
                df['ATR'] = self.calculate_atr(df, 14)
                df['ROC'] = ta.momentum.ROCIndicator(close=df['close'], window=14).roc()

                # Сохраняем последние значения индикаторов в кэш
                self.indicator_cache[asset_symbol] = self.indicator_cache.get(asset_symbol, {})
                self.indicator_cache[asset_symbol][f'{suffix}_RSI'] = df['RSI'].iloc[-1]
                self.indicator_cache[asset_symbol][f'{suffix}_MACD'] = df['MACD'].iloc[-1]
                self.indicator_cache[asset_symbol][f'{suffix}_Signal_Line'] = df['Signal_Line'].iloc[-1]
                self.indicator_cache[asset_symbol][f'{suffix}_SMA_20'] = df['SMA_20'].iloc[-1]
                self.indicator_cache[asset_symbol][f'{suffix}_EMA_20'] = df['EMA_20'].iloc[-1]
                self.indicator_cache[asset_symbol][f'{suffix}_ATR'] = df['ATR'].iloc[-1]
                self.indicator_cache[asset_symbol][f'{suffix}_ROC'] = df['ROC'].iloc[-1]
                self.indicator_cache[asset_symbol][f'{suffix}_volume'] = df['volume'].iloc[-1]
            else:
                # Заполняем нулями при недостатке данных
                logger.warning(f"Недостаточно данных для расчета индикаторов для {asset_symbol} на интервале {interval}.")
                self.indicator_cache[asset_symbol] = self.indicator_cache.get(asset_symbol, {})
                for key in [f'{interval.capitalize()}_RSI', f'{interval.capitalize()}_MACD', f'{interval.capitalize()}_Signal_Line',
                            f'{interval.capitalize()}_SMA_20', f'{interval.capitalize()}_EMA_20', f'{interval.capitalize()}_ATR', 
                            f'{interval.capitalize()}_ROC', f'{interval.capitalize()}_volume']:
                    self.indicator_cache[asset_symbol][key] = 0.0
        except Exception as e:
            logger.error(f"Ошибка при расчете индикаторов для {asset_symbol} на интервале {interval}: {e}\n{traceback.format_exc()}")
            self.indicator_cache[asset_symbol] = self.indicator_cache.get(asset_symbol, {})
            for key in [f'{interval.capitalize()}_RSI', f'{interval.capitalize()}_MACD', f'{interval.capitalize()}_Signal_Line',
                        f'{interval.capitalize()}_SMA_20', f'{interval.capitalize()}_EMA_20', f'{interval.capitalize()}_ATR', 
                        f'{interval.capitalize()}_ROC', f'{interval.capitalize()}_volume']:
                self.indicator_cache[asset_symbol][key] = 0.0

    def _next_observation(self):
        """Обновляет наблюдения, включая цену актива, баланс и данные глубины рынка."""
        obs = []
        
        for asset in self.traded_assets[:self.num_assets]:
            asset_symbol = asset + 'USDT'
            price = self.prices.get(asset + "USDT", 0.0)
            
            # Добавляем базовые данные: цена актива, баланс актива, текущий баланс USDT
            obs.extend([price, self.asset_balances.get(asset + "USDT", 0), self.usdt_balance])

            # Обновляем индикаторы, если цена изменилась с прошлого шага
            if self.cached_prices[asset_symbol] != price:
                hourly_candles = get_hourly_candles(asset_symbol, limit=168)
                min15_candles = get_15min_candles(asset_symbol, limit=96)
                daily_candles = get_daily_candles(asset_symbol, limit=365)

                # Проверяем наличие данных и рассчитываем индикаторы
                if hourly_candles:
                    hourly_df = pd.DataFrame(hourly_candles, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
                        'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 
                        'taker_buy_quote_asset_volume', 'ignore'
                    ])
                    hourly_df['high'] = pd.to_numeric(hourly_df['high'], errors='coerce')
                    hourly_df['low'] = pd.to_numeric(hourly_df['low'], errors='coerce')
                    hourly_df['close'] = pd.to_numeric(hourly_df['close'], errors='coerce')
                    hourly_df['volume'] = pd.to_numeric(hourly_df['volume'], errors='coerce')
                    self._calculate_indicators(hourly_df, asset_symbol)
                    self.cached_prices[asset_symbol] = price
                else:
                    logger.warning(f"Недостаточно данных для часовых свечей {asset_symbol}")

                if min15_candles:
                    min15_df = pd.DataFrame(min15_candles, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
                        'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 
                        'taker_buy_quote_asset_volume', 'ignore'
                    ])
                    min15_df['high'] = pd.to_numeric(min15_df['high'], errors='coerce')
                    min15_df['low'] = pd.to_numeric(min15_df['low'], errors='coerce')
                    min15_df['close'] = pd.to_numeric(min15_df['close'], errors='coerce')
                    min15_df['volume'] = pd.to_numeric(min15_df['volume'], errors='coerce')
                    self._calculate_indicators(min15_df, asset_symbol, interval='15min')
                else:
                    logger.warning(f"Недостаточно данных для 15-минутных свечей {asset_symbol}")

                if daily_candles:
                    daily_df = pd.DataFrame(daily_candles, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
                        'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 
                        'taker_buy_quote_asset_volume', 'ignore'
                    ])
                    daily_df['high'] = pd.to_numeric(daily_df['high'], errors='coerce')
                    daily_df['low'] = pd.to_numeric(daily_df['low'], errors='coerce')
                    daily_df['close'] = pd.to_numeric(daily_df['close'], errors='coerce')
                    daily_df['volume'] = pd.to_numeric(daily_df['volume'], errors='coerce')
                    self._calculate_indicators(daily_df, asset_symbol, interval='Daily')
                else:
                    logger.warning(f"Недостаточно данных для дневных свечей {asset_symbol}")
            
            # Получаем индикаторы из кэша или ставим значения по умолчанию
            obs.extend([
                self.indicator_cache[asset_symbol].get('Hourly_RSI', 0.0),
                self.indicator_cache[asset_symbol].get('Hourly_MACD', 0.0),
                self.indicator_cache[asset_symbol].get('Hourly_Signal_Line', 0.0),
                self.indicator_cache[asset_symbol].get('Hourly_SMA_20', 0.0),
                self.indicator_cache[asset_symbol].get('Hourly_EMA_20', 0.0),
                self.indicator_cache[asset_symbol].get('Hourly_ATR', 0.0),
                self.indicator_cache[asset_symbol].get('Hourly_ROC', 0.0),
                self.indicator_cache[asset_symbol].get('Hourly_volume', 0.0),
                self.indicator_cache[asset_symbol].get('Min15_RSI', 0.0),
                self.indicator_cache[asset_symbol].get('Min15_MACD', 0.0),
                self.indicator_cache[asset_symbol].get('Min15_Signal_Line', 0.0),
                self.indicator_cache[asset_symbol].get('Min15_SMA_20', 0.0),
                self.indicator_cache[asset_symbol].get('Min15_EMA_20', 0.0),
                self.indicator_cache[asset_symbol].get('Min15_ATR', 0.0),
                self.indicator_cache[asset_symbol].get('Min15_ROC', 0.0),
                self.indicator_cache[asset_symbol].get('Min15_volume', 0.0),
                self.indicator_cache[asset_symbol].get('Daily_RSI', 0.0),
                self.indicator_cache[asset_symbol].get('Daily_MACD', 0.0),
                self.indicator_cache[asset_symbol].get('Daily_Signal_Line', 0.0),
                self.indicator_cache[asset_symbol].get('Daily_SMA_20', 0.0),
                self.indicator_cache[asset_symbol].get('Daily_EMA_20', 0.0),
                self.indicator_cache[asset_symbol].get('Daily_ATR', 0.0),
                self.indicator_cache[asset_symbol].get('Daily_ROC', 0.0),
                self.indicator_cache[asset_symbol].get('Daily_volume', 0.0),
            ])

            # Добавляем данные глубины рынка
            order_book = safe_binance_request(lambda: client.get_order_book(symbol=asset_symbol, limit=10))
            if order_book:
                # Добавляем уровни бидов и асков
                bids = [float(bid[0]) for bid in order_book['bids'][:5]]  # Первые 5 бидов по цене
                asks = [float(ask[0]) for ask in order_book['asks'][:5]]  # Первые 5 асков по цене
                obs.extend(bids + asks)
            else:
                # Если данные недоступны, заполняем нулями
                obs.extend([0.0] * 10)
        
        # Проверяем размер массива наблюдений
        expected_size = self.num_assets * (self.num_indicators + 10)  # +10 для глубины рынка
        if len(obs) < expected_size:
            obs.extend([0.0] * (expected_size - len(obs)))

        obs = np.clip(obs, -np.inf, np.inf)
        obs = np.nan_to_num(obs, nan=0.0)  # Заменяем NaN на 0
        logger.debug(f"Размер массива наблюдений: {len(obs)}, ожидаемый размер: {expected_size}")

        assert len(obs) == expected_size, (
            f"Размер массива наблюдений: {len(obs)}, ожидаемый размер: {expected_size}"
        )

        obs = np.array(obs, dtype=np.float32)  # Преобразуем в массив NumPy
        print(f"Размер наблюдения: {obs.shape}, Тип: {type(obs)}")  # Временный вывод информации

        return obs

    def get_assets_on_balance(self):
        balances = get_binance_balances()
        available_symbols_set = set(self.available_symbols)
        assets_on_balance = [asset for asset in self.traded_assets if asset in balances and asset + 'USDT' in available_symbols_set]
        return assets_on_balance

    def get_prices(self, asset_symbol):
        """Запрашивает цену для указанного актива, если символ доступен."""
        if asset_symbol not in self.available_symbols:
            print(f"Символ {asset_symbol} недоступен на Testnet.")
            return 0.0
        ticker = safe_binance_request(lambda: client.get_symbol_ticker(symbol=asset_symbol))
        return float(ticker['price']) if ticker else 0.0

    def step(self, actions):
        """
        Выполняет шаг симуляции торговой среды, обновляет состояние портфеля и рассчитывает награду.
        """
        # Если actions скаляр, преобразуем его в список действий
        if not isinstance(actions, (list, np.ndarray)):
            actions = [actions]

        # Убедитесь, что длина списка соответствует количеству активов
        if len(actions) != self.num_assets:
            raise ValueError(f"Длина actions ({len(actions)}) не соответствует количеству активов ({self.num_assets}).")
    
        self.update_prices()  # Обновляем текущие цены
        self.last_action = actions
        total_reward = 0.0
        observation = self._next_observation()

        for idx, asset in enumerate(self.traded_assets):  # Учитываем все активы
            action = actions[idx]
            price = self.prices.get(f"{asset + "USDT"}USDT", 0.0)

            # Покупка
            if action == 1:
                buy_quantity = self.calculate_purchase_quantity(asset)
                if buy_quantity > 0:
                    self._buy(asset)

            # Продажа
            elif action == 2:
                sell_quantity = self.portfolio.get(asset, 0)  # Используем get() для безопасного доступа
                if sell_quantity > 0:
                    self._sell(asset)

            # Рассчитываем текущую награду
            floating_reward = self.calculate_floating_reward(asset)
            total_reward += floating_reward

        self.last_reward = total_reward
        self.current_step += 1

        terminated = self.current_step >= self.max_steps
        truncated = False

        return observation, total_reward, terminated, truncated, {}

    def calculate_rsi(df, period=14):
        """Расчет RSI для заданного периода."""
        df['RSI'] = ta.momentum.RSIIndicator(close=df['close'], window=period).rsi()
        return df['RSI'].iloc[-1]  # Возвращаем последнее значение RSI

    def select_top_assets(self):
        """
        Выбирает топ-10 активов для торговли, используя RSI на разных таймфреймах и объемы.
        """
        all_assets = get_available_symbols()
        selected_assets = []

        for symbol in all_assets:
            # Пропускаем, если это не пара с USDT
            if not symbol.endswith("USDT"):
                continue

            # Получаем свечи и вычисляем RSI на разных таймфреймах
            try:
                rsi_1m = calculate_rsi(pd.DataFrame(get_1min_candles(symbol)))
                rsi_15m = calculate_rsi(pd.DataFrame(get_15min_candles(symbol)))
                rsi_1h = calculate_rsi(pd.DataFrame(get_hourly_candles(symbol)))
                rsi_1d = calculate_rsi(pd.DataFrame(get_daily_candles(symbol)))

                # Фильтруем активы на основе значений RSI и других условий
                if (30 < rsi_1m < 70) and (30 < rsi_15m < 70) and (30 < rsi_1h < 70) and (30 < rsi_1d < 70):
                    selected_assets.append(symbol)
            except Exception as e:
                logger.warning(f"Не удалось получить данные RSI для {symbol}: {e}")

        # Оставляем первые 10 активов по объему или другим критериям
        return selected_assets[:10]

    def save_report_if_needed(self):
        """Сохраняет отчет, если достигнут установленный интервал."""
        if self.current_step % self.report_interval == 0 and self.report_data["Step"]:
            report_df = pd.DataFrame(self.report_data)
            self.save_report(report_df, format_type="json")

            logger.info(f"\nОтчет по шагу {self.current_step}:\n{report_df}")
            self.report_data = {key: [] for key in self.report_data}  # Очистка после сохранения

            if len(self.report_history) > self.max_report_length:
                self.report_history = []
                logger.info("Сброс истории отчетов из-за превышения максимального количества записей.")

    def save_report(self, report_df, format_type="json"):
        """Сохраняет отчет по указанному формату (JSON, текст или CSV)."""
        try:
            if format_type == "json":
                # Сохраняем данные в формате JSON
                report_json = report_df.to_dict(orient="records")
                self.report_history.append(report_json)
                with open("report_history.json", "w") as file:
                    # logger.info(f"Записываем данные в report_history.json: {self.report_history}")
                    json.dump(self.report_history, file, indent=4)
                logger.info("Отчёт сохранен в формате JSON: report_history.json")
        
            elif format_type == "text":
                # Сохраняем данные в текстовом формате
                report_text = report_df.to_string(index=False)
                with open("report_history.txt", "a") as file:
                    file.write(report_text + "\n\n")
                logger.info("Отчёт сохранен в текстовом формате: report_history.txt")
            
            elif format_type == "csv":
                # Сохраняем данные в формате CSV для удобного анализа
                report_df.to_csv("report_history.csv", mode='a', index=False, header=not os.path.exists("report_history.csv"))
                logger.info("Отчёт сохранен в формате CSV: report_history.csv")
        
            else:
                logger.warning("Неизвестный формат. Пожалуйста, выберите 'json', 'text' или 'csv'.")
                
        except Exception as e:
            logger.error(f"Ошибка при сохранении отчёта: {e}\n{traceback.format_exc()}")

    def update_usdt_balance(self, action, quantity, price, commission):
        """
        Обновляет баланс USDT в зависимости от действия.
        
        :param action: Тип действия (0 - покупка, 1 - продажа).
        :param quantity: Количество актива для покупки или продажи.
        :param price: Текущая цена актива.
        :param commission: Комиссия за транзакцию.
        """
        if action == 0:  # Покупка
            self.usdt_balance -= (quantity * price) + commission
        elif action == 1:  # Продажа
            self.usdt_balance += (quantity * price) - commission

    def calculate_purchase_quantity(self, asset):
        """
        Рассчитать количество покупки для указанного актива.
        """
        # Проверяем оба возможных формата ключей
        pair_usdt = asset + 'USDT'
        pair_slash = asset + '/USDT'
        
        if pair_usdt in self.prices:
            pair = pair_usdt
        elif pair_slash in self.prices:
            pair = pair_slash
        else:
            raise KeyError(
                f"Ключ '{pair_usdt}' или '{pair_slash}' отсутствует в self.prices. "
                f"Доступные ключи: {list(self.prices.keys())}"
            )
        
        step_size = self.get_step_size(pair)
        buy_amount = self.usdt_balance * 0.1  # Используем 10% от текущего баланса USDT
        quantity = round_step_size(buy_amount / self.prices[pair], step_size)
        return quantity

    def calculate_sale_quantity(self, asset):
        """Определяет объем продажи без жестких ограничений."""
        step_size = get_symbol_precision(asset + 'USDT')
        sell_amount = self.asset_balances.get(asset, 0) * self.base_sell_fraction
        quantity = round_step_size(sell_amount, step_size)
        return quantity

    def export_full_report(self, format_type="text"):
        """
        Экспортирует накопленный отчет self.full_report в указанный формат.
        Поддерживаются форматы: "text", "markdown", "json".
        """
        if self.full_report.empty:
            print("Отчет пуст. Нет данных для экспорта.")
            return
        
        # Вызываем save_report с накопленным отчетом
        self.save_report(self.full_report, format_type=format_type)

    def _buy(self, asset):
        if self.asset_balances.get("USDT", 0) > 0:
            buy_amount = self.asset_balances["USDT"] * 0.2
            step_size = get_symbol_precision(asset + 'USDT')
            quantity = round_step_size(buy_amount / self.prices[asset + 'USDT'], step_size)
            cost = quantity * self.prices[asset + 'USDT']
            if cost < 10:
                return None
            order_response = safe_binance_request(lambda: client.order_market_buy(symbol=asset + 'USDT', quantity=quantity))
            if order_response:
                self.update_balances()
            return order_response
        return None

    def _sell(self, asset):
        if self.asset_balances[asset] > 0:
            step_size = get_symbol_precision(asset + 'USDT')
            quantity = round_step_size(self.asset_balances[asset] * 0.5, step_size)
            revenue = quantity * self.prices[asset + 'USDT']
            if revenue < 10:
                return None
            order_response = safe_binance_request(lambda: client.order_market_sell(symbol=asset + 'USDT', quantity=quantity))
            if order_response:
                self.update_balances()
            return order_response
        return None

    def _hold(self):
        logger.info("Действие удержания позиции выполнено.")

    def encode_action(self, actions):
        return sum([action * (3 ** i) for i, action in enumerate(actions)])

    def decode_action(self, action):
        actions = []
        for _ in range(self.num_assets):
            actions.append(action % 3)
            action //= 3
        return actions

    def calculate_floating_reward(self, asset):
        """
        Рассчитывает плавающую прибыль/убыток для указанного актива.

        :param asset: Название актива, например 'BTC'.
        :return: Плавающая прибыль/убыток.
        """
        try:
            pair = f'{asset}USDT'
            if pair not in self.prices or self.portfolio.get(asset, 0.0) <= 0:
                return 0.0  # Если данных нет или актива нет в портфеле, плавающий доход = 0

            # Цена покупки и количество актива
            position = self.positions.get(asset, {})
            open_price = position.get('open_price', 0.0)
            quantity = position.get('quantity', 0.0)
            fees = position.get('fees', 0.0)

            # Текущая цена актива
            current_price = self.prices[pair]

            # Плавающая прибыль/убыток
            floating_reward = (current_price - open_price) * quantity - fees
            return floating_reward
        except Exception as e:
            logger.error(f"Ошибка при расчете плавающего дохода для {asset}: {e}")
            return 0.0

    def save_portfolio_state(env):
        """Saves the portfolio state and current step to a JSON file."""
        portfolio_state = {
            'usdt_balance': env.usdt_balance,
            'asset_balances': env.asset_balances,
            'current_step': env.current_step,
            'portfolio_value': env.current_portfolio_value,
        }
        with open("portfolio_state.json", "w") as f:
            json.dump(portfolio_state, f)
        print("Portfolio state saved.")

    # Consolidated load_portfolio_state function
    def load_portfolio_state(env):
        """Loads the portfolio state and step from a JSON file if available."""
        if os.path.exists("portfolio_state.json"):
            with open("portfolio_state.json", "r") as f:
                portfolio_state = json.load(f)
                env.usdt_balance = portfolio_state.get("usdt_balance", 10000.0)
                env.asset_balances = portfolio_state.get("asset_balances", {})
                env.current_step = portfolio_state.get("current_step", 0)
                env.current_portfolio_value = portfolio_state.get("portfolio_value", 0)
            print("Portfolio state loaded.")
        else:
            print("Portfolio state file not found. Starting with default values.")
            env.usdt_balance = 10000.0
            env.asset_balances = {}

    # Combined print_report method
    def print_report(env):
        """Generates and displays a consolidated report of trading performance."""
        report_df = pd.DataFrame(env.report_data)
        totals = report_df.sum(numeric_only=True)
        totals["Asset"] = "Total"
        totals["Step"] = "Overall"
        report_df = pd.concat([report_df, pd.DataFrame([totals])], ignore_index=True)
        print(f"\nStep {env.current_step} Report:")
        print(report_df)

# Flask сервер
def run_flask():
    print("Запуск Flask сервера...")
    app.run(port=5000, debug=False)

# Путь для сохранения состояния портфеля
portfolio_state_path = "portfolio_state.json"
model_save_path = "dqn_trading_bot_checkpoint"

def initialize_trading_env():
    """
    Создает и инициализирует торговую среду.
    """
    global env
    env = TradingEnv(client=client)
        
    env.reset()  # Первичная настройка пространства через reset()
        
    # Проверка среды после инициализации пространства
    check_env(env)
    env.load_portfolio_state()  # Загружаем состояние портфеля

# Задаем гиперпараметры модели DQN
learning_rate = 0.0001  # Скорость обучения
buffer_size = 100000    # Размер буфера опыта
batch_size = 64         # Размер пакета для обучения

class TensorboardCallback(BaseCallback):
    def __init__(self, log_interval=10, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_start_time = time.time()
        self.step_durations = []
        self.losses = []
        self.log_interval = log_interval
        self.episode_count = 0

    def _on_step(self) -> bool:
        start_time = time.time()  # Начало отсчета времени

        # Логируем в TensorBoard
        self.logger.record('balance_usdt', env.usdt_balance)
        self.logger.record('portfolio_value_usdt', env.current_portfolio_value)

        # Логируем награду и действие
        action = env.get_last_action()
        reward = env.get_last_reward()
        self.logger.record("reward", reward)

        # Логирование потерь, если они доступны
        if 'loss' in self.locals:
            loss = self.locals['loss']
            self.losses.append(loss)
            self.logger.record('loss', loss)

        # Печать метрик каждые `log_interval` шагов
        if self.n_calls % self.log_interval == 0:
            mean_reward_per_step = np.mean(self.episode_rewards) if self.episode_rewards else 0
            ep_len_mean = np.mean([len(r) for r in self.episode_rewards]) if self.episode_rewards else 0
            ep_rew_mean = np.mean([sum(r) for r in self.episode_rewards]) if self.episode_rewards else 0
            exploration_rate = self.locals.get('exploration_rate', 0.99)

            print("-----------------------------------")
            print(f"| balance_usdt         | {env.usdt_balance:.2e} |")
            print(f"| mean_reward_per_step | {mean_reward_per_step:.3f} |")
            print(f"| portfolio_value_usdt | {env.current_portfolio_value:.2e} |")
            print(f"| rollout/             |          |")
            print(f"|    ep_len_mean       | {ep_len_mean:.2f} |")
            print(f"|    ep_rew_mean       | {ep_rew_mean:.3f} |")
            print(f"|    exploration_rate  | {exploration_rate:.3f} |")
            print(f"| step_duration        | {np.mean(self.step_durations):.2e} |")
            print(f"| time/                |          |")
            print(f"|    episodes          | {self.episode_count} |")
            print(f"|    fps               | {self.locals.get('fps', 0)} |")
            print(f"|    time_elapsed      | {int(time.time() - self.episode_start_time)} |")
            print(f"|    total_timesteps   | {self.n_calls} |")
            print("-----------------------------------")

        # Вывод отчета в терминал каждые 100 шагов
        if self.n_calls % 100 == 0:
            print(f"Шаг {self.n_calls}: Баланс USDT: {env.usdt_balance}, "
                  f"Стоимость портфеля: {env.current_portfolio_value}, "
                  f"Накопленная награда: {sum(self.episode_rewards)}")

        # Проверка на завершение эпизода и расчет метрик
        if env.current_step >= 100:
            self.episode_rewards.append(self.episode_rewards)
            self.episode_count += 1
            self.episode_rewards.clear()  # Очистка наград для следующего эпизода
            self.episode_start_time = time.time()

        # Записываем длительность шага
        step_duration = time.time() - start_time
        self.step_durations.append(step_duration)
        self.logger.record('step_duration', step_duration)

        return True

if __name__ == '__main__':
    # 1. Создаем и инициализируем среду торговли
    initialize_trading_env()
    
    # Настройки и параметры для модели DQN
    policy_kwargs = dict(
        features_extractor_class=CustomCNNFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128)
    )
    
    model = DQN("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

    # 2. Определяем папку для логов, включая текущую дату и время для уникальности
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(log_dir, exist_ok=True)  # Создаем папку, если она не существует
    
    # 3. Гиперпараметры для модели DQN
    learning_rate = 0.00001
    buffer_size = 100000
    batch_size = 64
    
    # 4. Путь для автосохранения модели и состояния портфеля
    model_save_path = "dqn_trading_bot_checkpoint" 
    
    # 5. Загружаем модель, если она существует, иначе создаем новую
    if os.path.exists(f"{model_save_path}.zip"):
        model = DQN.load(model_save_path, env=env)
        print("Модель загружена.")
    else:
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            verbose=1,
            tensorboard_log=log_dir,  # Устанавливаем путь для логов TensorBoard
            exploration_fraction=0.7,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.1,
        )
    
    # 6. Запуск обучения модели с периодическим сохранением
    total_steps = 100000
    for step in range(0, total_steps, 1000):
        model.learn(total_timesteps=1000, reset_num_timesteps=False)  # Обучение на 1000 шагов
        model.save(model_save_path)  # Сохранение модели каждые 1000 шагов
        save_portfolio_state(env)    # Сохранение состояния портфеля

