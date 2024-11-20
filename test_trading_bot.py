import numpy as np
from trading_bot_v41 import TradingEnv, client

def test_trading_env():
    # 1. Инициализация среды
    print("\n[ТЕСТ] Инициализация среды")
    env = TradingEnv(client=client)
    obs, _ = env.reset()
    assert obs is not None, "Ошибка: Наблюдение не инициализировано."
    print("Среда успешно инициализирована.")

    # 2. Проверка работы с данными
    print("\n[ТЕСТ] Работа с данными")
    for asset in env.traded_assets:
        print(f"Проверка данных для актива: {asset}")
        assert asset + 'USDT' in env.prices, f"Ошибка: Цена для {asset}USDT не найдена."
        assert len(env.cached_prices) > 0, "Ошибка: Кэш цен не заполнен."
        print(f"Данные для {asset} проверены.")

    # 3. Тестирование действий
    print("\n[ТЕСТ] Выполнение действий")
    action = np.zeros(len(env.traded_assets), dtype=int)  # Все действия - удержание
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs is not None, "Ошибка: Наблюдение после шага пустое."
    assert reward == 0, "Ошибка: Награда при удержании должна быть 0."
    print("Действие 'удержание' выполнено корректно.")

    # 4. Проверка покупок и продаж
    print("\n[ТЕСТ] Покупка и продажа")
    action[0] = 1  # Покупка первого актива
    obs, reward, terminated, truncated, info = env.step(action)
    assert reward <= 0, "Ошибка: Награда должна учитывать комиссию при покупке."
    action[0] = 2  # Продажа первого актива
    obs, reward, terminated, truncated, info = env.step(action)
    assert reward >= 0, "Ошибка: Награда должна быть положительной при продаже с прибылью."
    print("Покупка и продажа протестированы.")

    # 5. Проверка завершения эпизода
    print("\n[ТЕСТ] Завершение эпизода")
    for _ in range(env.max_steps - env.current_step):
        action = np.zeros(len(env.traded_assets), dtype=int)  # Удержание
        obs, reward, terminated, truncated, info = env.step(action)
    assert terminated, "Ошибка: Эпизод должен завершиться при достижении max_steps."
    print("Завершение эпизода протестировано.")

    # 6. Проверка логирования и отчетов
    print("\n[ТЕСТ] Логирование и отчеты")
    env.print_report()
    env.save_report_if_needed()
    print("Логирование и отчеты успешно протестированы.")

    # 7. Тест взаимодействия с API Binance
    print("\n[ТЕСТ] Взаимодействие с API Binance")
    available_symbols = env.available_symbols
    assert len(available_symbols) > 0, "Ошибка: Binance API не вернул символы."
    print("API Binance успешно протестировано.")

    print("\n[ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО]")

if __name__ == "__main__":
    test_trading_env()
