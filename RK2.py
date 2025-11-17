import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np
from collections import Counter

# Конфигурация
BASE_URL = "http://193.233.171.205:5000"
LOGIN = "analyst_is"
CODE = "HjK89sTu01Op"


def make_authenticated_request(endpoint, params=None):
    """
    Выполняет аутентифицированный GET запрос к API
    """
    url = f"{BASE_URL}{endpoint}"
    params = params or {}
    params['login'] = LOGIN
    params['code'] = CODE

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Ошибка запроса к {endpoint}: {e}")
        return None
    except json.JSONDecodeError:
        print(f"Ошибка: Неверный JSON ответ от {endpoint}")
        return None


def get_tickets():
    """Получить список всех тикетов"""
    print(make_authenticated_request('/api/v1/tickets'))
    return make_authenticated_request('/api/v1/tickets') or []


def get_ticket_details(ticket_id):
    """Получить детальную информацию по тикету"""
    return make_authenticated_request(f'/api/v1/tickets/{ticket_id}')


def get_comparison_data():
    """Получить данные для сравнения производительности"""
    return make_authenticated_request('/api/v1/comparison')


def get_metrics_data():
    """Получить метрики"""
    return make_authenticated_request('/api/v1/metrics')


def load_and_prepare_data():
    """
    Загрузка и подготовка данных для анализа
    """
    print("Загрузка данных из API...")

    # Получаем список тикетов
    tickets = get_tickets()
    if not tickets:
        print("Нет данных для анализа")
        return None

    print(f"Получено {len(tickets)} тикетов")

    # Если API сразу возвращает детальную информацию, используем как есть
    # Если нет - запрашиваем детали по каждому тикету
    if isinstance(tickets, list) and len(tickets) > 0 and 'ticket_id' in tickets[0]:
        # Уже есть детальная информация
        detailed_tickets = tickets
        print("Используются данные из основного запроса тикетов")
    else:
        # Нужно запрашивать детали
        print("Запрашиваем детальную информацию по тикетам...")
        detailed_tickets = []
        for i, ticket in enumerate(tickets):
            if i % 20 == 0:  # Прогресс каждые 20 тикетов
                print(f"Обработано {i}/{len(tickets)} тикетов")

            ticket_id = ticket.get('id') or ticket.get('ticket_id')
            if ticket_id:
                ticket_details = get_ticket_details(ticket_id)
                if ticket_details:
                    detailed_tickets.append(ticket_details)

    if not detailed_tickets:
        print("Нет данных после обработки")
        return None

    # Создаем DataFrame
    df = pd.DataFrame(detailed_tickets)

    print(f"Успешно загружено {len(df)} тикетов для анализа")

    # Преобразование дат
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'])

    if 'closed_at' in df.columns:
        # Заменяем None на текущее время для открытых тикетов
        df['closed_at'] = pd.to_datetime(df['closed_at'])
        df['closed_at_filled'] = df['closed_at'].fillna(pd.Timestamp.now())

    # Расчет времени решения (если есть данные о закрытии)
    if 'closed_at_filled' in df.columns and 'created_at' in df.columns:
        df['resolution_time_hours'] = (
                                              df['closed_at_filled'] - df['created_at']
                                      ).dt.total_seconds() / 3600

    # Для незакрытых тикетов время решения = NaN
    if 'closed_at' in df.columns and 'created_at' in df.columns:
        mask = df['closed_at'].notna()
        df.loc[mask, 'resolution_time_hours'] = (
                                                        df.loc[mask, 'closed_at'] - df.loc[mask, 'created_at']
                                                ).dt.total_seconds() / 3600

    print("Данные подготовлены для анализа")
    return df


def setup_plot_style():
    """Настройка стиля графиков"""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10


def plot_ticket_trends(df):
    """Линейный график: динамика создания тикетов по дням за последние 30 дней"""
    if 'created_at' not in df.columns:
        print("Нет данных о дате создания тикетов")
        return

    plt.figure(figsize=(14, 7))

    # Фильтрация данных за последние 30 дней
    cutoff_date = datetime.now() - timedelta(days=30)
    recent_tickets = df[df['created_at'] >= cutoff_date].copy()

    if recent_tickets.empty:
        print("Нет данных за последние 30 дней")
        return

    # Группировка по дням
    recent_tickets['date'] = recent_tickets['created_at'].dt.date
    daily_trends = recent_tickets.groupby('date').size()

    plt.plot(daily_trends.index, daily_trends.values, marker='o', linewidth=2,
             markersize=6, color='#2E86AB')
    plt.title('Динамика создания тикетов по дням (последние 30 дней)',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Дата', fontsize=12)
    plt.ylabel('Количество тикетов', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Статистика
    print("СТАТИСТИКА ДИНАМИКИ ТИКЕТОВ:")
    print(f"   • Среднее количество тикетов в день: {daily_trends.mean():.1f}")
    print(f"   • Максимальное количество за день: {daily_trends.max()}")
    print(f"   • Минимальное количество за день: {daily_trends.min()}")
    print(f"   • Всего тикетов за период: {daily_trends.sum()}")
    print()


def plot_hourly_distribution(df):
    """Столбчатая диаграмма: распределение тикетов по часам суток"""
    if 'created_at' not in df.columns:
        print("Нет данных о дате создания тикетов")
        return

    plt.figure(figsize=(14, 7))

    df['hour'] = df['created_at'].dt.hour
    hourly_dist = df.groupby('hour').size()

    colors = plt.cm.viridis(np.linspace(0, 1, len(hourly_dist)))
    bars = plt.bar(hourly_dist.index, hourly_dist.values, color=colors, alpha=0.7, edgecolor='black')

    plt.title('Распределение тикетов по часам суток',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Час суток', fontsize=12)
    plt.ylabel('Количество тикетов', fontsize=12)
    plt.xticks(range(0, 24))
    plt.grid(True, alpha=0.3, axis='y')

    # Добавляем значения на столбцы
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{int(height)}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    peak_hour = hourly_dist.idxmax()
    print(" РАСПРЕДЕЛЕНИЕ ПО ЧАСАМ:")
    print(f"   • Пиковый час создания тикетов: {peak_hour}:00")
    print(f"   • Количество тикетов в пиковый час: {hourly_dist.max()}")
    print()


def plot_heatmap(df):
    """Тепловая карта: активность по дням недели и часам"""
    if 'created_at' not in df.columns:
        print("Нет данных о дате создания тикетов")
        return

    plt.figure(figsize=(16, 8))

    # Создаем данные для тепловой карты
    df['hour'] = df['created_at'].dt.hour
    df['day_of_week'] = df['created_at'].dt.day_name()

    # Создаем сводную таблицу
    heatmap_data = df.pivot_table(
        index='day_of_week',
        columns='hour',
        values='ticket_id' if 'ticket_id' in df.columns else df.columns[0],
        aggfunc='count',
        fill_value=0
    )

    # Упорядочиваем дни недели
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(days_order)

    sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlOrRd',
                cbar_kws={'label': 'Количество тикетов'}, linewidths=0.5)
    plt.title('Активность тикетов: Дни недели vs Часы',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Час суток', fontsize=12)
    plt.ylabel('День недели', fontsize=12)
    plt.tight_layout()
    plt.show()

    # Находим самый активный период
    max_day = heatmap_data.sum(axis=1).idxmax()
    max_hour = heatmap_data.sum(axis=0).idxmax()
    print("ТЕПЛОВАЯ КАРТА АКТИВНОСТИ:")
    print(f"   • Самый активный день: {max_day}")
    print(f"   • Самый активный час: {max_hour}:00")
    print()


def plot_category_distribution(df):
    """Круговая диаграмма: распределение тикетов по категориям проблем"""
    category_column = None
    for col in ['category_name', 'category', 'category_id']:
        if col in df.columns:
            category_column = col
            break

    if not category_column:
        print("В данных отсутствует информация о категориях")
        return

    plt.figure(figsize=(12, 10))

    category_dist = df[category_column].value_counts()

    # Создаем круговую диаграмму
    colors = plt.cm.Set3(np.linspace(0, 1, len(category_dist)))
    wedges, texts, autotexts = plt.pie(category_dist.values,
                                       labels=category_dist.index,
                                       autopct=lambda p: f'{p:.1f}%' if p > 3 else '',
                                       startangle=90,
                                       colors=colors,
                                       shadow=True,
                                       textprops={'fontsize': 10})

    # Улучшаем отображение текста
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')

    plt.title('Распределение тикетов по категориям проблем',
              fontsize=16, fontweight='bold', pad=20)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    print(" РАСПРЕДЕЛЕНИЕ ПО КАТЕГОРИЯМ:")
    for category, count in category_dist.head().items():
        percentage = (count / len(df)) * 100
        print(f"   • {category}: {count} тикетов ({percentage:.1f}%)")
    print()


def plot_resolution_time_by_category(df):
    """Горизонтальная столбчатая диаграмма: среднее время решения по категориям"""
    if 'resolution_time_hours' not in df.columns:
        print("Нет данных о времени решения тикетов")
        return

    category_column = None
    for col in ['category_name', 'category', 'category_id']:
        if col in df.columns:
            category_column = col
            break

    if not category_column:
        print("В данных отсутствует информация о категориях")
        return

    plt.figure(figsize=(14, 8))

    # Берем только закрытые тикеты для расчета времени решения
    closed_tickets = df[df['resolution_time_hours'].notna()]
    if closed_tickets.empty:
        print("Нет данных о закрытых тикетах")
        return

    resolution_times = closed_tickets.groupby(category_column)['resolution_time_hours'].mean().sort_values()

    colors = plt.cm.plasma(np.linspace(0, 1, len(resolution_times)))
    bars = plt.barh(range(len(resolution_times)), resolution_times.values, color=colors, alpha=0.7)

    plt.yticks(range(len(resolution_times)), resolution_times.index)
    plt.title('Среднее время решения по категориям (часы)',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Среднее время решения (часы)', fontsize=12)
    plt.grid(True, alpha=0.3, axis='x')

    # Добавляем значения на столбцы
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.1, bar.get_y() + bar.get_height() / 2.,
                 f'{width:.1f} ч', ha='left', va='center', fontweight='bold')

    plt.tight_layout()
    plt.show()

    print("️  ВРЕМЯ РЕШЕНИЯ ПО КАТЕГОРИЯМ:")
    for category, time in resolution_times.items():
        print(f"   • {category}: {time:.1f} часов")
    print()


def show_top_problem_categories(df):
    """Таблица: топ-5 самых проблемных категорий по количеству тикетов"""
    category_column = None
    for col in ['category_name', 'category', 'category_id']:
        if col in df.columns:
            category_column = col
            break

    if not category_column:
        print("В данных отсутствует информация о категориях")
        return

    # Создаем агрегацию
    agg_data = {
        'count': ('ticket_id' if 'ticket_id' in df.columns else df.columns[0], 'count')
    }

    if 'resolution_time_hours' in df.columns:
        agg_data['avg_resolution_hours'] = ('resolution_time_hours', 'mean')

    problem_categories = df.groupby(category_column).agg(**agg_data).round(1)

    top_5 = problem_categories.nlargest(5, 'count')

    print("ТОП-5 ПРОБЛЕМНЫХ КАТЕГОРИЙ:")
    print("=" * 65)
    if 'avg_resolution_hours' in problem_categories.columns:
        print(f"{'Категория':<25} {'Кол-во тикетов':<15} {'Ср. время решения':<20}")
        print("-" * 65)
        for category, row in top_5.iterrows():
            print(f"{category:<25} {row['count']:<15} {row['avg_resolution_hours']:<20} часов")
    else:
        print(f"{'Категория':<25} {'Кол-во тикетов':<15}")
        print("-" * 45)
        for category, row in top_5.iterrows():
            print(f"{category:<25} {row['count']:<15}")
    print()


def plot_status_distribution(df):
    """Дополнительный график: распределение тикетов по статусам"""
    status_column = None
    for col in ['status_name', 'status', 'status_id']:
        if col in df.columns:
            status_column = col
            break

    if not status_column:
        return

    plt.figure(figsize=(12, 6))

    status_dist = df[status_column].value_counts()

    colors = plt.cm.Pastel1(np.linspace(0, 1, len(status_dist)))
    bars = plt.bar(range(len(status_dist)), status_dist.values, color=colors, alpha=0.7)

    plt.title('Распределение тикетов по статусам', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Статус', fontsize=12)
    plt.ylabel('Количество тикетов', fontsize=12)
    plt.xticks(range(len(status_dist)), status_dist.index, rotation=45)
    plt.grid(True, alpha=0.3, axis='y')

    # Добавляем значения на столбцы
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()

    print("РАСПРЕДЕЛЕНИЕ ПО СТАТУСАМ:")
    for status, count in status_dist.items():
        percentage = (count / len(df)) * 100
        print(f"   • {status}: {count} тикетов ({percentage:.1f}%)")
    print()


def plot_performance_comparison_speed(df):
    """
    График сравнения производительности сотрудников с отделом по скорости решения
    на основе данных тикетов
    """

    if df is None or df.empty:
        print("Нет данных для анализа производительности")
        return

    # Проверяем необходимые колонки
    required_columns = ['assigned_staff_id', 'created_at', 'closed_at']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Отсутствуют необходимые колонки: {missing_columns}")
        return

    # Фильтруем только закрытые тикеты
    closed_tickets = df[df['closed_at'].notna()].copy()
    if closed_tickets.empty:
        print("Нет данных о закрытых тикетах для анализа производительности")
        return

    print("АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ ПО СКОРОСТИ РЕШЕНИЯ")
    print("=" * 60)

    # Анализ по сотрудникам
    employee_performance = closed_tickets.groupby('assigned_staff_id').agg({
        'resolution_time_hours': ['count', 'mean', 'median'],
        'ticket_id': 'count'
    }).round(2)

    employee_performance.columns = ['ticket_count', 'avg_resolution_time', 'median_resolution_time', 'total_tickets']
    employee_performance = employee_performance[employee_performance['ticket_count'] >= 3]  # Минимум 3 тикета

    if employee_performance.empty:
        print("Недостаточно данных для сравнения производительности")
        return

    # Средние показатели по отделу
    department_avg_time = closed_tickets['resolution_time_hours'].mean()
    department_median_time = closed_tickets['resolution_time_hours'].median()

    print(f"Общие показатели отдела:")
    print(f"  Среднее время решения: {department_avg_time:.1f} часов")
    print(f"  Медианное время решения: {department_median_time:.1f} часов")
    print(f"  Всего закрытых тикетов: {len(closed_tickets)}")
    print()

    # Создаем график
    plt.figure(figsize=(14, 10))

    # График 1: Сравнение среднего времени решения
    plt.subplot(2, 1, 1)

    employees = [f"Сотр. {idx}" for idx in employee_performance.index]
    emp_avg_times = employee_performance['avg_resolution_time'].values
    dept_avg_line = [department_avg_time] * len(employees)

    x = np.arange(len(employees))
    width = 0.6

    bars = plt.bar(x, emp_avg_times, width, color='lightblue', alpha=0.7, edgecolor='black')

    # Линия среднего по отделу
    plt.axhline(y=department_avg_time, color='red', linestyle='--', linewidth=2,
                label=f'Среднее по отделу: {department_avg_time:.1f} ч')

    # Добавляем значения на столбцы
    for i, bar in enumerate(bars):
        height = bar.get_height()
        diff = height - department_avg_time
        color = 'red' if diff > 0 else 'green'
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                 f'{height:.1f}ч', ha='center', va='bottom', fontweight='bold')

        # Аннотация разницы
        if abs(diff) > department_avg_time * 0.1:  # Если разница более 10%
            plt.annotate(f"{'+' if diff > 0 else ''}{diff:.1f}ч",
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(bar.get_x() + bar.get_width() / 2, height + 2),
                         ha='center', va='bottom',
                         color=color, fontweight='bold',
                         arrowprops=dict(arrowstyle='->', color=color, alpha=0.7))

    plt.title('Сравнение среднего времени решения тикетов', fontsize=14, fontweight='bold')
    plt.xlabel('Сотрудники')
    plt.ylabel('Среднее время решения (часы)')
    plt.xticks(x, employees, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    # График 2: Количество тикетов и эффективность
    plt.subplot(2, 1, 2)

    ticket_counts = employee_performance['ticket_count'].values
    efficiency_ratio = [dept_avg_time / emp_time for emp_time in emp_avg_times]

    fig, ax1 = plt.subplots(figsize=(14, 6))

    color = 'tab:blue'
    bars = ax1.bar(x - width / 3, ticket_counts, width / 1.5, color=color, alpha=0.7, label='Количество тикетов')
    ax1.set_xlabel('Сотрудники')
    ax1.set_ylabel('Количество тикетов', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(x)
    ax1.set_xticklabels(employees, rotation=45)

    # Добавляем значения на столбцы количества
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                 f'{int(height)}', ha='center', va='bottom', fontweight='bold')

    # График эффективности
    color = 'tab:red'
    ax2 = ax1.twinx()
    ax2.plot(x + width / 3, efficiency_ratio, 'o-', color=color, linewidth=3, markersize=8, label='Коэф. эффективности')
    ax2.set_ylabel('Коэффициент эффективности', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Средний уровень')

    # Добавляем значения на точки эффективности
    for i, ratio in enumerate(efficiency_ratio):
        ax2.text(x[i] + width / 3, ratio + 0.02, f'{ratio:.2f}',
                 ha='center', va='bottom', fontweight='bold', color=color)

    plt.title('Количество тикетов и эффективность сотрудников', fontsize=14, fontweight='bold')
    fig.tight_layout()

    plt.show()

    # Детальная статистика
    print("ДЕТАЛЬНАЯ СТАТИСТИКА ПРОИЗВОДИТЕЛЬНОСТИ:")
    print("-" * 50)

    for idx, row in employee_performance.iterrows():
        emp_avg = row['avg_resolution_time']
        diff = emp_avg - department_avg_time
        percentage_diff = (diff / department_avg_time) * 100
        efficiency = department_avg_time / emp_avg

        status = "быстрее" if diff < 0 else "медленнее"

        print(f"Сотрудник {idx}:")
        print(f"  Среднее время: {emp_avg:.1f} ч, Разница: {diff:+.1f} ч ({percentage_diff:+.1f}%) {status}")
        print(f"  Коэффициент эффективности: {efficiency:.2f}")
        print(f"  Обработано тикетов: {int(row['ticket_count'])}")
        print()


def plot_performance_by_category(df):
    """
    Дополнительный анализ: производительность по категориям проблем
    """

    if df is None or df.empty:
        return

    # Проверяем необходимые колонки
    required_columns = ['category_name', 'assigned_staff_id', 'resolution_time_hours', 'closed_at']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Недостаточно данных для анализа по категориям: {missing_columns}")
        return

    # Фильтруем закрытые тикеты
    closed_tickets = df[df['closed_at'].notna()].copy()
    if closed_tickets.empty:
        return

    print("АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ ПО КАТЕГОРИЯМ ПРОБЛЕМ")
    print("=" * 50)

    # Группируем по категориям и сотрудникам
    category_performance = closed_tickets.groupby(['category_name', 'assigned_staff_id']).agg({
        'resolution_time_hours': ['mean', 'count']
    }).round(2)

    category_performance.columns = ['avg_time', 'ticket_count']
    category_performance = category_performance.reset_index()

    # Среднее время по категориям (по отделу)
    dept_category_avg = closed_tickets.groupby('category_name')['resolution_time_hours'].mean().round(2)

    # Создаем график
    plt.figure(figsize=(16, 10))

    # Берем топ-5 категорий по количеству тикетов
    top_categories = closed_tickets['category_name'].value_counts().head(5).index

    for i, category in enumerate(top_categories):
        plt.subplot(2, 3, i + 1)

        # Данные по текущей категории
        cat_data = category_performance[category_performance['category_name'] == category]
        dept_avg = dept_category_avg.get(category, 0)

        if len(cat_data) > 1:  # Только если есть несколько сотрудников
            employees = [f"Сотр. {idx}" for idx in cat_data['assigned_staff_id']]
            times = cat_data['avg_time'].values

            colors = ['lightgreen' if time <= dept_avg else 'lightcoral' for time in times]
            bars = plt.bar(range(len(employees)), times, color=colors, alpha=0.7, edgecolor='black')

            # Линия среднего по отделу
            plt.axhline(y=dept_avg, color='blue', linestyle='--', linewidth=2,
                        label=f'Среднее: {dept_avg:.1f}ч')

            # Добавляем значения
            for j, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                         f'{height:.1f}ч', ha='center', va='bottom', fontweight='bold')

            plt.title(f'Категория: {category}', fontsize=12, fontweight='bold')
            plt.xlabel('Сотрудники')
            plt.ylabel('Время решения (часы)')
            plt.xticks(range(len(employees)), employees, rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Выводим статистику по категориям
    print("СТАТИСТИКА ПО КАТЕГОРИЯМ:")
    print("-" * 40)

    for category in top_categories:
        cat_tickets = closed_tickets[closed_tickets['category_name'] == category]
        avg_time = cat_tickets['resolution_time_hours'].mean()
        std_time = cat_tickets['resolution_time_hours'].std()
        count = len(cat_tickets)

        print(f"{category}:")
        print(f"  Среднее время: {avg_time:.1f} ч, Стандартное отклонение: {std_time:.1f} ч")
        print(f"  Количество тикетов: {count}")

        # Лучший сотрудник в категории
        best_emp = cat_tickets.loc[cat_tickets['resolution_time_hours'].idxmin(), 'assigned_staff_id']
        best_time = cat_tickets['resolution_time_hours'].min()
        print(f"  Лучший результат: Сотр. {best_emp} - {best_time:.1f} ч")
        print()


def analyze_workload_distribution(df):
    """
    Анализ распределения нагрузки между сотрудниками
    """

    if df is None or df.empty:
        return

    print("АНАЛИЗ РАСПРЕДЕЛЕНИЯ НАГРУЗКИ")
    print("=" * 40)

    # Распределение тикетов по сотрудникам
    workload = df['assigned_staff_id'].value_counts()

    plt.figure(figsize=(12, 6))

    # График распределения нагрузки
    plt.subplot(1, 2, 1)
    colors = plt.cm.viridis(np.linspace(0, 1, len(workload)))
    bars = plt.bar(range(len(workload)), workload.values, color=colors, alpha=0.7, edgecolor='black')

    plt.title('Распределение тикетов по сотрудникам', fontweight='bold')
    plt.xlabel('Сотрудники')
    plt.ylabel('Количество тикетов')
    plt.xticks(range(len(workload)), [f"Сотр. {idx}" for idx in workload.index], rotation=45)

    # Добавляем значения
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                 f'{int(height)}', ha='center', va='bottom', fontweight='bold')

    # График загрузки (тикеты в работе)
    plt.subplot(1, 2, 2)

    # Текущие открытые тикеты
    open_tickets = df[df['closed_at'].isna()]
    open_workload = open_tickets['assigned_staff_id'].value_counts()

    if not open_workload.empty:
        colors = plt.cm.plasma(np.linspace(0, 1, len(open_workload)))
        bars = plt.bar(range(len(open_workload)), open_workload.values, color=colors, alpha=0.7, edgecolor='black')

        plt.title('Текущие открытые тикеты', fontweight='bold')
        plt.xlabel('Сотрудники')
        plt.ylabel('Количество открытых тикетов')
        plt.xticks(range(len(open_workload)), [f"Сотр. {idx}" for idx in open_workload.index], rotation=45)

        # Добавляем значения
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                     f'{int(height)}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()

    # Статистика нагрузки
    print("СТАТИСТИКА НАГРУЗКИ:")
    print(f"Всего тикетов: {len(df)}")
    print(f"Открытых тикетов: {len(open_tickets)}")
    print(f"Закрытых тикетов: {len(df) - len(open_tickets)}")
    print()

    for staff_id, count in workload.items():
        open_count = open_workload.get(staff_id, 0)
        closed_count = count - open_count
        completion_rate = (closed_count / count * 100) if count > 0 else 0

        print(f"Сотрудник {staff_id}:")
        print(f"  Всего тикетов: {count}")
        print(f"  Открыто: {open_count}, Закрыто: {closed_count}")
        print(f"  Процент завершения: {completion_rate:.1f}%")
        print()

def generate_text_report(df):
    """Генерация текстового отчета для руководства"""
    if df is None or df.empty:
        return "Нет данных для генерации отчета"

    # Основные метрики
    total_tickets = len(df)

    # Время решения (только для закрытых тикетов)
    if 'resolution_time_hours' in df.columns:
        closed_tickets = df[df['resolution_time_hours'].notna()]
        avg_resolution = closed_tickets['resolution_time_hours'].mean() if not closed_tickets.empty else 0
        resolution_info = f"{avg_resolution:.1f} часов ({len(closed_tickets)} закрытых тикетов)"
    else:
        resolution_info = "данные недоступны"

    # Пиковые периоды
    if 'created_at' in df.columns:
        peak_hour = df['created_at'].dt.hour.mode()[0] if not df.empty else 0
        peak_day = df['created_at'].dt.day_name().mode()[0] if not df.empty else "N/A"
        date_range = f"{df['created_at'].min().strftime('%d.%m.%Y')} - {df['created_at'].max().strftime('%d.%m.%Y')}"
    else:
        peak_hour = 0
        peak_day = "N/A"
        date_range = "неизвестно"

    # Самая частая категория
    category_column = None
    for col in ['category_name', 'category', 'category_id']:
        if col in df.columns:
            category_column = col
            break

    top_category = df[category_column].mode()[0] if category_column and not df.empty else "N/A"

    # Статусы
    status_column = None
    for col in ['status_name', 'status', 'status_id']:
        if col in df.columns:
            status_column = col
            break

    if status_column:
        open_tickets = len(df[df[status_column].str.contains('открыт|open|в работе', case=False, na=False)])
        closed_tickets = len(df[df[status_column].str.contains('закрыт|closed|решено', case=False, na=False)])
        status_info = f"{open_tickets} открытых, {closed_tickets} закрытых"
    else:
        status_info = "статусы недоступны"

    report = f"""
АНАЛИТИЧЕСКИЙ ОТЧЕТ СИСТЕМЫ ТЕХПОДДЕРЖКИ
{'=' * 50}

 КЛЮЧЕВЫЕ ПОКАЗАТЕЛИ:
   • Всего тикетов: {total_tickets}
   • Среднее время решения: {resolution_info}
   • Период анализа: {date_range}
   • Статус тикетов: {status_info}

 ВРЕМЕННЫЕ ЗАКОНОМЕРНОСТИ:
   • Пиковое время создания: {peak_hour}:00
   • Самый активный день: {peak_day}
   • Самая проблемная категория: {top_category}

 РЕКОМЕНДАЦИИ ДЛЯ РУКОВОДСТВА:
   1. Увеличить количество сотрудников в пиковые часы ({peak_hour}:00-{peak_hour + 1}:00)
   2. Сфокусировать внимание на категории "{top_category}"
   3. Оптимизировать процессы для сокращения времени решения
   4. Рассмотреть автоматизацию для часто встречающихся проблем

 ДЕТАЛЬНАЯ СТАТИСТИКА:
   • Всего дней в анализе: {(df['created_at'].max() - df['created_at'].min()).days + 1}
   • Среднее количество тикетов в день: {total_tickets / ((df['created_at'].max() - df['created_at'].min()).days + 1):.1f}
"""
    return report


def main():
    """Главная функция для запуска аналитической панели"""
    print(" ЗАПУСК АНАЛИТИЧЕСКОЙ ПАНЕЛИ ТЕХПОДДЕРЖКИ")
    print("=" * 60)

    # Загрузка данных
    df = load_and_prepare_data()
    if df is None:
        return

    print("\n📊 ЗАПУСК АНАЛИТИКИ...")
    print("=" * 60)

    # Построение графиков
    plot_ticket_trends(df)
    plot_hourly_distribution(df)
    plot_heatmap(df)
    plot_category_distribution(df)
    plot_resolution_time_by_category(df)
    plot_status_distribution(df)
    show_top_problem_categories(df)
    plot_performance_comparison_speed(df)
    plot_performance_by_category(df)
    analyze_workload_distribution(df)

    # Генерация отчета
    print(" ФИНАЛЬНЫЙ ОТЧЕТ:")
    print("=" * 60)
    report = generate_text_report(df)
    print(report)


if __name__ == "__main__":
    # Настройка стиля перед запуском
    setup_plot_style()
    main()