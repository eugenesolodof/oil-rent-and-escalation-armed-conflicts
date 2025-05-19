import re
import pandas as pd
import numpy as np
from typing import List, Optional, Dict
import glob
import os
from pathlib import Path
import country_converter as coco
from sklearn.preprocessing import PowerTransformer

def select_columns(
    df: pd.DataFrame,
    id_cols: List[str] = ['year', 'code'],
    value_cols: List[str] = ['price'],
    column_mapping: Dict[str, str] = None,
    country_code_conversion: bool = True,
    skip_rows: Optional[int] = None,
    covert_to_int: bool = True
) -> pd.DataFrame:
    """
    Выбирает и переименовывает колонки с обработкой данных.
    
    :param df: Исходный DataFrame
    :param id_cols: Список идентификационных колонок
    :param value_cols: Список колонок со значениями
    :param column_mapping: Словарь для переименования колонок
    :param country_code_conversion: Флаг конвертации в коды стран
    :param skip_rows: Пропуск строк
    :param covert_to_int: Конвертация значений в строке в int
    :return: Обработанный DataFrame
    """
    df = df.copy()

    # Проверка необходимости пропуска строк
    if 'Unnamed: 1' in df.columns:
        df.columns = df.iloc[skip_rows-1]
    
    # Унифицированная обработка колонок
    if column_mapping:
        df = df.rename(columns=column_mapping)
    
    # Автоматический выбор колонок если не указаны явно
    selected_cols = []
    for col_group in [id_cols, value_cols]:
        selected_cols += [c for c in col_group if c in df.columns]
    
    df = df[selected_cols]
    
    # Конвертация кодов стран
    if country_code_conversion and 'code' in df.columns:
        df['code'] = convert_to_gw_code(df['code'])
    
    # Обработка числовых данных
    for col in value_cols:
        if col in df.columns:
            df[col] = clean_numeric(df[col])
    
    # Конвертация переменную-ключей в строковые
    df[id_cols[0]] = pd.to_numeric(df[id_cols[0]],errors='coerce').astype('Int64')
    if covert_to_int:
        df[id_cols[1]] = pd.to_numeric(df[id_cols[1]],errors='coerce').astype('Int64')

    return df

def rotate_columns(
    df: pd.DataFrame,
    value_name: str,
    id_vars: str = 'code',
    column_mapping: Dict[str, str] = None,
    var_name: str = 'year',
    indicator_filter: Optional[str] = None,
    threshold: Optional[float] = None,
    skip_rows: Optional[int] = None
) -> pd.DataFrame:
    """
    Трансформирует данные из широкого формата в длинный.
    
    :param df: Исходный DataFrame
    :param value_name: Имя для колонки со значениями
    :param id_vars: Колонка-идентификатор
    :param var_name: Имя для колонки с годами
    :param indicator_filter: Фильтр для индикатора
    :param threshold: Порог для фильтрации
    :param skip_rows: Пропуск строк
    :return: Трансформированный DataFrame
    """
    df = df.copy()

    # Проверка необходимости пропуска строк
    if 'Unnamed: 1' in df.columns:
        df.columns = df.iloc[skip_rows-1]

    # Унифицированная обработка колонок
    if column_mapping:
        df = df.rename(columns=column_mapping)
    
    # Фильтрация по индикатору
    if indicator_filter and 'Indicator Code' in df.columns:
        df = df[df['Indicator Code'] == indicator_filter]
    
    # Автоматическое определение колонок с годами
    year_columns = detect_year_columns(df)
    
    # Преобразование формата
    df_long = pd.melt(
        df,
        id_vars=id_vars,
        value_vars=year_columns,
        var_name=var_name,
        value_name=value_name
    )
    
    # Очистка данных
    df_long[var_name] = clean_years(df_long[var_name])
    df_long[value_name] = clean_numeric(df_long[value_name])

    # Конвертация кодов стран
    if 'code' in df_long.columns:
        df_long['code'] = convert_to_gw_code(df_long['code'])

    # Группировка данных и фильтрация по порогу
    if threshold is not None and 'code' in df_long.columns:
        grouped = df_long.groupby(by=['code']).mean(value_name)
        list_with_countries = grouped[grouped[value_name] > threshold].index.to_list()
        list_with_countries = [item for item in list_with_countries if isinstance(item, (int, float))]
        df_long = df_long[df_long['code'].isin(list_with_countries)]

    # Конвертация переменную-ключей в строковые
    df_long[var_name] = pd.to_numeric(df_long[var_name],errors='coerce').astype('Int64')
    df_long[id_vars] = pd.to_numeric(df_long[id_vars],errors='coerce').astype('Int64')
    
    return df_long

def expand_rows(
    df: pd.DataFrame,
    start_year_col: str = 'styear',
    end_year_col: str = 'endyear',
    min_year: int = 1960,
    date_convert_flg: bool = False
) -> pd.DataFrame:
    """
    Расширяет временные ряды для интервалов лет.
    
    :param df: Исходный DataFrame
    :param start_year_col: Колонка с начальным годом
    :param end_year_col: Колонка с конечным годом
    :param min_year: Минимальный год для генерации
    :param date_convert_flg: Флаг конвертации даты в год
    :return: Расширенный DataFrame
    """
    if date_convert_flg:
        df[start_year_col] = df[start_year_col].apply(lambda x: pd.to_datetime(x).year)
        df[end_year_col] = df[end_year_col].apply(lambda x: pd.to_datetime(x).year)

    def generate_years(row):
        start = max(row[start_year_col], min_year)
        return list(range(start, row[end_year_col] + 1))
    
    df = df.copy()
    df['year'] = df.apply(generate_years, axis=1)
    return df.explode('year').reset_index(drop=True)

def make_extension_data(
    df: pd.DataFrame,
    metric_name: str,
    groupby_cols: List[str] = ['code', 'year'],
    date_cols: Dict[str, str] = None,
    side_cols: Dict[str, str] = None,
    level_filter: Optional[int] = None
) -> pd.DataFrame:
    """
    Агрегирует данные по заданным параметрам.
    
    :param df: Исходный DataFrame
    :param metric_name: Имя для результирующей метрики
    :param groupby_cols: Колонки для группировки
    :param date_cols: Словарь с колонками дат
    :param side_cols: Словарь для обработки сторон
    :param level_filter: Порог уровня для фильтрации
    :return: Агрегированный DataFrame
    """
    df = df.copy()

    # Обработка дат
    if date_cols:
        for col, dt_col in date_cols.items():
            df[col] = pd.to_datetime(df[dt_col]).dt.year
    
    # Обработка сторон
    if side_cols:
        for side_col, code_col in side_cols.items():
            if side_col in df.columns:
                df[code_col] = df[side_col].str.split(',').str[0]
                df[code_col] = convert_to_gw_code(df[code_col])
    
    # Фильтрация по уровню
    if level_filter and 'hostlev' in df.columns:
        df = df[df['hostlev'] >= level_filter]
    
    # Агрегация
    result = df.groupby(groupby_cols, observed=True).size().reset_index(name=metric_name)
    
    return result


##########################################################

def preprocess_events(
        df: pd.DataFrame,
        start: List[str] = ['stday','stmon','styear'],
        end: List[str] = ['endday','endmon','endyear'],
        date_filter_left: str = '1986-01-02',
        drop_unknown_dates: bool = True,
        id_col: str = 'id',
        duration_col: str = 'days'
) -> pd.DataFrame:
    """
    Описание функции

    :param df: Исходный DataFrame
    :param start: Список компонентов даты начала [день, месяц, год]
    :param end: Список компонентов даты завершения [день, месяц, год]
    :param date_filter_left: Левая граница времени
    :param drop_unknown_dates: Флаг очистики данных от неизвестных дат
    :param id_col: Колонка с идентификатором события
    :param duration_col: Колонка с продолжительностью
    :return DataFrame с новыми колонками
    """

    # Очистка данных от неизвестныз дат
    if drop_unknown_dates:
        df = df[(df[start[0]] > 0) & (df[end[0]] > 0)]

    # Трансформация даты в формат 'YYYY-mm-dd'
    for time in [start, end]:
        df = create_datetime_column(df, {'year':time[2],'month':time[1],'day':time[0]},f'{time[0]}')

    # Фильтрация данных по нижней границе даты
    if date_filter_left:
        df = df[df['stday'] >= date_filter_left]

    # Трансформация даты в количество дней 
    df['days'] = (df['endday']-df['stday']).apply(lambda x: x.days)
    df = df[df['days'] > 0].reset_index(drop=True)
    df = df.reset_index(names='id')

    # Трансформация даты в год (понадобится далее для джойна)
    df['year'] = df['stday'].apply(lambda x: x.year)
    
    df = expand_events(df,id_col,duration_col)

    return df

def expand_events(
    df: pd.DataFrame, 
    id_col: str = 'id',
    duration_col: str = 'days',
) -> pd.DataFrame:
    """
    Расширяет события по дням.
    
    :param df: DataFrame с событиями
    :param id_col: Колонка с идентификатором события
                                                                    #:param date_col: Колонка с датой начала
    :param duration_col: Колонка с продолжительностью
    :return: Расширенный DataFrame
    """
    df_expanded = pd.DataFrame({
        id_col: np.repeat(df[id_col], df[duration_col]),
        'date': [d for row in df.itertuples() 
               for d in pd.date_range(row.stday, periods=row.days)]
    })
    
    df_expanded = df_expanded.merge(df, on=id_col)
    df_expanded['event'] = df_expanded.groupby(id_col).cumcount() + 1 == df_expanded[duration_col]
    df_expanded['event'] = df_expanded['event'].astype(int)
    
    return df_expanded

def prepare_and_fill_data(
    df: pd.DataFrame,
    start_date: str = '1986-01-01', 
    end_date: str = '2023-01-01',
    right_on: str = 'observation_date',
    col_name: str = 'price',
) -> pd.DataFrame:
    """
    Подготавливает данные о ценах на нефть.
    
    :param df: DataFrame с ценами
    :param start_date: Начальная дата диапазона
    :param end_date: Конечная дата диапазона
    :param right_on: Колонка-ключ в правой таблице
    :param col_name: Колонка с пропусками
    :return: DataFrame с ценами
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    df[right_on]=pd.to_datetime(df[right_on])

    df_full = pd.DataFrame({'Date': date_range}).merge(
        df, how='left', left_on='Date', right_on=right_on
    )
    df_full[col_name] = df_full[col_name].ffill()

    # Трансформация даты в год (понадобится далее для джойна)
    df_full['year'] = df_full['Date'].apply(lambda x: x.year)
    return df_full

def calculate_inflation_adjustment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Рассчитывает инфляционную корректировку.
    
    :param df: DataFrame с данными инфляции
    :return: DataFrame с коэффициентами корректировки
    """
    df = df.copy()
    df['cum'] = df['Cumulative price change'].str.split('%').str[0].astype(float)*0.01 + 1
    return df.drop(columns=['Cumulative price change'])

def merge_and_finalize_data(
    events_df: pd.DataFrame,
    price_df: pd.DataFrame,
    inflation_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Объединяет все данные и выполняет финальную обработку.
    
    :param events_df: DataFrame с событиями
    :param price_df: DataFrame с ценами
    :param inflation_df: DataFrame с инфляцией
    :return: Итоговый DataFrame для анализа
    """
    merged = events_df.merge(
        price_df.merge(inflation_df, left_on='year', right_on='Year'),
        left_on='date',
        right_on='Date'
    )
    
    merged['price_daily'] = round(merged['price'] * merged['cum'], 2)
    
    merged['tstop'] = merged.groupby('id')['date'].cumcount() + 1
    merged['tstart'] = merged.groupby('id')['date'].cumcount()

    merged[['ccode1','ccode2']] = merged[['ccode1','ccode2']].astype(int)
    
    return merged[['id', 'ccode1', 'ccode2', 'days', 'year_x', 
                 'price_daily', 'event', 'tstart', 'tstop']]

# Вспомогательные функции
def convert_to_gw_code(series: pd.Series) -> pd.Series:
    """Универсальная конвертация в GW-коды"""
    return (
        coco.
        convert(
            names=series
            .astype(str)
            .str.split('[.,]')
            .str[0],
            to='GWcode',
            not_found=None)
            )

def clean_numeric(series: pd.Series) -> pd.Series:
    """Очистка числовых данных"""
    return (
        series
        .astype(str)
        .str.replace('[^\d.]', '', regex=True)
        .replace({'': None, 'nan': None})
        .astype(float)
    )

def detect_year_columns(df: pd.DataFrame) -> List[str]:
    """Автоматическое определение колонок с годами"""
    return [
        col for col in df.columns 
        if (
            (isinstance(col, (int, float)) and 1960 <= col <= 2008)
            ) or (
                isinstance(col, str) and re.match(r"^\d{4}$", col) and 1960 <= int(col) <= 2008
                )
                ]

def clean_years(series: pd.Series) -> pd.Series:
    """Очистка годовых значений"""
    return (
        series
        .astype(str)
        .str.extract('(\d{4})')[0]
        .astype(float)
        .astype(pd.Int64Dtype())
    )

def create_datetime_column(
    df: pd.DataFrame, 
    date_components: Dict[str, str], 
    new_col: str,
) -> pd.DataFrame:
    """
    Создает колонку с датой из компонентов.
    
    :param df: Исходный DataFrame
    :param date_components: Словарь компонентов даты {год: ..., месяц: ..., день: ...}
    :param new_col: Название новой колонки с датой
    :return: DataFrame с новой колонкой
    """
    temp_df = df.rename(columns={
        date_components['year']: 'year',
        date_components['month']: 'month',
        date_components['day']: 'day'
    })

    df[new_col] = pd.to_datetime(temp_df[['year', 'month', 'day']])
    
    return df.drop(columns=['year', 'month', 'day'], errors='ignore')