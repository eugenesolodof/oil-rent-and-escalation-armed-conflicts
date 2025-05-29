import glob
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer
from my_module import (
    select_columns, rotate_columns, preprocess_events, calculate_inflation_adjustment, expand_rows, make_extension_data, prepare_and_fill_data, merge_and_finalize_data
    )

"""
КОНФИГИ ДЛЯ ПОДАЧИ В ФУНКЦИИ
"""

conf_for_select_columns = {
    'atop' : {
        'id_cols' : ['year','code'],
        'value_cols' : ['number'],
        'column_mapping' : {'state': 'code'},
        'country_code_conversion' : False,
        'skip_rows' : None,
        'covert_to_int' : False
        },
    'brunn' : {
        'id_cols' : ['year','code'],
        'value_cols' : ['for_priv'],
        'column_mapping' : {'wbcode': 'code'},
        'country_code_conversion' : True,
        'skip_rows' : None,
        'covert_to_int' : False
        },
    'colgan' : {
        'id_cols' : ['year','ccode'],
        'value_cols' : ['revolutionaryleader'],
        'column_mapping' : None,
        'country_code_conversion' : False,
        'skip_rows' : None,
        'covert_to_int' : True
        },
    'vdem' : {
        'id_cols' : ['year','code'],
        'value_cols' : ['v2x_polyarchy','e_gdppc','v2svstterr','v2svindep'],
        'column_mapping' : {'country_text_id': 'code'},
        'country_code_conversion' : True,
        'skip_rows' : None,
        'covert_to_int' : False
        },
    'oil crude prices since 1861' : {
        'id_cols' : ['year'],
        'value_cols' : ['price'],
        'column_mapping' : {'Year': 'year', '$ 2023': 'price'},
        'country_code_conversion' : False,
        'skip_rows' : 3,
        'covert_to_int' : False
        }
    }

conf_for_rotate_columns = {
    'rent15' : {
        'value_name' :'rent15',
        'id_vars' : 'code',
        'column_mapping' : {'Country Code': 'code'},
        'var_name' : 'year',
        'indicator_filter' : 'NY.GDP.PETR.RT.ZS',
        'threshold' : 15,
        'skip_rows' : None
    },
    'rent10' : {
        'value_name' :'rent10',
        'id_vars' : 'code',
        'column_mapping' : {'Country Code': 'code'},
        'var_name' : 'year',
        'indicator_filter' : 'NY.GDP.PETR.RT.ZS',
        'threshold' : 10,
        'skip_rows' : None
    },
    'export50' : {
        'value_name' :'export50',
        'id_vars' : 'code',
        'column_mapping' : {'Country Code': 'code'},
        'var_name' : 'year',
        'indicator_filter' : 'TX.VAL.FUEL.ZS.UN',
        'threshold' : 50,
        'skip_rows' : None
    },
    'oil production - barrels' : {
        'value_name' :'barrels',
        'id_vars' : 'code',
        'column_mapping' : {'Thousand barrels daily': 'code'},
        'var_name' : 'year',
        'indicator_filter' : None,
        'threshold' : None,
        'skip_rows' : 2
    },
        'kane' : {
        'value_name' :'us_troops',
        'id_vars' : 'code',
        'column_mapping' : {'Unnamed: 0': 'code'},
        'var_name' : 'year',
        'indicator_filter' : None,
        'threshold' : None,
        'skip_rows' : None
    },
}

conf_for_expand_rows = {
    'mie' : {
        'start_year_col' :'styear',
        'end_year_col' : 'endyear',
        'min_year' : 1960,
        'date_convert_flg' : False
    }, 
    'nsa' : {
        'start_year_col' :'startdate',
        'end_year_col' : 'enddate',
        'min_year' : 1960,
        'date_convert_flg' : True
    }, 
    'miller' : {
        'start_year_col' :'styear',
        'end_year_col' : 'endyear',
        'min_year' : 1960,
        'date_convert_flg' : False
    }
}

conf_for_make_extension_data = {
    'mie(2)' : {
        'metric_name' :'mie(2)',
        'groupby_cols' : ['ccode1','year'],
        'side_cols' : None,
        'level_filter' : 2,
        'date_cols' : None
    }, 
    'mie(4)' : {
        'metric_name' :'mie(4)',
        'groupby_cols' : ['ccode1','year'],
        'side_cols' : None,
        'level_filter' : 4,
        'date_cols' : None
    }, 
    'mie(2)_against' : {
        'metric_name' :'mie(2)_against',
        'groupby_cols' : ['ccode2','year'],
        'side_cols' : None,
        'level_filter' : 2,
        'date_cols' : None
    }, 
    'mie(4)_against' : {
        'metric_name' :'mie(4)_against',
        'groupby_cols' : ['ccode2','year'],
        'side_cols' : None,
        'level_filter' : 4,
        'date_cols' : None
    }, 
    'miller_a' : {
        'metric_name' :'rivalries_a',
        'groupby_cols' : ['code','year'],
        'side_cols' : {'sidea': 'code'},
        'level_filter' : None,
        'date_cols' : None
    }, 
    'miller_b' : {
        'metric_name' :'rivalries_b',
        'groupby_cols' : ['code','year'],
        'side_cols' : {'sideb': 'code'},
        'level_filter' : None,
        'date_cols' : None
    },
        'nsa' : {
        'metric_name' :'rebels',
        'groupby_cols' : ['code','year'],
        'side_cols' : {'side_a': 'code'},
        'level_filter' : None,
        'date_cols' : None
    }
}

conf_for_prepare_and_fill_data = {
    'price_daily' : {
        'start_date' : '1986-01-01',
        'end_date' : '2023-01-01',
        'right_on' : 'observation_date',
        'col_name' : 'price'
    }
}

conf_for_preprocess_events = {
    'events' : {
        'start' : ['stday','stmon','styear'],
        'end' : ['endday','endmon','endyear'],
        'date_filter_left' : '1986-01-02',
        'drop_unknown_dates' : True,
        'id_col' : 'id',
        'duration_col' : 'days'
    }   
}

"""
НАСТРОЙКА ПАРАМЕТРОВ ЦИКЛА 
"""

input_base = glob.glob(r'C:\Users\ESolodov\Desktop\mine\df_base\*') #
input_added = glob.glob(r'C:\Users\ESolodov\Desktop\mine\df_added\*') #
output = r'C:\Users\ESolodov\Desktop\mine\df_res' #

correct_extension = 'csv'

samples = [
    'rent15', 'rent10', 'export50'
    ]

function_to_dataframe = {
    'select_columns' : ['atop', 'brunn', 'colgan', 'oil crude prices since 1861', 'vdem'],
    'rotate_columns' : ['kane', 'oil production - barrels'],
    'make_extension_data' : ['mie', 'nsa', 'miller'],
    'prepare_and_fill_data' : ['price_daily'],
    'preprocess_events' : ['events'],
    'calculate_inflation_adjustment' : ['inflation']
    }

dataframe_from_mie = [
    'mie(2)',
    'mie(4)',
    'mie(2)_against',
    'mie(4)_against'
]

dataframe_from_miller = [
    'miller_a', 'miller_b'
    ]

cols_with_replaced_nan = [
    'riv', 'number', 'rebels', 'mie(2)', 'mie(4)', 'mie(2)_against', 'mie(4)_against'
]

# справочник для перекодировки GWcode в наименования стран
refer = pd.read_csv('https://correlatesofwar.org/wp-content/uploads/COW-country-codes.csv')
refer = refer.iloc[:,1:]
refer = refer.rename(columns={refer.columns[0]: 'code'})
refer = refer.drop_duplicates()
"""
НИЖЕ НАЧИНАЕТСЯ СБОРКА ДАННЫХ
"""

# Запускаем цикл по сборке переменных в рамках одной выборки
for s in samples:

    # Открываем базовый датафрейм, на основе которого формируем выборки страны
    df = pd.read_csv(input_base[0])
    
    params = conf_for_rotate_columns.get(s, {})
    df_sample = rotate_columns(
        df,
        value_name=params.get('value_name',{}),
        id_vars=params.get('id_vars',{}),
        column_mapping=params.get('column_mapping',{}),
        var_name=params.get('var_name',{}),
        indicator_filter=params.get('indicator_filter',{}),
        threshold=params.get('threshold',{}),
        skip_rows=params.get('skip_rows',{})
        )
    print(f'{input_base[0]} успешно открыт.')

    # Во внутреннем цикле открываем файлы с переменными
    for file in input_added:
        file_format = file.split('.')[-1]
        file_name = file.split('.')[0].split('\\')[-1]
        
        # Проверка расширения файла
        if file_format == correct_extension:
            try:
                df = pd.read_csv(file)

                # Определяем функцию, которую нужно применить к таблице
                for key, values in function_to_dataframe.items():
                    if file_name in values:
                        correct_function = key

                if correct_function == 'select_columns':
                    try:
                        params = conf_for_select_columns.get(file_name, {}) 
                        df_added = select_columns(
                            df,
                            id_cols=params.get('id_cols',{}),
                            value_cols=params.get('value_cols',{}),
                            column_mapping=params.get('column_mapping',{}),
                            country_code_conversion=params.get('country_code_conversion',{}),
                            skip_rows=params.get('skip_rows',{}),
                            covert_to_int=params.get('covert_to_int',{})
                            )
                        if len(params.get('id_cols',{})) < 2: left = ['year']
                        else: left = ['year', 'code']
                        df_sample = pd.merge(df_sample,df_added,left_on=left,right_on=params.get('id_cols',{}),how='left')
                        print(f'{file} обработан и соединен с базовым датафреймом')
                    
                    except Exception as e:
                        print(f'{file} не был обработан / соединен с базовым датафреймом: {e}')

                elif correct_function == 'rotate_columns':
                    try:
                        params = conf_for_rotate_columns.get(file_name, {}) 
                        df_added = rotate_columns(
                            df,
                            value_name=params.get('value_name',{}),
                            id_vars=params.get('id_vars',{}),
                            column_mapping=params.get('column_mapping',{}),
                            var_name=params.get('var_name',{}),
                            indicator_filter=params.get('indicator_filter',{}),
                            threshold=params.get('threshold',{}),
                            skip_rows=params.get('skip_rows',{})
                            )
                        df_sample = pd.merge(df_sample,df_added,left_on=['code','year'],right_on=[params.get('id_vars',{}),params.get('var_name',{})],how='left')
                        print(f'{file} обработан и соединен с базовым датафреймом')
                    except Exception as e:
                        print(f'{file} не был обработан / соединен с базовым датафреймом: {e}')

                elif correct_function == 'make_extension_data':
                    try:
                        if file_name == 'miller':
                            params = conf_for_expand_rows.get(file_name, {})
                            df_exp = expand_rows(
                                df,
                                start_year_col=params.get('start_year_col',{}),
                                end_year_col=params.get('end_year_col',{}),
                                min_year=params.get('min_year',{}),
                                date_convert_flg=params.get('date_convert_flg',{}),
                                )
                            for element in dataframe_from_miller:
                                params = conf_for_make_extension_data.get(element, {})
                                df_added = make_extension_data(
                                    df_exp,
                                    metric_name=params.get('metric_name',{}),
                                    groupby_cols=params.get('groupby_cols',{}),
                                    side_cols=params.get('side_cols',{}),
                                    level_filter=params.get('level_filter',{}),
                                    date_cols=params.get('date_cols',{})
                                )
                                df_sample = pd.merge(df_sample,df_added,on=['code','year'],how='left')
                                print(f'{file} обработан и соединен с базовым датафреймом')

                        elif file_name == 'nsa':
                            params = conf_for_expand_rows.get(file_name, {})
                            df_exp = expand_rows(
                                df,
                                start_year_col=params.get('start_year_col',{}),
                                end_year_col=params.get('end_year_col',{}),
                                min_year=params.get('min_year',{}),
                                date_convert_flg=params.get('date_convert_flg',{}),
                                )
                            params = conf_for_make_extension_data.get(file_name, {})
                            df_added = make_extension_data(
                                df_exp,
                                metric_name=params.get('metric_name',{}),
                                groupby_cols=params.get('groupby_cols',{}),
                                side_cols=params.get('side_cols',{}),
                                level_filter=params.get('level_filter',{}),
                                date_cols=params.get('date_cols',{})
                                )
                            df_sample = pd.merge(df_sample,df_added,on=['code','year'],how='left')
                            print(f'{file} обработан и соединен с базовым датафреймом')

                        elif file_name == 'mie':
                            params = conf_for_expand_rows.get(file_name, {})
                            df_exp = expand_rows(
                                df,
                                start_year_col=params.get('start_year_col',{}),
                                end_year_col=params.get('end_year_col',{}),
                                min_year=params.get('min_year',{}),
                                date_convert_flg=params.get('date_convert_flg',{}),
                                )
                            for element in dataframe_from_mie:
                                params = conf_for_make_extension_data.get(element, {})
                                df_added = make_extension_data(
                                    df_exp,
                                    metric_name=params.get('metric_name',{}),
                                    groupby_cols=params.get('groupby_cols',{}),
                                    side_cols=params.get('side_cols',{}),
                                    level_filter=params.get('level_filter',{}),
                                    date_cols=params.get('date_cols',{}),
                                    )
                                df_sample = pd.merge(df_sample,df_added,left_on=['code','year'],right_on=[params.get('groupby_cols',{})[0],'year'],how='left')
                                print(f'{file} обработан и соединен с базовым датафреймом')

                    except Exception as e:
                        print(f'{file} не был обработан / соединен с базовым датафреймом: {e}')

                elif correct_function == 'prepare_and_fill_data':
                    try:
                        params = conf_for_prepare_and_fill_data.get(file_name, {})
                        df_price = prepare_and_fill_data(
                            df,
                            start_date=params.get('start_date',{}),
                            end_date=params.get('end_date',{}),
                            right_on=params.get('right_on',{}),
                            col_name=params.get('col_name',{})
                            )
                        print(f'{file} обработан')

                    except Exception as e:
                        print(f'{file} не был обработан / соединен с базовым датафреймом: {e}')

                elif correct_function == 'preprocess_events':
                    try:
                        params = conf_for_preprocess_events.get(file_name, {})
                        df_events = preprocess_events(
                            df,
                            start=params.get('start',{}),
                            end=params.get('end',{}),
                            date_filter_left=params.get('date_filter_left',{}),
                            drop_unknown_dates=params.get('drop_unknown_dates',{}),
                            id_col=params.get('id_col',{}),
                            duration_col=params.get('duration_col',{}),
                            )
                        print(f'{file} обработан')

                    except Exception as e:
                        print(f'{file} не был обработан / соединен с базовым датафреймом: {e}')

                elif correct_function == 'calculate_inflation_adjustment':
                    try:
                        df_infl = calculate_inflation_adjustment(df)
                        print(f'{file} обработан')
                    
                    except Exception as e:
                        print(f'{file} не был обработан / соединен с базовым датафреймом: {e}')

            except Exception as e:
                print(f'{file} не был открыт: {e}')
            
    # Объединение данных для регрессии Кокса
    merged_events_df = merge_and_finalize_data(df_events,df_price,df_infl)

    # Очистка данных
    df_sample['code'] = (
        pd.to_numeric(df_sample['code'],errors='coerce').astype('Int64')
    )
    df_sample = df_sample.dropna(subset=['code'])
    df_sample = df_sample.replace([-9,'-'],np.nan)

    # Рассчет дополнительных переменных
    df_sample['lagged'] = df_sample.groupby(['code'])['price'].shift(1)
    df_sample['roll_mean3'] = df_sample.groupby(['code'])['price'].transform(lambda x: x.rolling(window=3).mean())
    df_sample['roll_mean5'] = df_sample.groupby(['code'])['price'].transform(lambda x: x.rolling(window=5).mean())

    # Трансформация переменных
    df_sample['log_gdppc'] = np.log(df_sample['e_gdppc'])
    pt = PowerTransformer(method='yeo-johnson')
    df_sample['troops_yj'] = pt.fit_transform(df_sample[['us_troops']])
    df_sample['prod_yj'] = pt.fit_transform(df_sample[['barrels']])
    df_sample['riv'] = df_sample['rivalries_a'].fillna(0) + df_sample['rivalries_b'].fillna(0)

    # Фильтрация (оставляем только независимые страны + убираем из наблюдений СССР)
    df_sample = df_sample[df_sample['v2svindep'] == 1]
    df_sample = df_sample[~((df_sample['code'] == 365) & (df_sample['year'] <= 1990))]

    # Обработка нулевых значений
    df_sample['barrels'] = df_sample['barrels'].replace(0, np.nan)
    df_sample[cols_with_replaced_nan] = df_sample[cols_with_replaced_nan].fillna(0)

    # Перекодировка GWcode в названия стран
    df_sample = pd.merge(df_sample,refer,on='code',how='left')
    df_sample.insert(0,'StateNme',df_sample.pop('StateNme'))

    # Объединение данных
    merged_events_df = df_sample.merge(merged_events_df,left_on=['code','year'],right_on=['ccode1','year_x'],how='left')
    merged_events_df_against = df_sample.merge(merged_events_df,left_on=['code','year'],right_on=['ccode2','year_x'],how='left')

    # Сохранение собранных данных
    datasets = [
        (df_sample, f'{s}_sample.csv'),
        (merged_events_df, f'{s}_events.csv'),
        (merged_events_df_against, f'{s}_events_against.csv')
        ]

    for df, written_file_name in datasets:
        full_path = os.path.join(output, written_file_name)
        try:
            df.to_csv(full_path, index=False)
            print(f"Успешно сохранено: {full_path}")
        except Exception as e:
            print(f"Ошибка при сохранении {full_path}: {str(e)}")
