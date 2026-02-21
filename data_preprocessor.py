import pandas as pd
import numpy as np

class DataPreprocessor:
    def __init__(self, df: pd.DataFrame):
        # проверим входные данные
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Входные данные должны быть типа pandas.DataFrame!")
        
        self.df = df.copy()     # работаем с копией, чтобы не изменять оригинал
        self._is_fitted = False
        
        # хранилище для данных
        self.params = {
            'cols_to_keep': None, # колонки, прошедшие порог пропусков
            'impute_values': {},  # значения (медианы или моды) для заполнения
            'scaling_stats': {},  # мин/макс/среднее/стандартизация
            'cat_columns': [],    # исходные категориальные колонки
            'final_columns': [],  # финальные колонки после one-hot
            'norm_method': 'minmax'   
        }


    def remove_missing(self, threshold: float = 0.5, strategy = 'median'):      
        '''
        Метод, который удаляет столбцы при проценте пропусков
        более threshold. Пропуски в остальных столбцах
        заполняет медианой или модой.
        '''
        if not (0 <= threshold <= 1):
            raise ValueError("Threshold должен быть в диапазоне от 0 до 1")
        if not strategy in ['median', 'mean', 'mode']:
            raise ValueError("Method должен быть 'median', 'mean' или 'mode'")
        
        # вычислим долю пропусков, берем данные из исходного raw_df, чтобы избежать накопления удалений
        missing_ratio = self.df.isnull().mean()

        # сохраняем имена колонок, чтобы в тесте удалить те же самые, если % изменится
        self.params['cols_to_keep'] = missing_ratio[missing_ratio <= threshold].index.tolist()

        # cоздаем копию из колонок
        self.df = self.df[self.params['cols_to_keep']]
        
        # выбор типа заполнения
        for col in self.df.columns:
                # проверяем на числа 
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    if strategy == 'median':
                        val = self.df[col].median()
                    elif strategy == 'mean':
                        val = self.df[col].mean()
                    else: # mode
                        mode_res = self.df[col].mode()
                        val = mode_res[0] if not mode_res.empty else 0
                else:
                # для категорий всегда мода
                    mode_res = self.df[col].mode()
                    val = mode_res[0] if not mode_res.empty else "Unknown"

                # сохраняем значения во избежание утечки данных
                self.params['impute_values'][col] = val
                # заполняем пропуски и сохраняем
                self.df[col] = self.df[col].fillna(val)      
                
        return self.df
    

    def normalize_numeric(self, method: str='minmax'):
        '''
        нормализует числовые столбцы
        '''
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.params['norm_method'] = method

        for col in numeric_cols:
            if self.df[col].nunique() <= 2: continue # бинарные признаки не трогаем
            stats = {
                'min': self.df[col].min(),
                'max': self.df[col].max(),
                'mean': self.df[col].mean(),
                'std': self.df[col].std()
            }
            self.params['scaling_stats'][col] = stats
        
            if (method=='minmax'): #применяется Min-Max нормализация
                denom = stats['max'] - stats['min']
                self.df[col] = (self.df[col] - stats['min']) / (denom if denom != 0 else 1) # защита от деления на ноль
                
            elif (method=='std'): # применяется стандартизация
                self.df[col] = (self.df[col] - stats['mean']) / (stats['std'] if stats['std'] != 0 else 1)
            else:
                raise ValueError("Метод должен быть 'minmax' или 'std'")
        
        return self.df 


    def encode_categorical(self):
        '''
        выполняет one-hot encoding всех строковых
        (категориальных) столбцов
        '''

        # находим категориальные признаки
        self.params['cat_columns'] = self.df.select_dtypes(include= ['object', 'category']).columns.tolist()

        if  self.params['cat_columns']:
            self.df = pd.get_dummies(self.df, columns=self.params['cat_columns'], drop_first=True, dtype=int)
            # drop_first = true во избежании ловушки мультиколлинеарности

        # сохраняем список
        self.params['final_columns'] = self.df.columns.tolist()
        self._is_fitted = True
        return self.df    


    def fit_transform(self, threshold: float = 0.5, method: str = 'minmax', strategy: str = 'median'):
        '''
        Последовательно применяет все преобразования
        '''

        self.remove_missing(threshold=threshold, strategy=strategy)
        self.normalize_numeric(method=method)
        self.encode_categorical()
        
        
        self._is_fitted = True
        return self.df
    
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Применяет сохраненные правила к новым данным
        '''

        if not self._is_fitted:
            raise RuntimeError("Сначала вызовите fit_transform на обучающих данных")
        
        res = df.copy()

        # оставляем только нужные колонки
        res = res.reindex(columns=self.params['cols_to_keep'])
        cols_to_keep = [c for c in self.params['cols_to_keep'] if c in res.columns]
        res = res[cols_to_keep]

        # заполняем пропуски значениями из тренировочного набора
        for col, val in self.params['impute_values'].items():
            if col in res.columns:
                res[col] = res[col].fillna(val)

        # нормализуем по параметрам из тренировочного набор
        method = self.params['norm_method']
        for col, stats in self.params['scaling_stats'].items():
            if col in res.columns:
                if method == 'minmax':
                    denom = stats['max'] - stats['min']
                    res[col] = (res[col] - stats['min']) / (denom if denom != 0 else 1)
                elif method == 'std':
                    res[col] = (res[col] - stats['mean']) / (stats['std'] if stats['std'] != 0 else 1)

        # кодируем категории
        if self.params['cat_columns']:
            # Кодируем только те, что есть в датасете
            cat_cols_present = [c for c in self.params['cat_columns'] if c in res.columns]
            if cat_cols_present:
                res = pd.get_dummies(res, columns=cat_cols_present, drop_first=True, dtype=int)

        # выравниваем структуру, если какой-то категории не будет
        res = res.reindex(columns=self.params['final_columns'], fill_value=0)
        return res