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
        
        # если мы еще не знаем, какие колонки оставить, то вычисляем
        if self.params['cols_to_keep'] is None:
        # вычисляем долю пропусков, берем данные из исходного df, чтобы избежать накопления удалений
            missing_ratio = self.df.isnull().mean()
            # сохраняем имена колонок, чтобы в тесте удалить те же самые, если % изменится
            self.params['cols_to_keep'] = missing_ratio[missing_ratio <= threshold].index.tolist()

        # cоздаем копию из колонок
        self.df = self.df[self.params['cols_to_keep']]
        
        # выбор типа заполнения
        for col in self.df.columns:
            if col not in self.params['impute_values']:
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
                    val = self.df[col].mode()[0] if not self.df[col].mode().empty else "Unknown"
                self.params['impute_values'][col] = val

                # сохраняем значения во избежание утечки данные и заполняем пропуски
            self.df[col] = self.df[col].fillna(self.params['impute_values'][col])    
                
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

            if col not in self.params['scaling_stats']:
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
        if not self.params['cat_columns']:
            self.params['cat_columns'] = self.df.select_dtypes(include= ['object', 'category']).columns.tolist()

        if  self.params['cat_columns']:
            self.df = pd.get_dummies(self.df, columns=self.params['cat_columns'], drop_first=True, dtype=int)
            # drop_first = true во избежании ловушки мультиколлинеарности

        # сохраняем список
        if not self.params['final_columns']:
            self.params['final_columns'] = self.df.columns.tolist()
        else:
            # подгоняем новые данные под старую структуру
            self.df = self.df.reindex(columns=self.params['final_columns'], fill_value=0)    

        self._is_fitted = True
        return self.df    


    def fit_transform(self, threshold: float = 0.5, method: str = 'minmax', strategy: str = 'median'):
        '''
        ТЗ требует этот метод. Последовательно применяет все преобразования. 
        '''

        self.remove_missing(threshold=threshold, strategy=strategy)
        self.normalize_numeric(method=method)
        self.encode_categorical()
        
        self._is_fitted = True
        return self.df
    
    
    def transform(self, new_df: pd.DataFrame) -> pd.DataFrame:
        '''
        "Метод для новых данных, использующий сохраненные параметры.
        '''

        if not self._is_fitted:
            raise RuntimeError("Сначала вызовите fit_transform на обучающих данных")
        
        # чтобы не дублировать код, временно подменяем self.df и вызываем те же методы
        original_df = self.df
        self.df = new_df.copy()

        self.remove_missing() 
        self.normalize_numeric(method=self.params['norm_method'])
        self.encode_categorical()

        result = self.df
        self.df = original_df # возвращаем исходный df на место

        return result