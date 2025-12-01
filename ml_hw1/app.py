import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle
import random
import io
import os

# from ydata_profiling import ProfileReport
# from streamlit_pandas_profiling import st_profile_report
from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score, mean_squared_error as MSE
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import Lasso
# from sklearn.model_selection import GridSearchCV
# from sklearn.linear_model import ElasticNet
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.linear_model import Ridge

random.seed(42)
np.random.seed(42)
RANDOM_STATE=42

st.title("Обучение модели регрессии для предсказания стоимости автомобилей")
st.header("EDA и предобработка данных")

df_train, df_test = None, None

upload_file = st.file_uploader("Загрузите датасет без разбиения для EDA", type=["csv"], key="eda_file")
if upload_file is not None:
    df = pd.read_csv(upload_file)
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=RANDOM_STATE)

st.text('Или загрузите тренировочный и тестовый датасет отдельно')
df_train_uploaded = st.file_uploader("Загрузите тренировочный датасет", type=["csv"])
df_test_uploaded = st.file_uploader("Загрузите тестовый датасет", type=["csv"])
if df_train_uploaded is not None and df_test_uploaded is not None:
    df_train = pd.read_csv(df_train_uploaded)
    df_test = pd.read_csv(df_test_uploaded)

st.text('Или используйте датасет по-умолчанию')
button_1 = st.button("Загрузить существующий датасет", key="1")
if button_1:
    df_train = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv')
    df_test = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_test.csv')

if df_train is not None and df_test is not None:
    st.text('Размеры учебного и тестового датасета:')
    st.text(f"Train data shape: {df_train.shape}")
    st.text(f"Test data shape: {df_test.shape}")

    st.text('30 случайных строк тренировочного датасета')
    st.dataframe(df_train.sample(30))

    st.text('Первые 5 объектов тестового датасета')
    st.dataframe(df_test.head())

    st.text('Последние 5 объектов тестового датасета')
    st.dataframe(df_test.tail())

    df_train_isna = df_train.isna().sum()[df_train.isna().sum() != 0] # делаем таблицу сумм пропусков и берем в нее только те столбцы, в которых сумма пропусков не равна 0 (срез таблицы)
    df_train_isna_columns = df_train.isna().sum()[df_train.isna().sum() != 0].index.tolist() # тут тоже самое, только берем индексы (названия колонок трейна) и преобразуем все в список

    st.text('Пропуски, выявленные в тренировочном датасете')
    for col, name in zip(st.columns(len(df_train_isna_columns)), df_train_isna_columns):
        with col:
            st.write(f'Колонка: {name}')
            st.write(f'кол-во пропусков: {df_train_isna.loc[name]}')

    df_test_isna = df_test.isna().sum()[df_test.isna().sum() != 0]
    df_test_isna_columns = df_test.isna().sum()[df_test.isna().sum() != 0].index.tolist()

    st.text('Пропуски, выявленные в тестовом датасете')
    for col, name in zip(st.columns(len(df_test_isna_columns)), df_test_isna_columns):
        with col:
            st.write(f'Колонка: {name}')
            st.write(f'кол-во пропусков: {df_test_isna.loc[name]}')

    st.write(f'Количество явных дубликатов в тренировочном датасете: {df_train.duplicated().sum()}')
    st.write(f'Количество явных дубликатов в тестовом датасете: {df_test.duplicated().sum()}')


    df_train['mileage'] = df_train['mileage'].apply(lambda x: x.split()[0] if not isinstance(x, float) else x).astype(float) # отбрасываем единицы измерения, если не float (т.е. если не NaN)
    df_test['mileage'] = df_test['mileage'].apply(lambda x: x.split()[0] if not isinstance(x, float) else x).astype(float) # далее и ниже аналогично для остальных столбцов

    mileage_median = df_train['mileage'].median()

    df_train['mileage'] = df_train['mileage'].fillna(mileage_median)
    df_test['mileage'] = df_test['mileage'].fillna(mileage_median)


    df_train['engine'] = df_train['engine'].apply(lambda x: x.split()[0] if not isinstance(x, float) else x).astype(float)
    df_test['engine'] = df_test['engine'].apply(lambda x: x.split()[0] if not isinstance(x, float) else x).astype(float)

    engine_median = df_train['engine'].median()

    df_train['engine'] = df_train['engine'].fillna(engine_median)
    df_test['engine'] = df_test['engine'].fillna(engine_median)


    df_train['max_power'] = df_train['max_power'].apply(lambda x: x.split()[0].strip() if not isinstance(x, float) else x) # здесь сразу не удалось преобразовать к float
    df_train['max_power'] = df_train['max_power'].apply(lambda x: 0 if x == 'bhp' else x).astype(float) # поэтому заменяем строку с пропуском числового значения'bhp' на 0

    df_test['max_power'] = df_test['max_power'].apply(lambda x: x.split()[0].strip() if not isinstance(x, float) else x)
    df_test['max_power'] = df_test['max_power'].apply(lambda x: 0 if x == 'bhp' else x).astype(float)

    max_power_median = df_train['max_power'].median()

    df_train['max_power'] = df_train['max_power'].fillna(max_power_median)
    df_test['max_power'] = df_test['max_power'].fillna(max_power_median)

    df_train['max_power'] = df_train['max_power'].apply(lambda x: max_power_median if x == 0 else x)
    df_test['max_power'] = df_test['max_power'].apply(lambda x: max_power_median if x == 0 else x)


    seats_median = df_train['seats'].median()

    df_train['seats'] = df_train['seats'].fillna(seats_median)
    df_test['seats'] = df_test['seats'].fillna(seats_median)
        

    df_train_isna = df_train.isna().sum()[df_train.isna().sum() != 0] # делаем таблицу сумм пропусков и берем в нее только те столбцы, в которых сумма пропусков не равна 0 (срез таблицы)
    df_train_isna_columns = df_train.isna().sum()[df_train.isna().sum() != 0].index.tolist() # тут тоже самое, только берем индексы (названия колонок трейна) и преобразуем все в список

    st.write(f'Пропуски, вявленные в тренировочном датасете после запонения медианой')
    for col, name in zip(st.columns(len(df_train_isna_columns)), df_train_isna_columns):
        with col:
            st.write(f'Колонка: {name}')
            st.write(f'кол-во пропусков: {df_train_isna.loc[name]}')

    df_test_isna = df_test.isna().sum()[df_test.isna().sum() != 0]
    df_test_isna_columns = df_test.isna().sum()[df_test.isna().sum() != 0].index.tolist()

    st.text('Пропуски, выявленные в тестовом датасете после заполнения медианой')
    for col, name in zip(st.columns(len(df_test_isna_columns)), df_test_isna_columns):
        with col:
            st.write(f'Колонка: {name}')
            st.write(f'кол-во пропусков: {df_test_isna.loc[name]}')

    # st.subheader("Автоотчет средствами YData Profiling") 
    # profile = ProfileReport(df_train, title="Profiling Report")
    # profile.to_file("report.html")

    # with open("report.html", "r", encoding="utf-8") as f:
        # html_content = f.read()

    # st.components.v1.html(html_content, height=500, scrolling=True)


    duplicated_objects = df_train[df_train.columns.difference(['selling_price'])].duplicated()
    st.write(f'Объектов с одинаковым признаковым описанием в тренировочном датафрейме: {duplicated_objects.sum()}')
    st.dataframe(df_train[duplicated_objects])

    df_train.drop_duplicates(subset=df_train.columns.difference(['selling_price']), keep='first', inplace=True)
    df_train.reset_index(drop=True, inplace=True)
    st.write("Данные объекты удалены")

    df_train[['engine', 'seats']] = df_train[['engine', 'seats']].astype(int)
    df_test[['engine', 'seats']] = df_test[['engine', 'seats']].astype(int)

    df_train.drop(['torque', 'name'], axis=1, inplace=True)
    df_test.drop(['torque', 'name'], axis=1, inplace=True)

    st.subheader('Информация о тренировочном и тестовом датасетах после «косметической» предобработки')
    st.text('Столбцы torque и name удалены')
    buffer = io.StringIO()
    df_train.info(buf=buffer)
    train_info = buffer.getvalue()

    buffer = io.StringIO()
    df_test.info(buf=buffer)
    test_info = buffer.getvalue()

    col1, col2 = st.columns(2)
    with col1:
        st.text("Информация о тренировочных данных")
        st.text(train_info)
    
    with col2:
        st.text("Информация о тестовых данных")
        st.text(test_info)


    st.subheader('Информация об основных статистиках')
    st.text('Основные статистики для трейна по числовым столбцам')
    st.dataframe(df_train.describe(include='number').apply(lambda x: x.apply('{0:.2f}'.format)))
    
    st.text('Основные статистики для трейна по категориальным столбцам')
    st.dataframe(df_train.describe(include='object'))

    st.text('Основные статистики для теста по числовым столбцам')
    st.dataframe(df_test.describe(include='number').apply(lambda x: x.apply('{0:.2f}'.format)))

    st.text('Основные статистики для теста по категориальным столбцам')
    st.dataframe(df_test.describe(include='object'))
    
    if button_1:
        # В столбце mileage заменим ошибочные нулевые значения на медиану
        df_train['mileage'] = df_train['mileage'].apply(lambda x: mileage_median if x == 0 else x)
        df_test['mileage'] = df_test['mileage'].apply(lambda x: mileage_median if x == 0 else x)

        # Также на медиану заменим аномальный пробег автомобиля в столбце km_driven
        km_driven_median = df_train['km_driven'].median()
        df_train['km_driven'] = df_train['km_driven'].apply(lambda x: km_driven_median if x == 2360457.00 else x)

        st.write('В существующем датасете произведены замены:')
        st.write('- в столбце mileage заменим ошибочные нулевые значения на медиану')
        st.write('- также на медиану заменим аномальный пробег автомобиля в столбце km_driven')

    st.header("Инфографика")
    st.subheader("Pairplot визуализации")

    sns.set_theme(style="whitegrid")
    pairplot = sns.pairplot(df_train, hue="selling_price", palette="viridis", diag_kind="kde", corner=True)
    pairplot.fig.suptitle("Визуализация попарных распределений числовых признаков\nпо стоимости при продаже (трейн)", y=1.02, fontsize=14)
    st.pyplot(pairplot.fig)

    sns.set_theme(style="whitegrid")
    pairplot = sns.pairplot(df_test, hue="selling_price", palette="viridis", diag_kind="kde", corner=True)
    pairplot.fig.suptitle("Визуализация попарных распределений числовых признаков\nпо стоимости при продаже (тест)", y=1.02, fontsize=14)
    st.pyplot(pairplot.fig)


    st.subheader("Тепловая карта корреляций")
    corr_matrix = df_train.select_dtypes(include='number').corr()

    plt.figure(figsize=(8, 8))
    sns.set_style("white")
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    heatmap = sns.heatmap(corr_matrix, 
                            mask=mask, annot=True, 
                            fmt=".2f", 
                            cmap="coolwarm", 
                            vmin=-1, 
                            vmax=1, 
                            square=True)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    fig = plt.gcf()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Зависимости категориальных переменных и целевой переменной")
    df_category_target = df_train[['fuel', 'seller_type', 'transmission', 'owner', 'selling_price']]

    plt.figure(figsize=(10, 8))
    sns.set_style("white")
    sns.scatterplot(data=df_category_target, y='fuel', x='selling_price')
    plt.title("Зависимость стоимости автомобиля от типа топлива", y=1.02, fontsize=14)
    fig = plt.gcf()
    st.pyplot(fig)
    plt.close(fig)

    plt.figure(figsize=(10, 8))
    sns.set_style("white")
    sns.scatterplot(data=df_category_target, y='seller_type', x='selling_price')
    plt.title("Зависимость стоимости автомобиля от типа продавца", y=1.02, fontsize=14)
    fig = plt.gcf()
    st.pyplot(fig)
    plt.close(fig)

    plt.figure(figsize=(10, 2))
    sns.set_style("white")
    sns.scatterplot(data=df_category_target, y='transmission', x='selling_price')
    plt.title("Зависимость стоимости автомобиля от типа коробки передач", y=1.02, fontsize=14)
    fig = plt.gcf()
    st.pyplot(fig)
    plt.close(fig)

    plt.figure(figsize=(10, 2))
    sns.set_style("white")
    sns.scatterplot(data=df_category_target, y='owner', x='selling_price')
    plt.title("Зависимость стоимости автомобиля от количества владельцев", y=1.02, fontsize=14)
    fig = plt.gcf()
    st.pyplot(fig)
    plt.close(fig)

    if 'medians_saved' not in st.session_state:
        st.session_state.mileage_median = mileage_median
        st.session_state.engine_median = engine_median
        st.session_state.max_power_median = max_power_median
        st.session_state.seats_median = seats_median
        st.session_state.km_driven_median = km_driven_median
        st.session_state.medians_saved = True

st.header("Применение обученной модели регрессии к новым данным для предсказания стоимости автомобиля")
if 'medians_saved' not in st.session_state:
        st.error("Сначала выполните EDA анализ для вычисления медиан!")
else:
    @st.cache_resource
    def load_model():
        path = os.path.join(os.path.dirname(__file__), 'data', 'car_price_models_complete.pkl')
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model

    def preprocessing(df_preprocess, 
                        mileage_median=st.session_state.mileage_median, 
                        engine_median=st.session_state.engine_median, 
                        max_power_median=st.session_state.max_power_median, 
                        seats_median=st.session_state.seats_median, 
                        km_driven_median=st.session_state.km_driven_median):
        
        df_preprocess['km_driven'] = df_preprocess['km_driven'].apply(lambda x: km_driven_median if x == 2360457.00 else x)

        df_preprocess['mileage'] = (df_preprocess['mileage'].apply(lambda x: x.split()[0] if not isinstance(x, float) else x).astype(float))
        df_preprocess['mileage'] = df_preprocess['mileage'].fillna(mileage_median)
        df_preprocess['mileage'] = df_preprocess['mileage'].apply(lambda x: mileage_median if x == 0 else x)

        df_preprocess['engine'] = df_preprocess['engine'].apply(lambda x: x.split()[0] if not isinstance(x, float) else x).astype(float)
        df_preprocess['engine'] = df_preprocess['engine'].fillna(engine_median)
        
        df_preprocess['max_power'] = df_preprocess['max_power'].apply(lambda x: x.split()[0].strip() if not isinstance(x, float) else x)
        df_preprocess['max_power'] = df_preprocess['max_power'].apply(lambda x: 0 if x == 'bhp' else x).astype(float) 

        df_preprocess['max_power'] = df_preprocess['max_power'].fillna(max_power_median)
        df_preprocess['max_power'] = df_preprocess['max_power'].apply(lambda x: max_power_median if x == 0 else x)

        df_preprocess['seats'] = df_preprocess['seats'].fillna(seats_median)
        
        df_preprocess[['engine', 'seats']] = df_preprocess[['engine', 'seats']].astype(int)

        df_preprocess.drop(['torque', 'name'], axis=1, inplace=True)

        return df_preprocess

    up_file = st.file_uploader("Загрузите датасет для предсказания стоимости", type=["csv"], key="pred_file")
    if up_file is not None:
        df_predict = pd.read_csv(up_file)

        st.text('Первые строки вашего датасета до предобработки')
        df_predict.drop(['Unnamed: 0'], axis=1, inplace=True)
        st.dataframe(df_predict.head())

        
        predict_model = load_model()
        best_model = predict_model['models']['grid_search_ridge_reg'].best_estimator_

        df_prepr = df_predict.copy()
        df_prepr = preprocessing(df_prepr)


        numerical_cols = df_prepr.select_dtypes(include='number').columns.tolist()
        scaler_2 = predict_model['scalers']['numerical_scaler_2']
        df_prepr[numerical_cols] = scaler_2.transform(df_prepr[numerical_cols])


        st.text('Первые строки вашего датасета после предобработки')
        st.dataframe(df_prepr.head())


        encoder = predict_model['encoders']['onehot_encoder']
        cat_cols = predict_model['encoders']['categorical_columns']

        df_cat = encoder.transform(df_prepr[cat_cols])
        df_cat = pd.DataFrame(df_cat, columns=encoder.get_feature_names_out(cat_cols))

        df_num = df_prepr.drop(columns=cat_cols)

        df_prepr = pd.concat([df_num.reset_index(drop=True), df_cat.reset_index(drop=True)], axis=1)


        predict = best_model.predict(df_prepr)

        st.write('Предсказанные стоимости автомобилей')
        df_predict['selling_price'] = predict
        st.dataframe(df_predict)


st.header("Визуализация весов моделей")

import streamlit as st
import matplotlib.pyplot as plt
import pickle
import pandas as pd

st.header("Визуализация весов модели")
@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(__file__), 'data', 'car_price_models_complete.pkl')
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

if st.button("Показать веса модели"):
    model = load_model()
    weights = list(model['models']['grid_search_ridge_reg'].best_estimator_.coef_)
    clumns = list(model['columns'])

    fig, ax = plt.subplots(figsize=(15, 10))

    bars = ax.bar(clumns, weights, color=['green' if w >= 0 else 'red' for w in weights], alpha=0.7)

    ax.set_title('Веса модели', fontsize=16)
    ax.set_xlabel('Признаки', fontsize=12)
    ax.set_ylabel('Веса', fontsize=12)
    ax.set_xticklabels(clumns, rotation=45, ha='right', fontsize=10)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    for i, bar in enumerate(bars):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01 if h >= 0 else h - 0.01, 
                f"{h:.3f}", ha='center', va='bottom' if h >= 0 else 'top', fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)





