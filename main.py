import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE


# Функция для преобразования времени в минуты
def time_to_minutes(time_str):
    return time_str.hour * 60 + time_str.minute


# Функция для обработки смешанных цветов глаз
def process_eye_color(color):
    # Приведение к единому регистру и замена буквы "ё" на "е"
    color = color.lower().replace('ё', 'е').replace('коричневый', 'карий').replace('-', ' ').replace(' ', '').strip()
    # Словарь с базовыми цветами
    base_colors = ['голубой', 'зеленый', 'серый', 'карий', 'синий', 'болотный']
    # Разделяем на компоненты и создаем бинарные признаки
    color_components = color.split()  # В этом случае пробелы уже удалены
    return {f'Цвет глаз_{base_color.capitalize()}': 1 if base_color in color_components else 0 for base_color in
            base_colors}


# Загрузка данных
data = pd.read_excel('result.xlsx')

# Обработка бинарных признаков (label encoding)
binary_features = [
    'Курите ли Вы?',
    'Если поблизости с Вашим домом кофейня?',
    'Вы являетесь гурманом?',
    'Вы работаете из офиса?',
    'Вы домосед?',
    'Вы высыпаетесь?',
    'У Вас есть хронические заболевания?'
]

label_encoder = LabelEncoder()
for feature in binary_features:
    data[feature] = label_encoder.fit_transform(data[feature])

# Обработка цвета глаз
eye_color_features = data['Укажите цвет Вашего левого глаза'].apply(process_eye_color)
eye_color_df = pd.DataFrame(eye_color_features.tolist())

# Обрезаем значения больше 100 до 100
data['Много ли Вы испытываете стресса в жизни? Укажите число от 0 до 100'] = (
    data['Много ли Вы испытываете стресса в жизни? Укажите число от 0 до 100'].clip(0, 100))
data['Насколько здоровый образ жизни Вы ведете? \nУкажите число по шкале от 0 до 100'] = (
    data['Насколько здоровый образ жизни Вы ведете? \nУкажите число по шкале от 0 до 100'].clip(0, 100))

# Объединяем исходные данные с новыми признаками цвета глаз
data = pd.concat([data, eye_color_df], axis=1)
data.drop('Укажите цвет Вашего левого глаза', axis=1, inplace=True)

# Обработка многоклассовых категориальных признаков (one-hot encoding)
categorical_features = [
    'Укажите Ваш пол',
    'Какой напиток Вы предпочитаете утром?',
    'Укажите Ваш хронотип',
    'Какой рукой Вы пишите?',
    'Какой у Вас знак зодиака?'
]

data = pd.get_dummies(data, columns=categorical_features)

# Преобразование времени пробуждения в минуты
data['Во сколько Вы обычно просыпаетесь? Укажите время в формате "Час" и "Минуты"'] = (
    data['Во сколько Вы обычно просыпаетесь? Укажите время в формате "Час" и "Минуты"'].apply(time_to_minutes))

# Нормализация числовых признаков
numerical_features = [
    'Укажите Ваш возраст',
    'Насколько здоровый образ жизни Вы ведете? \nУкажите число по шкале от 0 до 100',
    'Много ли Вы испытываете стресса в жизни? Укажите число от 0 до 100',
    'Во сколько Вы обычно просыпаетесь? Укажите время в формате "Час" и "Минуты"',
    'Сколько Вы в среднем спите? Укажите среди время вашего сна'
]

scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Пример использования KNN
# Предположим, что целевая переменная — это предпочтение напитка
X = data.drop('Какой напиток Вы предпочитаете утром?_Кофе', axis=1)  # Удаляем целевую переменную из признаков
y = data['Какой напиток Вы предпочитаете утром?_Кофе']  # Целевая переменная (кофе)

# Балансировка классов
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=38)

# Оптимизация гиперпараметров KNN
param_grid = {'n_neighbors': range(1, 20)}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_knn = grid_search.best_estimator_

# Обучение улучшенной модели KNN
best_knn.fit(X_train, y_train)

# Оценка модели KNN
knn_accuracy = best_knn.score(X_test, y_test)
print(f'Optimized KNN Accuracy: {knn_accuracy * 100:.2f}%')

# Получаем предсказания для тестовой выборки
predictions_knn = best_knn.predict(X_test)

# Создаем DataFrame из тестовой выборки и предсказаний
results = pd.DataFrame(X_test)  # Берем тестовую выборку
results['Predicted KNN'] = predictions_knn  # Добавляем предсказания KNN
results['Actual'] = y_test.values  # Добавляем истинные значения

# Выводим результаты
print(results)
