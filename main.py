import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ============================================================
# 1. СОЗДАНИЕ И ОБРАБОТКА МАССИВОВ
# ============================================================

def create_vector():
    """
    Создать массив от 0 до 9.

    Returns:
        numpy.ndarray: Массив чисел от 0 до 9 включительно
    """
    return np.arange(10)


def create_matrix():
    """
    Создать матрицу 5x5 со случайными числами [0,1].

    Returns:
        numpy.ndarray: Матрица 5x5 со случайными значениями от 0 до 1
    """
    return np.random.rand(5, 5)


def reshape_vector(vec):
    """
    Преобразовать (10,) -> (2,5)

    Args:
        vec (numpy.ndarray): Входной массив формы (10,)

    Returns:
        numpy.ndarray: Преобразованный массив формы (2, 5)

    Raises:
        ValueError: Если входной массив не имеет размер 10
    """
    if vec.size != 10:
        raise ValueError(f"Вектор должен содержать 10 элементов, получено {vec.size}")

    return vec.reshape(2, 5)



def transpose_matrix(mat):
    """
    Транспонирование матрицы.

    Args:
        mat (numpy.ndarray): Входная матрица

    Returns:
        numpy.ndarray: Транспонированная матрица
    """

    return mat.T


# ============================================================
# 2. ВЕКТОРНЫЕ ОПЕРАЦИИ
# ============================================================

def vector_add(a, b):
    """
    Сложение векторов одинаковой длины.

    Args:
        a (numpy.ndarray): Первый вектор
        b (numpy.ndarray): Второй вектор

    Returns:
        numpy.ndarray: Результат поэлементного сложения

    Raises:
        ValueError: Если векторы имеют разную форму
    """

    if a.shape != b.shape:
        raise ValueError(f"Векторы должны быть одинаковой формы: {a.shape} vs {b.shape}")

    return a + b


def scalar_multiply(vec, scalar):
    """
    Умножение вектора на число.

    Args:
        vec (numpy.ndarray): Входной вектор
        scalar (float/int): Число для умножения

    Returns:
        numpy.ndarray: Результат умножения вектора на скаляр

    Raises:
        TypeError: Если scalar не является числом
    """

    if not isinstance(scalar, (int, float)):
        raise TypeError(f"Скаляр должен быть числом, получен {type(scalar)}")

    return vec * scalar


def elementwise_multiply(a, b):
    """
    Поэлементное умножение.

    Args:
        a (numpy.ndarray): Первый вектор/матрица
        b (numpy.ndarray): Второй вектор/матрица

    Returns:
        numpy.ndarray: Результат поэлементного умножения

    Raises:
        ValueError: Если массивы имеют разную форму
    """
    if a.shape != b.shape:
        raise ValueError(f"Массивы должны быть одинаковой формы: {a.shape} vs {b.shape}")

    return a * b


def dot_product(a, b):
    """
    Скалярное произведение.

    Args:
        a (numpy.ndarray): Первый вектор
        b (numpy.ndarray): Второй вектор

    Returns:
        float: Скалярное произведение векторов

    Raises:
        ValueError: Если векторы не являются 1D или имеют разную длину
    """

    if a.ndim != 1 or b.ndim != 1:
        raise ValueError(f"Ожидались одномерные векторы, получены размерности {a.ndim} и {b.ndim}")

    if a.shape != b.shape:
        raise ValueError(f"Векторы должны быть одинаковой длины: {a.shape} vs {b.shape}")

    return np.dot(a, b)


# ============================================================
# 3. МАТРИЧНЫЕ ОПЕРАЦИИ
# ============================================================

def matrix_multiply(a, b):
    """
    Умножение матриц.

    Args:
        a (numpy.ndarray): Первая матрица
        b (numpy.ndarray): Вторая матрица

    Returns:
        numpy.ndarray: Результат умножения матриц

    Raises:
        ValueError: Если матрицы имеют несовместимые размерности для умножения
    """

    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Несовместимые размерности для умножения: {a.shape} и {b.shape}")

    return a @ b


def matrix_determinant(a):
    """
    Определитель матрицы.

    Args:
        a (numpy.ndarray): Квадратная матрица

    Returns:
        float: Определитель матрицы

    Raises:
        ValueError: Если матрица не является квадратной
    """

    if a.shape[0] != a.shape[1]:
        raise ValueError(f"Матрица должна быть квадратной, получена форма {a.shape}")

    return np.linalg.det(a)


def matrix_inverse(a):
    """
    Обратная матрица.

    Args:
        a (numpy.ndarray): Квадратная матрица

    Returns:
        numpy.ndarray: Обратная матрица

    Raises:
        ValueError: Если матрица не является квадратной или если матрица вырождена
    """

    if a.shape[0] != a.shape[1]:
        raise ValueError(f"Матрица должна быть квадратной, получена форма {a.shape}")

    det = np.linalg.det(a)
    if np.abs(det) < 1e-10:
        raise ValueError(f"Матрица вырождена (определитель = {det:.2e}). Обратной матрицы не существует.")

    return np.linalg.inv(a)


def solve_linear_system(a, b):
    """
    Решить систему Ax = b

    Args:
        a (numpy.ndarray): Матрица коэффициентов A
        b (numpy.ndarray): Вектор свободных членов b

    Returns:
        numpy.ndarray: Решение системы x

    Raises:
        ValueError: Если матрица A не квадратная, размерности несовместимы или система не имеет решения
    """

    if a.shape[0] != a.shape[1]:
        raise ValueError(f"Матрица A должна быть квадратной, получена форма {a.shape}")

    if a.shape[0] != b.shape[0]:
        raise ValueError(f"Несовместимые размерности: A {a.shape} и b {b.shape}")

    try:
        return np.linalg.solve(a, b)
    except Exception as e:
        raise ValueError(f"Система не имеет решения или имеет бесконечно много решений: {e}")


# ============================================================
# 4. СТАТИСТИЧЕСКИЙ АНАЛИЗ
# ============================================================

def load_dataset(path="data/students_scores.csv"):
    """
    Загрузить CSV и вернуть NumPy массив.

    Args:
        path (str): Путь к CSV файлу

    Returns:
        numpy.ndarray: Загруженные данные в виде массива
    """

    try:
        return pd.read_csv(path).to_numpy()
    except Exception as e:
        raise Exception(f"Ошибка при загрузке файла {path}: {e}")


def statistical_analysis(data):
    """
    Статистический анализ данных.

    Args:
        data (numpy.ndarray): Одномерный массив данных

    Returns:
        dict: Словарь со статистическими показателями

    Raises:
        ValueError: Если массив пустой
    """

    if data.size == 0:
        raise ValueError("Массив данных пустой")

    stats = {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        '25_percentile': np.percentile(data, 25),
        '75_percentile': np.percentile(data, 75)
    }

    return stats


def normalize_data(data):
    """
    Min-Max нормализация.

    Формула: (x - min) / (max - min)

    Args:
        data (numpy.ndarray): Входной массив данных

    Returns:
        numpy.ndarray: Нормализованный массив данных в диапазоне [0, 1]

    Raises:
        ValueError: Если массив пустой или все значения одинаковы
    """

    if data.size == 0:
        raise ValueError("Массив данных пустой")

    min_val = np.min(data)
    max_val = np.max(data)

    if np.abs(max_val - min_val) < 1e-10:
        raise ValueError(f"Невозможно нормализовать: все значения одинаковы")

    return (data - min_val) / (max_val - min_val)


# ============================================================
# 5. ВИЗУАЛИЗАЦИЯ
# ============================================================

def plot_histogram(data, title="Распределение оценок по математике",
                   xlabel="Оценки", ylabel="Частота", filename="math_scores_histogram.png"):
    """
    Построить гистограмму распределения оценок по математике.

    Args:
        data (numpy.ndarray): Данные для гистограммы
        title (str): Заголовок графика
        xlabel (str): Подпись оси X
        ylabel (str): Подпись оси Y
        filename (str): Имя файла для сохранения

    Raises:
        ValueError: Если массив пустой
    """

    if data.size == 0:
        raise ValueError("Массив данных пустой")

    if not os.path.exists('plots'):
        os.makedirs('plots')

    plt.figure(figsize=(10, 6))

    plt.hist(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    mean_val = np.mean(data)

    plt.axvline(mean_val, color='red', label=f'Среднее: {mean_val:.2f}')
    plt.legend()

    plt.savefig(f'plots/{filename}')
    plt.close()


def plot_heatmap(matrix, title="Матрица корреляции предметов",
                 filename="correlation_heatmap.png"):
    """
    Построить тепловую карту корреляции предметов.

    Args:
        matrix (numpy.ndarray): Матрица корреляции
        title (str): Заголовок графика
        filename (str): Имя файла для сохранения
    """

    if not os.path.exists('plots'):
        os.makedirs('plots')

    plt.figure(figsize=(12, 10))

    # Настраиваем тепловую карту
    sns.heatmap(matrix, annot=True, fmt='.2f', linewidths=1, cbar_kws={"shrink": 0.8})

    plt.title(title)
    plt.savefig(f'plots/{filename}')
    plt.close()


def plot_line(x, y, title="Зависимость оценки от номера студента",
              xlabel="Номер студента", ylabel="Оценка", filename="student_scores.png"):
    """
    Построить график зависимости.

    Args:
        x (numpy.ndarray): Значения по оси X
        y (numpy.ndarray): Значения по оси Y
        title (str): Заголовок графика
        xlabel (str): Подпись оси X
        ylabel (str): Подпись оси Y
        filename (str): Имя файла для сохранения
    """

    if not os.path.exists('plots'):
        os.makedirs('plots')

    plt.figure(figsize=(14, 6))

    plt.plot(x, y, label='Оценки')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.margins(x=0.02)

    plt.savefig(f'plots/{filename}')
    plt.show()
    plt.close()
