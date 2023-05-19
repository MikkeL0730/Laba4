# 12.	Формируется матрица F следующим образом: скопировать в нее А и если в В количество простых чисел в нечетных
# столбцах больше, чем сумма чисел в четных строках, то поменять местами В и Е симметрично, иначе С и Е поменять местами
# несимметрично. При этом матрица А не меняется.После чего если определитель матрицы А больше суммы диагональных
# элементов матрицы F, то вычисляется выражение:A-1*AT–K * F-1, иначе вычисляется выражение (A-1+G-F-1)*K, где G-нижняя
# треугольная матрица, полученная из А.Выводятся по мере формирования А, F и все матричные операции последовательно.
# B C
# D E


import numpy as np
import matplotlib.pyplot as plt

def Prostoe(k):  # Простое или нет
    if k < 0:
        k *= -1
    if k == 2 or k == 3: return True
    if k % 2 == 0 or k < 2: return False
    for i in range(3, int(k ** 0.5) + 1, 2):
        if k % i == 0:
            return False

    return True

K = int(input('Введите K: '))
N = int(input('Введите N (больше 5): '))
if N < 6:
    print("Число N слишком малое. Введите N >= 6")
    exit()
# Формируем матрицу А размерности N на N числами от -10 до 10
A = np.random.randint(-10, 11, size=(N, N))

n = N // 2

w = N // 2
if N % 2 == 0:
    B = A[0:w, 0:w]
    C = A[0:w, w:]
    E = A[w:, w:]
    D = A[w:, 0:w]
else:
    B = A[0:w, 0:w]
    C = A[0:w, w + 1:]
    E = A[w + 1:, w + 1:]
    D = A[w + 1:, 0:w]

# печатаем матрицы E, B, C, D, A
print('Матрица E:')
print(E)

print('Матрица B:')
print(B)

print('Матрица C:')
print(C)

print('Матрица D:')
print(D)

print('Матрица A:')
print(A)

# считаем в матрице B количество простых чисел в нечетных столбцах
col_prost_B = 0
for col in range(1, n, 2):
    for row in range(n):
        if Prostoe(B[row][col]):
            col_prost_B += 1
print(f'количество в матрице B простых чисел в нечетных столбцах: {col_prost_B}')

# считаем сумма чисел в четных строках в матрице B
col_sum_B = 0
for row in range(1,n,2):
    for col in range (n):
        col_sum_B += B[row][col]
print(f'сумма чисел в четных строках в матрице B: {col_sum_B}')

F = A.copy()
col_prost_B = col_sum_B + 1
if col_prost_B > col_sum_B:
    print('количество в матрице B простых чисел в нечетных столбцах больше, чем сумма чисел в четных строках в матрице'
          ' B значит меняем местами В и Е симметрично')
    if N % 2 == 0:
        F[0:w, 0:w] = np.flipud(E)
        F[w:, w:] = np.flipud(B)
    else:
        F[0:w, 0:w] = np.flipud(E)
        F[w + 1:, w + 1:] = np.flipud(B)
else:
    print('количество в матрице B простых чисел в нечетных столбцах меньше, чем сумма чисел в четных строках в матрице'
          ' B значит меняем местами C и Е несимметрично')
    if N % 2 == 0:
        F[0:w, w:] = E
        F[w:, w:] = C
    else:
        F[0:w, w + 1:] = E
        F[w + 1:, w + 1:] = C

print('Матрица F')
print(F)

det_A = np.linalg.det(A)  # определитель матрицы A
sum_diag = np.trace(F)  # сумма диагональных элементов матрицы F

if det_A > sum_diag:
    print("вычисляем выражение: A-1*AT–K * F-1")
    A_inv = np.linalg.inv(A)  # обратная матрица А
    AT = np.transpose(A)     # транспонированная матрица А
    F_inv = np.linalg.inv(F)   # обратная матрица F
    K_F_inv = np.dot(K, F_inv)   # обратная матрица F умноженная на K
    A_inv_AT = np.dot(AT, A_inv)  # обратная матрица А умноженаня на транспонированную матрицу А
    result = np.subtract(K_F_inv, A_inv_AT)
    print(result)
else:
    print("вычисляем выражение:(A-1+G-F-1)*K")
    G = np.tril(A)     # нижняя треугольная матрица А
    A_inv = np.linalg.inv(A)  # обратная матрица А
    F_inv = np.linalg.inv(F)  # обратная матрица F
    A_inv_G = np.add(A_inv, G)
    A_inv_G_F = np.subtract(A_inv_G, F_inv)
    result = np.dot(A_inv_G_F, K)
    print(result)

# работа с графиками
plt.figure(figsize=(16, 9))

# вывод тепловой карты матрицы F
plt.subplot(2, 2, 1)
plt.xticks(ticks=np.arange(F.shape[1]))
plt.yticks(ticks=np.arange(F.shape[1]))
plt.xlabel('Номер столбца')
plt.ylabel('Номер строки')
hm = plt.imshow(F, cmap='gist_rainbow', interpolation="nearest")
plt.colorbar(hm)
plt.title('Тепловая карта элементов матрицы')

# Вывод гистрограммы
plt.subplot(2, 2, 2)
plt.hist(F.flatten(), bins=20, color='red')
plt.xlabel('Значение')
plt.ylabel('Количество')
plt.title('Гистограмма элементов матрицы')

# Вывод круговой диаграммы
plt.subplot(2, 2, 3)
x = np.arange(F.shape[1])
P = []
for i in range(N):
    P.append(abs(F[0][i]))
plt.pie(P, labels=x, autopct='%1.2f%%')
plt.title("круговая диаграмма элементов матрицы")

plt.tight_layout(pad=2, w_pad=1, h_pad=1) # расстояние от границ и между областями
plt.suptitle("Использование библиотеки Matplotlib", y=1)
plt.show()