Формируется матрица F следующим образом: скопировать в нее А и если в В количество простых чисел в нечетных столбцах больше, 
чем сумма чисел в четных строках, то поменять местами В и Е симметрично, иначе С и Е поменять местами несимметрично. 
При этом матрица А не меняется. После чего если определитель матрицы А больше суммы диагональных элементов матрицы F, 
то вычисляется выражение: A-1*AT – K * F-1, иначе вычисляется выражение (A-1 +G-F-1)*K, где G-нижняя треугольная матрица, полученная из А. 
Выводятся по мере формирования А, F и все матричные операции последовательно.

В	С

D	Е
