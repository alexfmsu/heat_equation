# heat_equation_2D
**Задача 1**<br/>
Численное решение двумерного нестационарного неоднородного уравнения теплопроводности в прямоугольной области. Следует использовать неявную разностную схему, метод переменных направлений, блочную прогонку.
[Решение двумерного уравнения теплопроводности](http://lira.imamod.ru/msu201702/sem10_Y01.pdf)
##
**Задача 2**<br/>
Численное решение двумерного нестационарного неоднородного уравнения теплопроводности в прямоугольной области, с вырезами, заданными набором произвольно расположенных прямоугольников, расчет в которых не производится. Следует использовать явную разностную схему. Обратить внимание на необходимость использования метода равномерного распределения по процессам именно тех узлов расчетной области, в которых требуется выполнять расчет (вырезанные узлы в расчете не участвуют).<br/><br/>
Пример расчетной области (расчетная область выделена штриховкой):<br/>
(/area.jpg)
##
**Общие требования к выполненным заданиям**<br/><br/>
**Требования к программе:**
1. Программа должна быть гибридной: одновременно использовать технологию MPI, для обеспечения взаимодействия вычислительных узлов, и одну из двух технологий posix threads или OpenMP, для взаимодействия процессов, запущенных на ядрах процессоров<br/>
2. Программа должна демонстрировать эффективность не менее 80%, на числе вычислительных ядер, не менее 512<br/><br/>

**Отчет должен содержать:**
1. Постановку задачи
2. Описание точного аналитического решения
3. Описание метода решения и способа декомпозиции области
4. Описание используемой вычислительной системы (число узлов, процессоров, ядер, вид интерконнекта, …)
5. Аналитическую зависимость ожидаемого времени решения от параметров задачи и от параметров вычислительной системы
6. Сведения о значении ошибки (отклонения от точного решения) при выполнении расчетов на последовательности сгущающихся сеток
7. Таблицы и графики, содержащие сведения о размерах сеток, времени решения и эффективности распараллеливания
8. Анализ полученных результатов
9. Другие требуемые, с вашей точки зрения, материалы
10. Приложение (тексты программ: последовательной и параллельной)
