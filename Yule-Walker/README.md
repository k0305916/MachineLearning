#MachineLearning


$\hat{r}_{xx} = \frac{1}{N-m}T \sum_{n=0}^{N-m-1} x[n+m]x^*[n] T$

$\check{r}_{xx} = \frac{1}{N}T \sum_{n=0}^{N-m-1} x[n+m]x^*[n] T$

$$\left( \begin{array}{cccc}
r_1 & r_2^* & \dots & r_{n}^*\\
r_2 & r_1^* & \dots & r_{n-1}^*\\
\dots & \dots & \dots & \dots\\
r_n & \dots & r_2 & r_1 \end{array} \right)
\left( \begin{array}{cccc}
a_2\\
a_3 \\
\dots \\
a_{N+1}  \end{array} \right)
=
\left( \begin{array}{cccc}
-r_2\\
-r_3 \\
\dots \\
-r_{N+1}  \end{array} \right)$$