## Descenso de gradiente con momentum

En esta parte implementarás el descenso de gradiente con momentum lo que puede mejorar 
considerablemente el tiempo de entrenamiento de tu red. 

En clases/video vimos dos formulaciones, una considerando el promedio exponencial móvil de los 
gradientes pasados, y otra como una interpretación física. 
En esta parte implementarás la segunda. 
Recuerda que en este caso la idea es incrementar una variable de *velocidad* en la dirección 
contraria del gradiente (que haría las veces de *aceleración*) y usarla para actualizar los 
parámetros en cada paso del descenso estocástico de gradiente.
Recuerda que adicionalmente un (hyper)parámetro $\mu$ de *fricción* se utiliza para evitar la 
oscilación en direcciones de gradiente muy pronunciadas.
En particular, para cada conjunto de parámetros $\theta$, y para cada paso del descenso de 
gradiente se realiza el siguiente cálculo:
$$
  \begin{aligned}
    V_{\partial \theta} & \gets & \mu V_{\partial \theta} 
      - \lambda \frac{\partial \mathcal{L}}{\partial \theta} \\
    \theta & \gets & \theta + V_{\partial \theta}
  \end{aligned}
$$
en donde $V_{\partial \theta}$ es un tensor de las mismas dimensiones que los parámetros $\theta$ y 
que se inicializa como $0$ antes de empezar el entrenamiento.

Modifica la implementación de `SGD` que ya tenías anteriormente para considerar la fórmula de 
momentum descrita.
Para esto agrega un nuevo argumento opcional `momentum` correspondiente al valor $\mu$ de la 
ecuación de arriba, con un valor por defecto de 1.
