# Topic Modeling and Latent Dirichlet Allocation (LDA)
A quick non-technical intro to LDA with an example

## Intro
En este doc vamos a explorar una técnica no supervisada que se utiliza para encontrar temas “ocultos” en una colección de textos. El algoritmo se llama Latent Dirichlet Allocation (LDA) y fue propuesto por David Blei, Andrew Ng, Michael Jordan (no es el basquetbolista!!!). Si bien hay varios artículos en internet sobre el tema, compartimos acá otra manera de interpretar cómo funciona el algoritmo y una aplicación del mismo en un caso con datos de conversaciones de Customer Support del equipo de Nuvemshop. Incluimos como ejemplo un Jupyter notebook (sin la fuente de datos por motivos de confidencialidad).

## ¿Cómo funciona el algoritmo de LDA?
Supongamos que tenemos una colección de imágenes que queremos clasificar en temas. Supongamos también, que todas las imágenes son solamente sobre tres temas. 

Acá hay un ejemplo de tres de esas imágenes. 

Como primer paso (lo mismo que haríamos con textos), extraemos las palabras u objetos que tiene cada imagen (documento si habláramos de textos). 

 - Vocabulario foto 1: árbol, animal, flor, río, arbusto,pasto
 - Vocabulario foto 2: auto, persona, moto, ventana, puerta, cemento, árbol, luces
 - Vocabulario foto 3: pasto, persona, luces, arco, cartel, pelota

Y luego armamos un vocabulario total

 - Vocabulario total: árbol, animal, flor, río, arbusto, pasto,  auto, persona, moto, ventana, puerta, cemento, arco, luces, cartel, pelota

Supongamos ahora que el resto de todas las imágenes de mi colección se generan a partir del vocabulario total. Es decir, las fotos solo pueden tener esos objetos. Entre las fotos van a aparecer otras muy parecidas a estas tres, pero también van a aparecer fotos como esta:

 - Vocabulario foto 4: ventana, árbol, auto, pasto, cemento, puerta, luces, cartel


Fíjense que hasta ahora no hablamos de qué temas son, aunque ya en su cabeza organizaron los temas: naturaleza, ciudad y ¿estadios de fútbol?. Es interesante notar que los temas no necesariamente van a ser coherentes entre ellos o tener el mismo “status” (los estadios de fútbol podrían ser también de la ciudad). Más aún, a veces puede no haber temas definidos o puede haber temas que no son coherentes para nosotros (imaginen una colección de fotos sumamente heterogénea con cosas muy extrañas). 

El objetivo del algoritmo de LDA es encontrar la combinación de palabras y el peso de cada palabra que define a cada tema. De manera un poco más técnica, el algoritmo de LDA crea una función de probabilidad (F) de ocurrencia de palabras (P) para cada tema (T) que encuentra.

Entonces, cada Tema va a tener asociada su Función de probabilidad de Palabras. Esta función de probabilidad es (dicho rápido y no tan exactamente) una lista con un número que indica qué tan probable es que ocurra la palabra de nuestro vocabulario total en un tema. Por ejemplo, para nuestros tres temas podemos definir tres funciones de probabilidad. Veamos el ejemplo de la F del tema “Ciudad”

 - Ciudad árbol: 0.3  , animal: 0.1, flor: 0.3, río: 0.07, arbusto: 0.4, pasto: 0.2,  auto: 0.8, persona: 0.7, moto: 0.8, ventana: 0.9, puerta: 0.9, cemento: 0.95, arco: 0.01, luces: 0.8, cartel: 0.6, pelota: 0.1

Entonces, para cada tema tenemos una lista de nuestro vocabulario y la probabilidad de que esa palabra esté presente en un documento sobre ese tema. Las palabras con números más altos van a ser más frecuentes o probables en el tema. Podríamos pensarlas como las keywords del tema siempre y cuando no tengan también una alta probabilidad en otro de los temas. Por ejemplo, cartel también es muy probable en el tema “estadios de fútbol” mientras que auto no. Las keywords van a depender  de las similitudes y diferencias entre las funciones de probabilidad, pero eso es algo que vamos a ver más adelante. 

Entonces, el LDA tiene como supuesto que un autor sigue los siguientes pasos para generar un texto (o una foto):

 - Elige cuántas palabras (objetos) va a tener el texto (o la imagen)
Elige el tema o mezcla de temas sobre los que va a ser el texto
Para cada palabra: 
Elige el tema sobre el que será la palabra
Elige una palabra siguiendo la función de probabilidad del tema

Dicho de otro modo: 
Para la foto 4, la persona decidió que iba a tener dos temas (ciudad 75% y naturaleza 25%) y 8 objetos. Y para cada objeto, sacó uno de la bolsa de vocabulario del tema naturaleza (i.e. vocabulario foto 1 con su función de probabilidad) y del tema ciudad (vocabulario foto 2 y su F). Así “escribió” una foto que tiene las palabras {ventana, árbol, auto, pasto, cemento, puerta, luces, cartel}. 

### ¿Cómo puedo saber sobre qué tema(s) es la imagen? 
Bueno, esto se puede saber a partir de dónde se ubicar la imagen en una distribución de Dirichlet (WTF!?). 

La distribución de Dirichlet es una distribución que se genera a partir de la suma de 2 o más distribuciones. Y como ya habíamos hablado de tres distribuciones para nuestros tres temas, nuestra distribución de Dirichlet va a ser la suma de esas tres funciones de probabilidad de palabras de nuestros tres temas. Suena medio trabalenguas, pero esa distribución de Dirichlet va a ser la suma de las tres distribuciones de los temas.

Una manera visual de ver este tipo de distribuciones es con un triángulo. Cada vértice representa un tema (y cada tema es una distribución de palabras!). 

En este caso, si yo ya definí las funciones de cada topic, al recibir esta nueva imagen, puedo ubicarla en alguna posición del triángulo y medir la distancia que tiene a los otros vértices. En la imagen se puede ver el triángulo que define mi distribución y dónde ubicaría la imagen 4. 

La imagen 4 tiene un auto, puerta, ventana, cemento, cartel y una persona. Pero también tiene pasto y árbol. Entonces podemos decir que tiene un 74% ciudad, un 24% naturaleza y un 2% estadio de fútbol (al fin de cuentas, tiene pasto, carteles y personas, pero no tiene las cosas más características que son el arco y la pelota). Acá nosotros ya definimos la distribución de palabras de cada topic. En el caso real podríamos hacer esto con keywords, por ejemplo. 

Pero si queremos hacer topic modelling y queremos descubrir temas latentes u ocultos (de ahí el nombre de latent) tenemos que ir probando diferentes distribuciones de palabras y pesos (y también de cantidad de temas!). Esa va a ser la tarea del algoritmo de LDA.

## ¿Cómo hace LDA para encontrar las Funciones de probabilidad de las palabras?
Básicamente LDA va probando diferentes funciones y se fija dónde quedan ubicados los puntos que nosotros le ofrecimos como input (la lista de imágenes) e intenta buscar la mejor distribución que describe los datos.  

¿Cómo sabemos cuándo la distribución está “bien”o es “la mejor”?
Bueno, veamos qué pasa si cambiamos la distribución del triángulo de la imagen anterior.


Acá repartí en partes iguales entre los tres temas las palabras que tiene la foto y el punto quedó en el medio del triángulo. Y si vemos las palabras de cada vértice, no tienen mucho sentido. Entonces, el algoritmo lo que hace es probar con diferentes distribuciones de palabras y pesos y detenerse cuando encuentra una distribución que “no tiene todos los puntos en el medio” o que tiene la menor cantidad de puntos en el medio.

Es decir, hace un poco de ingeniería inversa, a partir de la ubicación de los puntos, se fija cuál es la que los aleja del medio y los acerca a los bordes. A partir de esto, lo que vamos a tener como resultado es tres funciones de probabilidad para las palabras (las keywords de nuestros topics) y una función de Dirichlet que nos va a permitir decir (o alocar y de ahí la palabra allocate) para cada texto qué tanto tiene de cada uno de los temas (o qué tan cerca está de cada uno de los vértices). 

## Ejemplo
Para ver el uso del algoritmo, chequear el notebook en este repo.




