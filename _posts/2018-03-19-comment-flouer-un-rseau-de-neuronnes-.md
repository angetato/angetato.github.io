---
layout: post
title: "Comment flouer un réseau de neuronnes ?"
description: ""
category: 
tags: []
---
<div class="message">
  <p align="justify">
  Allo les chefs. Dans cette recette, nous allons voir comment flouer un réseau de neuronnes de la plus simple des manières. Nous allons programmer un réseau de neuronnes 
  dont l'objectif est de classifier les chiffres manuscrit de la base de données <a href="http://yann.lecun.com/exdb/mnist/" > MNIST</a>. 
  Ensuite nous allons <span> fabriquer </span> pour chacun des chiffres de 0 à 9, la <span> meilleure représentation </span>; c'est à dire l'image 
  parfaite que le réseau attendrait pour s'activer à 99% pour chacune des classes. Ce qui sera suprenant dans cette petite expérimentation est que, pour une image qui n'a aucun mais vraiment AUCUNNN sens à nos yeux, et bien le réseau lui est capable de dire si c'est un 0,1... ou 9.
  Ce qui est assez critique comme comportement!  
  </p>
</div>

 Les réseaux de neuronnes (RN) sont une technique d'apprentissage machine de type connexioniste inspirée du fonctionnement du cerveau humain. Le principe de fonctionnement d'une telle architecture est assez simple. De facon brève, voici l'idée générale :

> Nous avons un ensemble de données **X** (communément appelé les entrées du réeau) dont nous voulons prédire les sorties associées **Y** (**Y** peut être égale à **X** dans certaines architectures de RN comme les auto-encoders), le but est de minimiser l'erreur de prédiction qui est fonction de Y et de 
**Y'** (sortie prédit pas le réseau). L'erreur **E** peut prendre plusieurs forme mais en général, elle représente la différence entre le réel **Y** et le prédit **Y'**. On peut donc conclure que plus **E** est petit, le mieux c'est. L'aapprentissage vise à trouver la configuration du réseau (les poids **W**<sub><b>i</b></sub> **i** &#8712; &#925;) qui minimise cette erreur.
L'algorithme le plus utilisé pour cet apprentissage est la déscente du gradient.
. 

Vous trouverez sur mon [dépôt git](https://github.com/angetato/Optimizers-for-Tensorflow), quelques algorithmes d'optimisation dans les RN utilisant la descente du gradient. 

![img](http://latex.codecogs.com/svg.latex?%5Cfrac%7B%5Csigma%7D%7B%5Cmu%7D) 

# <a name="part1"></a>Partie 1: Architecture du réseau à flouer

Le code complet de cette recette se trouve [ici](#). J'utilise **Python** et **Tensorflow**, vous devez donc les avoir sur votre machine pour pouvoir exécuter le code.
Considérons le réseau de neurones suivant à 2 couches (Entrées + Sorties)
![Source](https://ml4a.github.io/images/figures/mnist_1layer.png)
[Source](https://ml4a.github.io/images/figures/mnist_1layer.png)
Il y'a 784 neurones en entrée puisque, chaque image de la BD mnist est encodée en une matrice de 28x28 pixels = 784 pixels (si on aplati l'image). En sortie, nous avons 10 neurones, chacun représentant les chiffres de 0 à 9. Par exemple, si le chiffre en entrée est 0 alors, le neurone chargé de reconnaître ce chiffre devra s'activer plus que les autres.  
**Remarque :** Pour ceux qui maîtrise le codage d'une architecture neuronale avec tensorflow, vous pouvez vous rendre directement à la partie [2](#part2).
Voici le code python pour charger la BD : 

```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
```
Une fois la BD chargée dans la variable *mnist*, nous allons définir les paramètres d'entraînement de notre modèle. **Remarque:** Je n'ai pas choisis les valeurs optimales, mon but ici n'est pas d'obtenir les meilleurs résultat de classification ...
```python
# Training Parameters
learning_rate = 0.001 # le pas d'apprentissage : une petite valeur implique des pas plus petis vers la solution et vice versa. Je ferais une recette bientôt sur l'importance de cette valeur et comment la paramétrer.
num_steps = 30000 
batch_size = 1000 # taille des données de training que le réseau devra voir avant de mettre à jour les poids.
display_step = 10 # nous allons afficher le résultat courant après chaque 10 step.

# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
num_out = 10 # 10 classes (0...9)
```
Les variables du réseau :
```python
X = tf.placeholder("float", [None, num_input]) #None sera remplacé par la taille du batchsize.
Y = tf.placeholder(tf.float32, shape=[None, 10]) # Le vecteur de sortie.
weights =  tf.Variable(tf.random_normal([num_input, num_out])) # les poids
```
La sortie est sous la forme d'un vecteur à 10 entrées (10 neurones en sorties). À chaque image correspond un vecteur de sortie. Par exemple pour une image correspondant à **1**, on aura **[0,1,0,0,0,0,0,0,0,0]** alors que pour une image correspondant à **9**, on aura **[0,0,0,0,0,0,0,0,0,1]**. 
Le calcul de prédiction, de l'erreur et de l'accuracy de notre réseau se fait comme sut :
```python
# Prediction
out =  tf.matmul(X, weights) # notre Y'

# Define loss and optimizer, minimize the squared error
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=out))
train_step = tf.train.AdamOptimizer(0.002).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```
Une fois tout ceci fait, nous pouvons entraîner et tester notre modèle. 
```python
    # Training
    for i in range(1, num_steps+1):
        # Prepare Data
        # Get the next batch of MNIST data (images and labels)
        batch = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop), cost op (to get loss value) and accuracy opp
        _, l, acc = sess.run([train_step,cross_entropy, accuracy], feed_dict={X: batch[0], Y: batch[1]})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print(' Step %i: Minibatch Loss: %f accuracy : %f ' % (i, l, acc))
```
Nous n'allons pas tester notre modèle sur le _test set_ de la BD. Après 30000 étapes, nous avons obtenus une accuracy d'environ 94% sur le _training set_ . Le modèle ainsi concu, est capable à partir d'une image d'un chiffre, de prédire avec 94% le chiffre écrit. Que se passe t-il si aucun chiffre n'est écrit sur l'image ? 

# <a name="part2"></a>Partie 2: Trompons notre réseau 

Le modèle que nous avons développé dans la partie [1](#part1) est capable de nous dire avec 94% d'accuracy le chiffre qui est contenu dans une image en entrée. Si on lui donne une image ne contenant pas de chiffre, il devrait être capable de nous dire qu'il n'y a pas de chiffre écrit dans l'image. Cependant, la réalité est tout autre ! Même si il n'y a rien dans une image que vous donnez en entré à un réseau, il vous donnera toujours un résultat (0...9). La bonne nouvelle est que, ce résultat est sous forme de tableau de probabilité. 
Regardons cette image pris d'[ici](https://ml4a.github.io/images/figures/mnist-mistakes.png):
![](https://ml4a.github.io/images/figures/mnist-mistakes.png)
On voit que, le réseau attribut des probabilités d'appartenance à chacun des 10 chiffres pour chaque image. On s'attends donc à ce que, pour une image ne contenant aucun chiffre, le réseau attribut des probabilités très faibles pour chacun des chiffres sachant qu'il n'a pas appris à dire non ("qu' aucun chiffre n'est présent"). On pourrait penser à rajouter un neurone en sortie pour les cas ou l'image ne contient aucun chiffre (elle peut contenir un chat, une maison, etc.). Néanmoins, ce dernier cas de figure soulèverait 3 problèmes :
* Trouver des exemples étiquetés pour pouvoir entraîner le réseau sur des images ne contenant pas de chiffres.
* Quelles images devront être prises en considération ? (les images contenant des chats ? des chiens ? des maisons ? ... impossible de déterminer les images qui aideront le réseau à distinguer celles qui contiennent des chiffres de celles qui n'en contiennent pas, il est impossible de couvrir tous les cas possibles d'images).
* Quelle quantité d'images faudra t-il au réseau pour qu'il puisse bien généraliser (_ca c'est un problème général de l'apprentissage machine_ ).

Je propose dans la conclusion, des pistes de réflexion pour une possible solution. 

Revenons à nos moutons 🐑 ... Nous allons donner à notre réseau ce qu'il veut !
## Qu' apprend le réseau ?

Nous allons visualiser ce que chaque neurones de la couche cachée (notre couche de sortie) apprends. Soit la [figure](https://ml4a.github.io/ml4a/looking_inside_neural_nets/) suivante :
![](https://ml4a.github.io/images/figures/weights_analogy_1.png)
Chaque neurone de la couche cachée

## Est il vraiment intelligent notre RNs?

All your files are listed in the file explorer. You can switch from one to another by clicking a file in the list.

# Conclusion
HTML defines a long list of available inline tags, a complete list of which can be found on the [Mozilla Developer Network](https://developer.mozilla.org/en-US/docs/Web/HTML/Element). (E & Huang 2001)[^Huang2001]

References
--------------------------
[^Huang2001]: E, W. & Huang, Z., 2001. Matching Conditions in Atomistic-Continuum Modeling of Materials. _arXiv.org_, (13), p.135501. Available at: [http://arxiv.org/abs/cond-mat/0106615v1](http://arxiv.org/abs/cond-mat/0106615v1).
