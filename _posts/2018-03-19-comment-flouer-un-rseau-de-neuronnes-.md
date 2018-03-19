---
layout: post
title: "Comment flouer un réseau de neuronnes ?"
description: ""
category: 
tags: []
---
<div class="message">
  Allo les chefs. Dans cette recette, nous allons voir comment flouer un réseau de neuronnes de la plus simple des manières. Nous allons programmer un réseau de neuronnes 
  dont l'objectif est de classifier les chiffres manuscrit de la base de données <a href="http://yann.lecun.com/exdb/mnist/" > MNIST</a>. 
  Ensuite nous allons <span> fabriquer </span> pour chacun des chiffres de 0 à 9, la <span> meilleure représentation </span>; c'est à dire l'image 
  parfaite que le réseau attendrait pour s'activer à 99% pour chacune des classes. Ce qui sera suprenant dans cette petite expérimentation est que, pour une image qui n'a aucun mais vraiment AUCUNNN sens à nos yeux, et bien le réseau lui est capable de dire si c'est un 0,1... ou 9.
  Ce qui est assez critique comme comportement!  
</div>

Les réseaux de neuronnes (RN) sont une technique d'apprentissage machine de type connexioniste inspirée du fonctionnement du cerveau humain. Le principe de fonctionnement d'une telle architecture est assez simple. De facon brève, voici l'idée générale :

> Nous avons un ensemble de données **X** (communément appelé les entrées du réeau) dont nous voulons prédire les sorties associées **Y** (**Y** peut être égale à **X** dans certaines architectures de RN comme les auto-encoders), le but est de minimiser l'erreur de prédiction qui est fonction de Y et de 
**Y'** (sortie prédit pas le réseau). L'erreur **E** peut prendre plusieurs forme mais en général, elle représente la différence entre le réel **Y** et le prédit **Y'**. On peut donc conclure que plus **E** est petit, le mieux c'est. L'aapprentissage vise à trouver la configuration du réseau (les poids **W**<sub><b>i</b></sub> **i** &#8712; &#925; ) qui minimise cette erreur.
L'algorithme le plus utilisé pour cet apprentissage est la déscente du gradient. 
. 

Vous trouverez sur mon [dépôt git](https://github.com/angetato/Optimizers-for-Tensorflow), quelques algorithmes d'optimisation dans les RN utilisant la descente du gradient. 

![img](http://latex.codecogs.com/svg.latex?%5Cfrac%7B%5Csigma%7D%7B%5Cmu%7D) 

## Inline HTML elements

HTML defines a long list of available inline tags, a complete list of which can be found on the [Mozilla Developer Network](https://developer.mozilla.org/en-US/docs/Web/HTML/Element). (E & Huang 2001)[^Huang2001]

- **To bold text**, use `<strong>`.
- *To italicize text*, use `<em>`.
- Abbreviations, like <abbr title="HyperText Markup Langage">HTML</abbr> should use `<abbr>`, with an optional `title` attribute for the full phrase.
- Citations, like <cite>&mdash; Mark otto</cite>, should use `<cite>`.
- <del>Deleted</del> text should use `<del>` and <ins>inserted</ins> text should use `<ins>`.
- Superscript <sup>text</sup> uses `<sup>` and subscript <sub>text</sub> uses `<sub>`.

Most of these elements are styled by browsers with few modifications on our part.

## Heading

Vivamus sagittis lacus vel augue rutrum faucibus dolor auctor. Duis mollis, est non commodo luctus, nisi erat porttitor ligula, eget lacinia odio sem nec elit. Morbi leo risus, porta ac consectetur ac, vestibulum at eros.

### Code

Cum sociis natoque penatibus et magnis dis `code element` montes, nascetur ridiculus mus.

{% highlight js %}
// Example can be run directly in your JavaScript console

// Create a function that takes two arguments and returns the sum of those arguments
var adder = new Function("a", "b", "return a + b");

// Call the function
adder(2, 6);
// > 8
{% endhighlight %}

Aenean lacinia bibendum nulla sed consectetur. Etiam porta sem malesuada magna mollis euismod. Fusce dapibus, tellus ac cursus commodo, tortor mauris condimentum nibh, ut fermentum massa.

### Gists via GitHub Pages

Vestibulum id ligula porta felis euismod semper. Nullam quis risus eget urna mollis ornare vel eu leo. Donec sed odio dui.

{% gist 5555251 gist.md %}

Aenean eu leo quam. Pellentesque ornare sem lacinia quam venenatis vestibulum. Nullam quis risus eget urna mollis ornare vel eu leo. Cum sociis natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Donec sed odio dui. Vestibulum id ligula porta felis euismod semper.

### Lists

Cum sociis natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Aenean lacinia bibendum nulla sed consectetur. Etiam porta sem malesuada magna mollis euismod. Fusce dapibus, tellus ac cursus commodo, tortor mauris condimentum nibh, ut fermentum massa justo sit amet risus.

* Praesent commodo cursus magna, vel scelerisque nisl consectetur et.
* Donec id elit non mi porta gravida at eget metus.
* Nulla vitae elit libero, a pharetra augue.

Donec ullamcorper nulla non metus auctor fringilla. Nulla vitae elit libero, a pharetra augue.

1. Vestibulum id ligula porta felis euismod semper.
2. Cum sociis natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus.
3. Maecenas sed diam eget risus varius blandit sit amet non magna.

Cras mattis consectetur purus sit amet fermentum. Sed posuere consectetur est at lobortis.

<dl>
  <dt>HyperText Markup Language (HTML)</dt>
  <dd>The language used to describe and define the content of a Web page</dd>

  <dt>Cascading Style Sheets (CSS)</dt>
  <dd>Used to describe the appearance of Web content</dd>

  <dt>JavaScript (JS)</dt>
  <dd>The programming language used to build advanced Web sites and applications</dd>
</dl>

Integer posuere erat a ante venenatis dapibus posuere velit aliquet. Morbi leo risus, porta ac consectetur ac, vestibulum at eros. Nullam quis risus eget urna mollis ornare vel eu leo.

### Images

Quisque consequat sapien eget quam rhoncus, sit amet laoreet diam tempus. Aliquam aliquam metus erat, a pulvinar turpis suscipit at.

![placeholder](http://placehold.it/800x400 "Large example image")
![placeholder](http://placehold.it/400x200 "Medium example image")
![placeholder](http://placehold.it/200x200 "Small example image")

### Tables

Aenean lacinia bibendum nulla sed consectetur. Lorem ipsum dolor sit amet, consectetur adipiscing elit.

<table>
  <thead>
    <tr>
      <th>Name</th>
      <th>Upvotes</th>
      <th>Downvotes</th>
    </tr>
  </thead>
  <tfoot>
    <tr>
      <td>Totals</td>
      <td>21</td>
      <td>23</td>
    </tr>
  </tfoot>
  <tbody>
    <tr>
      <td>Alice</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <td>Bob</td>
      <td>4</td>
      <td>3</td>
    </tr>
    <tr>
      <td>Charlie</td>
      <td>7</td>
      <td>9</td>
    </tr>
  </tbody>
</table>

Nullam id dolor id nibh ultricies vehicula ut id elit. Sed posuere consectetur est at lobortis. Nullam quis risus eget urna mollis ornare vel eu leo.

-----

Want to see something else added? <a href="https://github.com/poole/poole/issues/new">Open an issue.</a>



References
--------------------------
[^Huang2001]: E, W. & Huang, Z., 2001. Matching Conditions in Atomistic-Continuum Modeling of Materials. _arXiv.org_, (13), p.135501. Available at: [http://arxiv.org/abs/cond-mat/0106615v1](http://arxiv.org/abs/cond-mat/0106615v1).