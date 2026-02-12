# DL04 - CNN2D/Conv2D (Conteúdo Conceitual)

Este arquivo reúne o conteúdo conceitual relacionado ao notebook `notebooks/DL/DL04-cnn2d-conv2d.ipynb`.

## Sumário

- [Redes Neurais Convolucionais](#redes-neurais-convolucionais)
- [Introdução](#introducao)
    - [Operação de convolução](#operacao-de-convolucao)
  - [Camada de convolução (Conv2D)](#camada-de-convolucao-conv2d)
  - [Convolução 2D em numpy](#convolucao-2d-em-numpy)
  - [Convolução 2D em PyTorch](#convolucao-2d-em-pytorch)
  - [Caso geral](#caso-geral)
  - [Filtros de imagens](#filtros-de-imagens)
  - [Camada de Pooling](#camada-de-pooling)


# Redes Neurais Convolucionais

---

# Introdução

Uma Rede Neural Convolucional (CNN, do inglês *Convolutional Neural Network*) é um tipo de rede neural artificial projetada para processar e analisar dados que têm uma estrutura de grade regular, como imagens. As CNNs são especialmente eficazes em tarefas de visão computacional, como classificação de imagens, detecção de objetos e segmentação de imagens, devido à sua capacidade de capturar padrões espaciais e hierárquicos nas imagens.

A imagem a seguir (retirada do artigo [Computer science: The learning machines](https://www.nature.com/articles/505146a)) ilustra a capacidade de uma CNN para capturar padrões visuais hierárquicos em uma imagem.

<center><img src='https://media.springernature.com/w300/springer-static/image/art%3A10.1038%2F505146a/MediaObjects/41586_2014_Article_BF505146a_Figc_HTML.jpg'></center>

A figura a seguir apresenta uma arquitetura típica de uma rede neural convolucional.

<center><img src='https://editor.analyticsvidhya.com/uploads/36181719641_uAeANQIOQPqWZnnuH-VEyw.jpeg'></center>

---

Uma rede neural convolucional é usualmente composta de dois estágios sucessivos. Cada um desses estágios é composto de uma sequência de camadas. 

- O primeiro estágio é responsável por extrair as características (*features*) relevantes do objeto de entrada (e.g., imagem). Esse estágio é normalmente composto por camadas de convolução e de subamostragem (*pooling*). 

- Já o segundo estágio é responsável por realizar a tarefa a que a rede se pretende. Esse segundo estágio é composto por camadas completamente conectadas (as mesmas encontradas em redes [MLP](https://en.wikipedia.org/wiki/Multilayer_perceptron)).

A organização descrita acima é ilustrada na figura abaixo ([fonte](https://en.wikipedia.org/wiki/Convolutional_neural_network)). Essa figura apresenta dois modelos populares de rede convolução, [LeNet](https://en.wikipedia.org/wiki/LeNet) e [AlexNet](https://en.wikipedia.org/wiki/AlexNet). Essas redes foram originalmente propostas para treinamento sobre os conjuntos [MNIST](http://yann.lecun.com/exdb/mnist/) e [Imagenet](https://www.image-net.org), respectivamente.

<center><img src='https://upload.wikimedia.org/wikipedia/commons/thumb/c/cc/Comparison_image_neural_networks.svg/1280px-Comparison_image_neural_networks.svg.png'  width="600"></center>

---

### Operação de convolução

A figura abaixo ([fonte](https://www.mathworks.com/help/signal/ref/xcorr2.html)) apresenta um exemplo da operação de convolução 2D. Nessa figura, a matriz no canto direito superior é o filtro. No canto esquerdo inferior é apresentada a  matriz sobre a qual a convolução deve ser aplicada (i.e., o plano de entrada). 

<center><img src='https://i.imgur.com/iDfpijK.png'></center>

Em um dado momento da aplicação da operação, o filtro vai estar alinhado com a parte superior direita da matriz de entrada, conforme mostra a figura. Nesse alinhamento, o valor calculado é $585$, resultante da seguinte expressão:

$$
1 \times 8 + 7 \times 3+13 \times 4+8 \times 1+14 \times 5+20 \times 9+15 \times 6+16 \times 7+22 \times 2=585.
$$

A operação que produz o valor $585$ é denominada **correlação cruzada** (_cross-correlation_). De forma simplificada, a correlação cruzada sobre duas matrizes $A$ e $B$ corresponde a multiplicar os elementos correspondentes de $A$ e $B$ e em seguida realizar a soma desses produtos. Vamos denotar essa operação usando o símbolo $\star$. Repare que o resultado da operação de correlação cruzada é um único valor $v \in \Re$.

$$
v = A \star B
$$

O valor computado no alinhamento acima corresponde a uma das entradas no plano de saída. O plano de saída completo é obtido considerando todos os possíveis alinhamentos do filtro com a matriz de entrada. O plano de saída para o exemplo acima é apresentada abaixo, na qual o valor $585$, resultante do alinhamento apresentado na figura acima, está destacado.

\begin{pmatrix}
 405 & 570 & \mathbf{585}\\
 550 & 615 & 730\\
 595 & 760 & 575
\end{pmatrix}

A figura acima ilustra a aplicação da correlação cruzada em um alinhamento específico do filtro sobre o plano de entrada. Contudo, note que a operação descrita acima é aplicada aos diversos alinhamentos possíveis entre o plano de entrada e o filtro. Veja a imagem abaixo.

<center><img src='https://github.com/vdumoulin/conv_arithmetic/blob/master/gif/no_padding_no_strides.gif?raw=true'></center>

Por vezes, é desejável manter as dimensões dos planos de entrada e de saída. Nesse caso, as dimensões do plano de entrada são primeiramente aumentadas para em seguida realizar a aplicação da convolução cruzada. Essa extensão do plano de entrada é denominada **padding**. Como exemplo, a animação abaixo ilustra esquematicamente a correlação cruzada aplicada a uma plano de entrada com padding igual a 2. 
<center><img src='https://github.com/vdumoulin/conv_arithmetic/blob/master/gif/arbitrary_padding_no_strides.gif?raw=true'></center>

Outra variação da correlação cruzada diz respeito à distância entre diferentes alinhamentos do filtro sobre o plano de entrada. Essa distância é chamada de **stride**. Nas imagens anteriores, foi usado stride igual a 1, i.e., o filtro é deslocado uma posição para produzir o próximo alinhamento. Já na imagem a seguir, o stride foi definido como 2.

<center><img src='https://github.com/vdumoulin/conv_arithmetic/blob/master/gif/no_padding_strides.gif?raw=true'></center>

Há uma expressão que permite computar a dimensão $W_{\text{out}}$ do plano de saída, uma vez definidos as dimensões do filtro $K$, as dimensões do plano de entrada $W_{\text{in}}$, o padding $P$ e o stride $S$. Essa expressão é apresentada a seguir.

$$
W_{\text{out}} = \frac{(W_{\text{in}} - K + 2\times P)}{S}+1
$$

Na expressão acima:

- $W_{\text{out}}$ é a dimensão do plano de saída;
- $W_{\text{in}}$ é a dimensão do plano de entrada;
- $K$ é a dimensão do filtro (_kernel_);
- $P$ é o valor do preenchimento (_padding_);
- $S$ é o valor do salto (_stride_).

Se voltarmos ao exemplo da figura abaixo, $W_{in}=5$, $K=3$. Se definirmos $P=0$ e $S=1$, isso irá resultar em $W_{out} = 3$.

<center><img src='https://i.imgur.com/iDfpijK.png'></center>

---

## Camada de convolução (Conv2D)

Camadas de convolução são usadas para extrair as características visuais relevantes dos objetos de entrada. Em uma rede neural de convolução há normalmente várias camadas de convolução sucessivas. Camdas de convolução iniciais são responsáveis por extrair características visuais simples (arestas verticais, horizontais, inclinadas, etc.). Essas características visuais mais simples são então usadas pelas camadas de convolução seguintes para detectar padrões visuais cada vez mais complexos. Isso é ilustrado na imagem a seguir ([fonte](https://www.ais.uni-bonn.de/papers/ki2012_schulz_deeplearning.pdf)).

<center><img src='https://i.imgur.com/FLPS1Qb.png'></center>

Os neurônios em uma Conv2D são organizados em uma estrutura tridimensional (volume). Posto que uma Conv2D é uma estrutura tridimensional, esse tipo de camada possui altura, largura e profundidade. Pelo mesmo motivo, tanto a entrada quanto a saída de uma Conv2D são volumes. Veja a imagem a seguir ([fonte](https://cs231n.github.io)).

<center><img src='https://i.imgur.com/ExLnMKF.jpg'></center>

O conjunto de neurônios localizados na mesma coluna de profundidade de uma CONConv2DV são conectados à mesma região do volume de entrada. Essa região é conhecida como o _campo receptivo local_ desses neurônios. Veja a imagem abaixo ([fonte](https://cs231n.github.io)).

<center><img src='https://cs231n.github.io/assets/cnn/depthcol.jpeg'></center>

---


Em uma rede MLP, cada camada de neurônios está organizada conforme uma estrutura unidimensional.
<center><img src='https://cs231n.github.io/assets/nn1/neural_net2.jpeg'></center>

Já em uma camada de convolução, os neurônios são organizados em uma estrutura tridimensional. Posto que uma CONV é uma estrutura tridimensional, esse tipo de camada possui altura, largura e profundidade. Veja a imagem a seguir ([fonte](https://cs231n.github.io)). Além disso, tanto a entrada quanto a saída de uma CONV são também tensores tri-dimensionais denominados *volumes*.

<center><img src='https://cs231n.github.io/assets/cnn/cnn.jpeg'></center>

O conjunto de neurônios localizados na mesma coluna de profundidade de uma CONV são conectados à mesma região do volume de entrada. Essa região é conhecida como o _campo receptivo local_ desses neurônios. Veja a imagem abaixo ([fonte](https://cs231n.github.io)).

<center><img src='https://cs231n.github.io/assets/cnn/depthcol.jpeg' ></center>

---

## Caso geral

Quando consideramos uma camada de convolução 2D em um rede neural convolucional, 
- a entrada pode ser um lote composto de vários tensores, e cada um desses tensores é um **volume** (i.e., um tensor 3D) que pode possuir vários canais. 
- a saída é outro lote (de mesmo tamanho da entrada) de volumes, 
- os volumes de entrada e de saída podem conter uma quantidade de canais diferentes. 

Nesse caso geral, a expressão geral usada pelo [PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) para computar a convolução 2D é a seguinte:

$$
\text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) + \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i,k)
$$

Na expressão acima:
- considera-se um tensor de entrada com dimensões $(N,C_{in},H,W)$
- considera-se um tensor de saída com dimensões $(N,C_{out},H_{out},W_{out})$
- $⋆$ é o operador de correlação cruzada 2D,
- $N$ é o tamanho do lote,
- $C$ denota o número de canais,
- $H$ é a altura dos planos de entrada em pixels,
- $W$ é a largura dos planos de entrada em pixels.

Uma maneira de entender a expressão acima é analisar a animação disponível na seção _Convolution Demo_ nesta [página](https://cs231n.github.io/convolutional-networks/#conv).

---

## Filtros de imagens

Um filtro de imagem (_image filter_, _image kernel_) é uma pequena matriz usada para aplicar efeitos sobre uma imagem. Programas como Photoshop e Gimp fornecem filtros para diversas transformações, como desfoque, nitidez, contorno ou relevo. 

Filtros também são usados ​​no aprendizado de máquina para realizar _extração de recursos_ (_feature extraction_), uma técnica para determinar as partes mais importantes de uma imagem. Neste contexto, o processo é referido de forma mais geral como **convolução**.

Para exemplificar a aplicação de filtros sobre imagens, vamos considerar o [Filtro de Prewitt](https://en.wikipedia.org/wiki/Prewitt_operator), que é usado para detectar arestas em uma imagem de entrada. Há duas versões desse filtro, uma para detecção de arestas  verticais e outra para as horizontais. As matrizes desses filtros, denotadas por $\mathbf{G_{x}}$ (detector de arestas verticais) e $\mathbf{G_{y}}$ (detector de arestas horizontais), são apresentadas a seguir:

$$
\mathbf{G_{x}} =
   \begin{bmatrix}
   +1&0&-1\\
   +1&0&-1\\
   +1&0&-1
   \end{bmatrix}, \, \mathbf{G_{y}} =
   \begin{bmatrix}
   +1&+1&+1\\
   0&0&0\\
   -1&-1&-1
   \end{bmatrix}
$$

Para entender o porquê de os filtros acima serem adequados para detectar arestas em imagens, considere a imagem abaixo, que possui dimensões $390 \times 598$. 

<center><img src='https://i.imgur.com/MXkUq3k.png' width="200"></center>

Considere também o filtro $\mathbf{G_{x}}$, cujo propósito é detectar arestas verticais. A figura a seguir ilustra esquematicamente um alinhamento particular desse filtro sobre uma imagem: o filtro está localizado na região correspondente ao canto superior direito da imagem de entrada. 

<center><img src='https://i.imgur.com/PsqwVVf.png' width="600"></center>

Considere o efeito de aplicar $\mathbf{G_{x}}$ sobre essa região. Isso irá gerar o valor $b22$ na imagem de saída, que é computado conforme segue.

$$
b22 = a11 + a21  + a31 - (a13 + a23 + a33)
$$

Repare que se no alinhamento todas as intensidades de pixel forem iguais (o que equivale à inexistência de arestas nesta região da imagem de entrada), então o valor computado para $b22$ será igual a $0$. 

Por outro lado, se $a11 = a21  = a31 = 255$ e $a13 = a23  = a33 = 0$, isso indica que há uma aresta naquela região. Desta vez, filtro irá produzir o valor máximo possível. Dizemos nesse caso que o filtro detectou uma aresta vertical naquela região da imagem de entrada.

---

Para exemplificar a aplicação desses filtros de forma programática, considere novamente a imagem acima (esboço de uma face). O bloco de código a seguir usa a biblioteca [scikit-image](https://scikit-image.org) para aplicar os filtros de Prewitt sobre a imagem acima.

---

Outro exemplo, desta vez usando uma imagem real.

<center><img src='../img/cute_cat.jpeg'></center>

---

Esse [link](https://setosa.io/ev/image-kernels/) apresenta uma interface interativa que permite obter uma intuição sobre como filtros funcionam.

Esse outro [link](https://theailearner.com/tag/prewitt-operator/) fornece uma visão geral sobre outros filtros.

---

## Camada de Pooling

Uma camada de pooling (ou camada de amostragem) é uma operação que reduz a dimensionalidade dos mapas de características (feature maps) gerados pelas camadas convolucionais anteriores. 
O objetivo principal do pooling é diminuir a resolução espacial dos mapas de características, reduzindo assim a quantidade de parâmetros e cálculos na rede, além de ajudar a controlar o overfitting.

Existem diferentes tipos de operações de pooling, mas as mais comuns são:

- Max Pooling: Seleciona o valor máximo em cada janela de pooling.
- Average Pooling: Calcula a média dos valores em cada janela de pooling.
 
Benefícios da Camada de Pooling
1. Redução da Dimensionalidade: Reduz a resolução espacial dos mapas de características, o que diminui o número de parâmetros e cálculos na rede.
2. Controle de Overfitting: Ao reduzir a dimensionalidade, o pooling ajuda a evitar o overfitting, pois força a rede a aprender características mais robustas e invariantes.
3. Invariância à Translação: Pooling introduz uma forma de invariância à translação, já que pequenas mudanças na posição dos elementos dos mapas de características têm menos impacto após a operação de pooling.

Considerações
- Tamanho da Janela e Stride: O tamanho da janela de pooling e o stride são hiperparâmetros importantes que devem ser escolhidos cuidadosamente, pois influenciam diretamente a redução da dimensionalidade.
- Tipo de Pooling: A escolha entre max pooling e average pooling depende do problema específico e das características dos dados. Max pooling é mais comum, pois tende a capturar características salientes melhor do que average pooling.

Camadas de subamostragem (POOL) são usadas para realizar a operação homônima (*downsampling*) sobre o volume de entrada. O efeito dessa operação é a diminuição da quantidade de pixels no volume de saída. 

Normalmente, essa operação gera uma perda de qualidade no volume de saída quando comparado ao volume de entraga. Isso é ilustrado na imagem abaixo (fonte). Como compensação, essa operação diminui a quantidade de parâmetros do modelo e consequentemente pode servir para combater o *overfitting*.

<center><img src='https://cs231n.github.io/assets/cnn/pool.jpeg'></center>

Há suas formas de implementar a operação de subamostragem, *max pooling* e *average pooling*. Essas operações são ilustradas na imagem a seguir.
<center><img src='https://www.researchgate.net/profile/Alla-Eddine-Guissous/publication/337336341/figure/fig15/AS:855841334898691@1581059883782/Example-for-the-max-pooling-and-the-average-pooling-with-a-filter-size-of-22-and-a_W640.jpg'></center>

Uma alternativa ao uso de camadas POOL é usar camadas CONV com stride maior do que 1. Mais detalhes sobre isso podem ser obtidos no artigo [Striving for Simplicity: The All Convolutional Net](https://arxiv.org/abs/1412.6806).
