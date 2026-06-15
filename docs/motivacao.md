# Descrição e motivação dos algoritmos

Tema: análise de treino de academia. Dois problemas independentes, cada um com um
modelo clássico de Machine Learning e um de Deep Learning, conforme o enunciado.

| Problema | Tipo de dado | Modelo | Papel | Acurácia |
|----------|--------------|--------|-------|----------|
| Nível de experiência do aluno | Tabular (973×15) | Random Forest | ML clássico | 90,8% |
| Nível de experiência do aluno | Tabular (973×15) | MLP (rede densa) | DL clássico | 83,6% |
| Reconhecimento de exercício | Imagem (10 classes) | CNN treinada do zero | DL treinado | [acurácia] |
| Reconhecimento de exercício | Imagem (10 classes) | ResNet18 (transfer) | DL pré-treinado | [acurácia] |
| Reconhecimento de exercício | Pose (MediaPipe) | MLP sobre pose | abordagem complementar | [acurácia] |

---

## Problema clássico — dados tabulares

### Random Forest (ML)

A Random Forest é um comitê de árvores de decisão treinadas em amostras bootstrap com
seleção aleatória de atributos em cada divisão, o que reduz a variância sem aumentar o
viés de forma relevante (Breiman, 2001). A escolha se justifica por três motivos
diretamente ligados ao nosso dado:

- **Robustez em dados tabulares pequenos e heterogêneos.** O conjunto tem só 973 linhas
  misturando variáveis contínuas (BPM, calorias, IMC) e categóricas (gênero, tipo de
  treino). Árvores lidam com escalas distintas e fronteiras não lineares sem exigir muito
  pré-processamento.
- **Evidência empírica de superioridade em tabular.** Estudos recentes mostram que
  ensembles de árvore ainda batem redes neurais nesse regime (Grinsztajn et al., 2022;
  Shwartz-Ziv & Armon, 2022), justamente o que observamos: 90,8% contra 83,6% da MLP.
- **Interpretabilidade.** A importância de atributos permite defender o modelo para a
  banca, mostrando quais variáveis pesam na classificação do nível.

### MLP / rede densa (DL)

A rede densa multicamada é o representante clássico de Deep Learning para vetores de
atributos: camadas totalmente conectadas com ativação não linear que aprendem combinações
dos atributos por gradiente. Foi escolhida para o contraste exigido pelo enunciado (ML vs
DL sobre a mesma preparação). A literatura indica que arquiteturas densas só alcançam ou
superam árvores em tabular com ajuste cuidadoso e dados maiores (Gorishniy et al., 2021),
o que explica o resultado um pouco inferior aqui — um ponto de defesa, não uma falha: em
dados tabulares pequenos o DL não é a melhor ferramenta.

---

## Visão computacional — imagens

O enunciado pede, para visão, um modelo de DL treinado do zero e um pré-treinado.
Implementamos os dois sobre o dataset de exercícios do Kaggle e, a partir do resultado
deles, desenvolvemos uma abordagem complementar.

### CNN treinada do zero (DL treinado)

Redes convolucionais exploram a estrutura local da imagem por compartilhamento de pesos e
invariância a translação, padrão consolidado desde LeCun et al. (1998) e popularizado em
classificação de imagens por Krizhevsky et al. (2012). Nossa CNN (quatro blocos
convolução–BatchNorm–ReLU–pooling seguidos de cabeça densa) aprende todos os filtros
apenas com as classes de exercícios, partindo de pesos aleatórios. É o modelo "treinado do
zero" exigido.

### ResNet18 pré-treinada (DL pré-treinado)

Usamos transfer learning: ResNet18 (He et al., 2016) com pesos do ImageNet (Deng et al.,
2009), substituindo a camada final pelas classes do nosso problema. A motivação é a mesma
que sustenta o transfer learning na literatura: camadas iniciais de redes treinadas em
ImageNet aprendem bordas e texturas reutilizáveis em outros domínios visuais (Yosinski et
al., 2014; Pan & Yang, 2010), então não precisamos reaprender isso do zero. As conexões
residuais da ResNet permitem treinar redes profundas sem degradação do gradiente.

### Limitação observada e abordagem complementar (pose)

Os dois modelos acima classificam diretamente o pixel da imagem. Observamos que, embora
aprendam o conjunto de treino, eles não generalizam bem para imagens e vídeos reais de
academia: exercícios visualmente parecidos (e fundos, ângulos e iluminação variados)
levam a confusões frequentes. Isso é coerente com o tamanho e a heterogeneidade do dataset
e com a dificuldade de classificação direta de ação a partir de uma única imagem.

Diante disso, desenvolvemos uma abordagem complementar que separa a percepção da decisão:

- **Extração de pose com modelo pré-treinado.** Usamos o MediaPipe Pose
  (BlazePose — Bazarevsky et al., 2020), uma rede pré-treinada que detecta 33 pontos do
  corpo. Em vez do pixel cru, passamos a representar cada imagem pelo esqueleto da pessoa,
  normalizado pela posição do quadril e pela escala dos ombros.
- **Classificação com rede própria treinada do zero.** Uma MLP aprende a classificar o
  exercício a partir desses pontos de pose.

Essa separação remove a dependência de fundo, roupa e iluminação e generaliza muito melhor
para vídeos reais, motivo pelo qual a adotamos como solução principal de demonstração,
mantendo os dois modelos do requisito como comparação.

---

## Referências

- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
- Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still
  outperform deep learning on tabular data? *NeurIPS Datasets and Benchmarks*.
- Shwartz-Ziv, R., & Armon, A. (2022). Tabular data: Deep learning is not all you need.
  *Information Fusion*, 81, 84–90.
- Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). Revisiting Deep Learning
  Models for Tabular Data. *NeurIPS*.
- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied
  to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324.
- Krizhevsky, A., Sutskever, I., & Hinton, G. (2012). ImageNet Classification with Deep
  Convolutional Neural Networks. *NeurIPS*.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image
  Recognition. *CVPR*.
- Deng, J., Dong, W., Socher, R., Li, L., Li, K., & Fei-Fei, L. (2009). ImageNet: A
  Large-Scale Hierarchical Image Database. *CVPR*.
- Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). How transferable are features
  in deep neural networks? *NeurIPS*.
- Pan, S. J., & Yang, Q. (2010). A Survey on Transfer Learning. *IEEE Transactions on
  Knowledge and Data Engineering*, 22(10), 1345–1359.
- Bazarevsky, V., Grishchenko, I., Raveendran, K., Zhu, T., Zhang, F., & Grundmann, M.
  (2020). BlazePose: On-device Real-time Body Pose Tracking. *CVPR Workshop on Computer
  Vision for Augmented and Virtual Reality*.
