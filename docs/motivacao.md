# Descrição e motivação dos algoritmos

Tema: análise de treino de academia. Dois problemas independentes, cada um com um
modelo clássico de Machine Learning e um de Deep Learning, conforme o enunciado.

| Problema | Tipo de dado | Modelo | Papel | Acurácia |
|----------|--------------|--------|-------|----------|
| Nível de experiência do aluno | Tabular (973×15) | Random Forest | ML clássico | 90,8% |
| Nível de experiência do aluno | Tabular (973×15) | MLP (rede densa) | DL clássico | 83,6% |
| Reconhecimento de exercício | Imagem (22 classes) | CNN treinada do zero | DL treinado | 66,7% |
| Reconhecimento de exercício | Imagem (22 classes) | MobileNetV2 (transfer) | DL pré-treinado | 98,3% |

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

### CNN treinada do zero (DL treinado)

Redes convolucionais exploram a estrutura local da imagem por compartilhamento de pesos e
invariância a translação, padrão consolidado desde LeCun et al. (1998) e popularizado em
classificação de imagens por Krizhevsky et al. (2012). Nossa CNN (três blocos
convolução–ReLU–pooling seguidos de cabeça densa) serve como o modelo "treinado do zero"
exigido, aprendendo todos os filtros apenas com as 22 classes de exercícios. Chegou a
66,7% partindo de pesos aleatórios, bem acima do acaso (~4,5% para 22 classes), mas
limitada pela quantidade de imagens por classe e pelo custo de treino em CPU.

### MobileNetV2 pré-treinado (DL pré-treinado)

Usamos transfer learning: MobileNetV2 (Sandler et al., 2018) com pesos do ImageNet (Deng
et al., 2009), backbone congelado e apenas a camada final retreinada para as 22 classes.
A motivação é bem fundamentada:

- **Transferência de features.** Camadas iniciais de redes treinadas em ImageNet aprendem
  bordas e texturas reutilizáveis em outros domínios visuais (Yosinski et al., 2014; Pan &
  Yang, 2010), então não precisamos reaprender isso do zero.
- **Eficiência.** MobileNetV2 foi projetada para baixo custo (blocos residuais invertidos
  e bottlenecks lineares), viável em CPU. Com o backbone congelado, só a cabeça treina.
- **Resultado.** 98,3% em 5 épocas contra 66,7% da CNN do zero em 10 — exatamente o ganho
  que a teoria de transfer learning prevê para datasets de tamanho moderado.

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
- Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. (2018). MobileNetV2: Inverted
  Residuals and Linear Bottlenecks. *CVPR*.
- Deng, J., Dong, W., Socher, R., Li, L., Li, K., & Fei-Fei, L. (2009). ImageNet: A
  Large-Scale Hierarchical Image Database. *CVPR*.
- Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). How transferable are features
  in deep neural networks? *NeurIPS*.
- Pan, S. J., & Yang, Q. (2010). A Survey on Transfer Learning. *IEEE Transactions on
  Knowledge and Data Engineering*, 22(10), 1345–1359.
