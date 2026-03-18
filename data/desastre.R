pacotes <- c("plotly",
             "tidyverse",
             "ggrepel",
             "knitr", "kableExtra",
             "reshape2",
             "PerformanceAnalytics",
             "psych",
             "Hmisc",
             "readxl",
             "cluster",
             "factoextra") 

if(sum(as.numeric(!pacotes %in% installed.packages())) != 0){
  instalador <- pacotes[!pacotes %in% installed.packages()]
  for(i in 1:length(instalador)) {
    install.packages(instalador, dependencies = T)
    break()}
  sapply(pacotes, require, character = T) 
} else {
  sapply(pacotes, require, character = T) 
}
list.files()
paises <- read.csv("dados_07042023.csv", sep = ",", dec = ".")
desastres <- paises

# Estatísticas descritivas
summary(desastres)
resumo_estatistico <- summary(desastres)
write.csv(resumo_estatistico, file = "D:/OneDrive - defesacivil.rj.gov.br/USP/n supervisionadas exercicios PCA/resumo_estatistico.csv", row.names = TRUE)

# Scatter e ajuste entre as variáveis 'afetados' e 'precipitação'
desastres %>%
  ggplot() +
  geom_point(aes(x = afetados, y = precipitacao_24h),
             color = "darkorchid",
             size = 3) +
  geom_smooth(aes(x = afetados, y = precipitacao_24h),
              color = "orange", 
              method = "loess", 
              formula = y ~ x, 
              se = FALSE,
              size = 1.3) +
  labs(x = "Afetados",
       y = "Precipitação em 24h") +
  theme_bw()

# Coeficientes de correlação de Pearson para cada par de variáveis
rho <- rcorr(as.matrix(desastres[,2:12]), type="pearson")

correl <- rho$r # Matriz de correlações
sig_correl <- round(rho$P, 4) # Matriz com p-valor dos coeficientes
write.csv(sig_correl, file = "D:/OneDrive - defesacivil.rj.gov.br/USP/n supervisionadas exercicios PCA/sig_correl.csv", row.names = TRUE)


# Elaboração de um mapa de calor das correlações de Pearson entre as variáveis
ggplotly(
  desastres[,2:12] %>%
    cor() %>%
    melt() %>%
    rename(Correlação = value) %>%
    ggplot() +
    geom_tile(aes(x = Var1, y = Var2, fill = Correlação)) +
    geom_text(aes(x = Var1, y = Var2, label = format(Correlação, digits = 1)),
              size = 5) +
    scale_fill_viridis_b() +
    labs(x = NULL, y = NULL) +
    theme_bw())

# Visualização das distribuições das variáveis, scatters, valores das correlações
chart.Correlation(desastres[,2:12], histogram = TRUE, pch = "+")


### Elaboração a Análise Fatorial Por Componentes Principais ###

# Teste de esfericidade de Bartlett
cortest.bartlett(desastres[,2:12])
#rejeita-se h0 e ficamos com h1, ou seja a analise é significativa, pois p.value
# é ~ zero

# Elaboração da análise fatorial por componentes principais
fatorial <- principal(desastres[,2:12],
                      nfactors = length(desastres[,2:12]),
                      rotate = "none",
                      scores = TRUE)

# Eigenvalues (autovalores)
eigenvalues <- round(fatorial$values, 5)
eigenvalues
write.csv(eigenvalues, file = "D:/OneDrive - defesacivil.rj.gov.br/USP/n supervisionadas exercicios PCA/eigenvalues.csv", row.names = TRUE)


# Soma dos eigenvalues = 9 (quantidade de variáveis na análise)
# Também representa a quantidade máxima de possíveis fatores na análise
round(sum(eigenvalues), 2)

# Identificação da variância compartilhada em cada fator
variancia_compartilhada <- as.data.frame(fatorial$Vaccounted) %>% 
  slice(1:3)

write.csv(variancia_compartilhada, file = "D:/OneDrive - defesacivil.rj.gov.br/USP/n supervisionadas exercicios PCA/variancia_compartilhada.csv", row.names = TRUE)

rownames(variancia_compartilhada) <- c("Autovalores",
                                       "Prop. da Variância",
                                       "Prop. da Variância Acumulada")

# Variância compartilhada pelas variáveis originais para a formação de cada fator
round(variancia_compartilhada, 3) %>%
  kable() %>%
  kable_styling(bootstrap_options = "striped", 
                full_width = FALSE, 
                font_size = 20)

# Cálculo dos scores fatoriais
scores_fatoriais <- as.data.frame(fatorial$weights)
write.csv(scores_fatoriais, file = "D:/OneDrive - defesacivil.rj.gov.br/USP/n supervisionadas exercicios PCA/scores_fatoriais.csv", row.names = TRUE)


# Visualização dos scores fatoriais
round(scores_fatoriais, 3) %>%
  kable() %>%
  kable_styling(bootstrap_options = "striped", 
                full_width = FALSE, 
                font_size = 20)

# Cálculo dos fatores propriamente ditos
fatores <- as.data.frame(fatorial$scores)

View(fatores)

# Coeficientes de correlação de Pearson para cada par de fatores (ortogonais)
rho <- rcorr(as.matrix(fatores), type="pearson")
round(rho$r, 4)

# Cálculo das cargas fatoriais
cargas_fatoriais <- as.data.frame(unclass(fatorial$loadings))

# Visualização das cargas fatoriais
round(cargas_fatoriais, 3) %>%
  kable() %>%
  kable_styling(bootstrap_options = "striped", 
                full_width = FALSE, 
                font_size = 20)

# Cálculo das comunalidades
comunalidades <- as.data.frame(unclass(fatorial$communality)) %>%
  rename(comunalidades = 1)

# Visualização das comunalidades (aqui são iguais a 1 para todas as variáveis)
# Foram extraídos 10 fatores neste primeiro momento
round(comunalidades, 3) %>%
  kable() %>%
  kable_styling(bootstrap_options = "striped",
                full_width = FALSE,
                font_size = 20)


### Elaboração da Análise Fatorial por Componentes Principais ###
### Fatores extraídos a partir de autovalores maiores que 1 ###

# Definição da quantidade de fatores com eigenvalues maiores que 1
k <- sum(eigenvalues > 1)
print(k)

# Elaboração da análise fatorial por componentes principais sem rotação
# Com quantidade 'k' de fatores com eigenvalues maiores que 1
fatorial2 <- principal(desastres[,2:11],
                       nfactors = k,
                       rotate = "none",
                       scores = TRUE)
fatorial2
print(fatorial2)


#Cálculo das comunalidades com apenas os 'k' ('k' = 4) primeiros fatores
comunalidades2 <- as.data.frame(unclass(fatorial2$communality)) %>%
  rename(comunalidades = 1)

# Visualização das comunalidades com apenas os 'k' ('k' = 4) primeiros fatores
round(comunalidades2, 4) %>%
  kable() %>%
  kable_styling(bootstrap_options = "striped",
                full_width = FALSE,
                font_size = 20)


# Loading plot com as cargas dos dois primeiros fatores
cargas_fatoriais[, 1:2] %>% 
  data.frame() %>%
  rownames_to_column("variáveis") %>%
  ggplot(aes(x = PC1, y = PC2, label = variáveis)) +
  geom_point(color = "darkorchid",
             size = 3) +
  geom_text_repel() +
  ggtitle("Carga Fatorial dos dois primeiros fatores") +
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_vline(aes(xintercept = 0), linetype = "dashed", color = "orange") +
  geom_hline(aes(yintercept = 0), linetype = "dashed", color = "orange") +
  expand_limits(x= c(-1.25, 0.25), y=c(-0.25, 1)) +
  labs(x = "Principais Componentes 1",
       y = "Principais Componentes 2") +
  theme_bw()

library("FactoMineR")

res.pca <- PCA(desastres[,2:12], graph = FALSE)

eig.val <- get_eigenvalue(res.pca)
eig.val

fviz_eig(res.pca, addlabels = TRUE, ylim = c(0, 50))

var <- get_pca_var(res.pca)
var

fviz_contrib(res.pca, choice = "var", axes = 1, top = 10)

fviz_contrib(res.pca, choice = "var", axes = 2, top = 10)

fviz_contrib(res.pca, choice = "var", axes = 1:2, top = 10)

res.desc <- dimdesc(res.pca, axes = c(1,2), proba = 0.05)
# Description of dimension 1
res.desc$Dim.1

ind <- get_pca_ind(res.pca)
ind

fviz_pca_ind(res.pca)

fviz_pca_ind(res.pca, habillage = 96,
             addEllipses =TRUE, ellipse.type = "confidence",
             palette = "jco", repel = TRUE)


# encerrou a analise fatorial e agora será realizada a analise de cluster
# Adicionando os fatores extraídos no banco de dados original


# Análise de Cluster Utilizando os 4 Fatores
desastres <- bind_cols(desastres,
                    "fator_1" = fatores$PC1, 
                    "fator_2" = fatores$PC2,
                    "fator_3" = fatores$PC3,
                    "fator_4" = fatores$PC4)

# Análise de Cluster Utilizando os 4 Fatores

# Análise dos fatores (média e desvio padrão)
summary(desastres[,13:15])
sd(desastres[,13])
sd(desastres[,14])
sd(desastres[,15])

## ATENÇÃO: os clusters serão formados a partir dos 4 fatores
## Não aplicaremos o Z-Score, pois os fatores já são padronizados

# Matriz de dissimilaridades
matriz_D <- desastres[,13:15] %>% 
  dist(method = "euclidean")
print(matriz_D)

# Elaboração da clusterização hierárquica
cluster_hier <- agnes(x = matriz_D, method = "complete")
print(cluster_hier)

# Definição do esquema hierárquico de aglomeração

# As distâncias para as combinações em cada estágio
coeficientes <- sort(cluster_hier$height, decreasing = FALSE) 
coeficientes

# Tabela com o esquema de aglomeração. Interpretação do output:

## As linhas são os estágios de aglomeração
## Nas colunas Cluster1 e Cluster2, observa-se como ocorreu a junção
## Quando for número negativo, indica observação isolada
## Quando for número positivo, indica cluster formado anteriormente (estágio)
## Coeficientes: as distâncias para as combinações em cada estágio

esquema <- as.data.frame(cbind(cluster_hier$merge, coeficientes))
names(esquema) <- c("Cluster1", "Cluster2", "Coeficientes")
esquema
write.csv(desastres, file = "D:/OneDrive - defesacivil.rj.gov.br/USP/n supervisionadas exercicios PCA/desastres.csv", fileEncoding = "ISO-8859-1" , row.names = FALSE)

# Construção do dendrograma
dev.off()
fviz_dend(x = cluster_hier, show_labels = FALSE)

# Dendrograma com visualização dos clusters (definição de 10 clusters)
fviz_dend(x = cluster_hier,
          h = 3.0,
          show_labels = FALSE,
          color_labels_by_k = F,
          rect = F,
          rect_fill = F,
          ggtheme = theme_bw())

res.pca <- PCA(USArrests, ncp = 3, graph = FALSE)

fviz_dend(res.hcpc, show_labels = FALSE)

res.hcpc <- HCPC(res.pca, graph = FALSE)

fviz_cluster(res.hcpc, geom = "point", main = "Factor map")

fviz_dend(cluster_hier,
          cex = 0.7, # Label size
          palette = "jco", # Color palette see ?ggpubr::ggpar
          rect = TRUE, 
          rect_fill = TRUE, # Add rectangle around groups
          rect_border = "jco", # Rectangle color
          labels_track_height = 0.8 # Augment the room for labels
)


# Criando variável categórica para indicação do cluster no banco de dados
## O argumento 'k' indica a quantidade de clusters
desastres$cluster_H <- factor(cutree(tree = cluster_hier, k = 10))

# Análise de variância de um fator (ANOVA). Interpretação do output:

## Mean Sq do cluster_H: indica a variabilidade entre grupos
## Mean Sq dos Residuals: indica a variabilidade dentro dos grupos
## F value: estatística de teste (Sum Sq do cluster_H / Sum Sq dos Residuals)
## Pr(>F): p-valor da estatística 
## p-valor < 0.05: pelo menos um cluster apresenta média estatisticamente diferente dos demais

## A variável mais discriminante dos grupos contém maior estatística F (e significativa)

# ANOVA da variável 'fator 1'
summary(anova_fator_1 <- aov(formula = fator_1 ~ cluster_H,
                             data = desastres))

# ANOVA da variável 'fator 2'
summary(anova_fator_2 <- aov(formula = fator_2 ~ cluster_H,
                             data = desastres))

# ANOVA da variável 'fator 3'
summary(anova_fator_3 <- aov(formula = fator_3 ~ cluster_H,
                             data = desastres))

# ANOVA da variável 'fator 4'
summary(anova_fator_3 <- aov(formula = fator_4 ~ cluster_H,
                             data = desastres))


# Algumas estatísticas descritivas entre clusters

# precipitacação de 24h
group_by(desastres, cluster_H) %>%
  summarise(
    mean = mean(Precipitação.pluviométrica, na.rm = TRUE),
    sd = sd(Precipitação.pluviométrica, na.rm = TRUE),
    min = min(Precipitação.pluviométrica, na.rm = TRUE),
    max = max(Precipitação.pluviométrica, na.rm = TRUE),
    obs = n())

# afetados
group_by(desastres, cluster_H) %>%
  summarise(
    mean = mean(Afetados, na.rm = TRUE),
    sd = sd(Afetados, na.rm = TRUE),
    min = min(Afetados, na.rm = TRUE),
    max = max(Afetados, na.rm = TRUE),
    obs = n())

# Danos e prejuízos
group_by(desastres, cluster_H) %>%
  summarise(
    mean = mean(Danos.e.Prejuízos, na.rm = TRUE),
    sd = sd(Danos.e.Prejuízos, na.rm = TRUE),
    min = min(Danos.e.Prejuízos, na.rm = TRUE),
    max = max(Danos.e.Prejuízos, na.rm = TRUE),
    obs = n())

# FIM!
