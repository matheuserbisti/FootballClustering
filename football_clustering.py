### Libraries and Data Import

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import warnings
import scipy.stats as est
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering 
holandes = pd.read_csv("Holandes.csv")
portugues = pd.read_csv("Portugues.csv")

warnings.filterwarnings("ignore")


### Exploratory Analysis

## Victory Odds

# Selecting variables
# Home Teams
oddH = [holandes.B365H, portugues.B365H]
# Visitors
oddA = [holandes.B365A, portugues.B365A]

# Plots
fig, ((graf1, graf2), (graf3, graf4)) = plt.subplots(nrows = 2, ncols = 2, figsize = (9, 8))
hist = (graf1, graf2)
odds = [[holandes.B365H, portugues.B365H], [holandes.B365A, portugues.B365A]]
local = ["mandantes", "visitantes"]
saltos = [2,4]
amplitude = [18,45]
for i in range(2):
    hist[i].hist(odds[i], color = ["orange", "#7E0008"])
    hist[i].set_ylim(0, 250)
    hist[i].xaxis.set_ticks(np.arange(0, amplitude[i], saltos[i]))
    hist[i].set_title(str("Odds dos times " + local[i]))


cores = ["orange", "#7E0008", "black", "#006400"]
boxs = (graf3, graf4)

for i in range(2):
    for j in range(2):
        boxs[j].boxplot(odds[j][i], patch_artist = True, positions = [i + 1], 
                               boxprops = dict(facecolor = cores[i], color = cores[i + 2]),
                               capprops = dict(color = cores[i + 2]),
                               whiskerprops = dict(color = cores[i + 2]),
                               flierprops = dict(color = cores[i + 2], markeredgecolor = cores[i + 2]),
                               medianprops = dict(color = cores[i + 2]))


for i in (graf1, graf3):
    i.set_ylabel("Frequência")

for i in range(2):
    boxs[i].set_xticklabels(['Holandês', 'Português'])
plt.show()
# Table
dados = {'holandes_casa': holandes.B365H, 'portugues_casa': portugues.B365H,
         'holandes_fora': holandes.B365A, 'portugues_fora': portugues.B365A}

odds_v = pd.DataFrame(data = dados)

odds_v.describe()


## Draw Odds

# Selecting variables
oddD = [holandes.B365D, portugues.B365D]

# Plot details
cores = ["orange", "#7E0008", "black", "#006400"]
legendas = ["Holandês", "Português"]

# Plots
fig, (graf1, graf2) = plt.subplots(nrows = 1, ncols = 2, figsize = (9, 4))

graf1.hist(oddD, color = ["orange", "#7E0008"])

for i in range(2):
    graf2.boxplot(oddD[i], patch_artist = True, positions = [i + 1], 
                               boxprops = dict(facecolor = cores[i], color = cores[i + 2]),
                               capprops = dict(color = cores[i + 2]),
                               whiskerprops = dict(color = cores[i + 2]),
                               flierprops = dict(color = cores[i + 2], markeredgecolor = cores[i + 2]),
                               medianprops = dict(color = cores[i + 2]))
      
graf1.set_title('Odds de empate')
graf1.set_ylabel("Frequência")
graf2.set_xticklabels(legendas)

plt.show()

# Table
dados = {'holandes_empate': holandes.B365D, 'portugues_empate': portugues.B365D}

odds_e = pd.DataFrame(data = dados)

odds_e.describe()


## Goals per match

plt.style.use('seaborn-white')
# Selecting variables
gols = [[holandes.FTHG, portugues.FTHG], [holandes.FTAG, portugues.FTAG]]
# Plot details
cores = ["orange", "#7E0008", "black", "#006400"]
legendas = ["Holandês", "Português"]
local = ["mandantes", "visitantes"]

# Plots
fig, (graf1, graf2) = plt.subplots(nrows = 1, ncols = 2, figsize = (9, 4), sharey = True)

grafs = (graf1, graf2)

for i in range(2):
    for j in range(2):
        grafs[j].boxplot(gols[j][i], patch_artist = True, positions = [i + 1], 
                               boxprops = dict(facecolor = cores[i], color = cores[i + 2]),
                               capprops = dict(color = cores[i + 2]),
                               whiskerprops = dict(color = cores[i + 2]),
                               flierprops = dict(color = cores[i + 2], markeredgecolor = cores[i + 2]),
                               medianprops = dict(color = cores[i + 2]))
            
for i in range(2):
    grafs[i].set_title(str('Gols de ' + local[i]))
    grafs[i].set_xticklabels(["Holandês", "Português"])

plt.show()

# Table
gols = {'holandes_casa':holandes.FTHG, 'portugues_casa':portugues.FTHG,
        'holandes_fora':holandes.FTAG, 'portugues_fora':portugues.FTAG}

FTHG = pd.DataFrame(data = gols)
FTHG.describe()


## Shots per match

# Labels
campeonatos = [holandes, portugues]
cores = ["orange", "#7E0008"]
legenda = ["Holandês", "Português"]

# Plots
fig, (graf1, graf2) = plt.subplots(nrows = 1, ncols = 2, figsize = (9, 4), sharey = True)

for i in range(len(campeonatos)):
    counts1, bins1 = np.histogram(campeonatos[i].HS, bins = range(0, 40, 1))
    counts2, bins2 = np.histogram(campeonatos[i].AS, bins = range(0, 40, 1))
    
    graf1.plot(bins1[:-1] + 1, counts1, color = cores[i], label = legenda[i])
    graf1.set_ylabel('Frequência')
    graf1.set_title('Chutes na partida (Mandantes)')
    
    graf2.plot(bins2[:-1] + 1, counts2, color = cores[i], label = legenda[i])
    graf2.set_title('Chutes na partida (Visitantes)')
    
plt.legend()
plt.show()

# Table
cht = {"chutes_mandante_holanda":holandes.HS, "chutes_mandante_portugal":portugues.HS,
         "chutes_visitante_holanda":holandes.AS, "chutes_visitante_portugal":portugues.AS}

chutes = pd.DataFrame(data = cht)
chutes.describe()


## Shots precision

# Variables lists
precisao_holandes_casa = []
precisao_portugues_casa = []

precisao_holandes_fora = []
precisao_portugues_fora = []

# Calculating values
for i in range(len(holandes)):
    precisao_casa_hd = holandes.HST[i] / holandes.HS[i]
    precisao_casa_pt = portugues.HST[i] / portugues.HS[i]
    
    precisao_fora_hd = holandes.AST[i] / holandes.AS[i]
    precisao_fora_pt = portugues.AST[i] / portugues.AS[i]
    
    
    precisao_holandes_casa.append(precisao_casa_hd)
    precisao_portugues_casa.append(precisao_casa_pt)
    
    precisao_holandes_fora.append(precisao_fora_hd)
    precisao_portugues_fora.append(precisao_fora_pt)
    
# Precisions list
precisoes = [[precisao_holandes_casa, precisao_holandes_fora],
             [precisao_portugues_casa, precisao_portugues_fora]]

# Plot colors
cores = ["orange", "#7E0008"]

# Plots
fig, grafs = plt.subplots(nrows = 2, ncols = 2, figsize = (9, 8), sharey = True)

bins = np.linspace(0, 1, 10)
for i in range(2):
    for j in range(2):
        grafs[i, j].hist(precisoes[i][j], bins, rwidth = 1, color = cores[i])

local = ["(Mandantes)", "(Visitantes)"]
for i in range(2):
    grafs[0, i].set_title(str("Precisão de chutes no gol " + local[i]))
    grafs[i, 0].set_ylabel("Frequência")

plt.show()

# Table
pcs = {"precisao_mandante_holanda":precisao_holandes_casa, "precisao_visitante_holanda":precisao_holandes_fora,
       "precisao_mandante_portugal":precisao_portugues_casa, "precisao_visitante_portugal":precisao_portugues_fora}

precisao = pd.DataFrame(data = pcs)
precisao.describe()



## Match Results

# Creating variables
resultados_holandes = [sum(holandes.FTR == "H"), sum(holandes.FTR == "A"), sum(holandes.FTR == "D")]
resultados_portugues = [sum(portugues.FTR == "H"), sum(portugues.FTR == "A"), sum(portugues.FTR == "D")]
# Características dos gráficos
legenda = ["Mandantes", "Visitantes", "Empates"]
cores = ["#2FA4E2", "#FF7FAB", "gray"]

# Plots
fig, (graf1, graf2) = plt.subplots(nrows = 1, ncols = 2, figsize = (9, 4))

graf1.pie(resultados_holandes, autopct='%1.1f%%', labels = legenda, shadow = True, startangle = 90, colors = cores)
graf2.pie(resultados_portugues, autopct='%1.1f%%', labels = legenda, shadow = True, startangle = 90, colors = cores)

graf1.axis('equal')
graf2.axis('equal')

graf1.set_title("Vitórias do Campeonato Holandês")
graf2.set_title("Vitórias do Campeonato Português")

plt.show()


## Draw matches with goals

# Creating variables
empates_hd = holandes[holandes.FTHG == holandes.FTAG]
empates_pt = portugues[portugues.FTHG == portugues.FTAG]

# Removing indexes
empates_hd.reset_index(inplace = True, drop = True)
empates_pt.reset_index(inplace = True, drop = True)

# Counter
com_gols_hd = 0
com_gols_pt = 0

for jogo in range(len(empates_hd)):
    if empates_hd.FTHG[jogo] != 0:
        com_gols_hd += 1

for jogo in range(len(empates_pt)):
    if empates_pt.FTHG[jogo] != 0:
        com_gols_pt += 1

legenda = ['Sem gols', 'Com gols']
hd = [len(empates_hd) - com_gols_hd, com_gols_hd]
pt = [len(empates_pt) - com_gols_pt, com_gols_pt]

# Plots
x = np.arange(len(legenda))  
width = 0.35  

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, hd, width, label = 'Holandês', color = "orange")
rects2 = ax.bar(x + width/2, pt, width, label = 'Português', color = "#7E0008")

ax.set_ylabel('Frequência')
ax.set_title('Empates')
ax.set_xticks(x)
ax.set_xticklabels(legenda)
ax.legend()

fig.tight_layout()

plt.show()

# Table
empt = {'holandes_empates':empates_hd.FTHG, 'portugues_empates':empates_pt.FTHG}
empates = pd.DataFrame(data = empt)
empates.describe()



## Fouls per match

# Variables lists
faltas_dentro = [holandes.HF, portugues.HF]
faltas_fora = [holandes.AF, portugues.AF]
faltas = [faltas_dentro, faltas_fora]

# Plot details
cores = ["orange", "#7E0008", "black", "#006400"]
legendas = ["Holandês", "Português"]
local = ["mandantes", "visitantes"]

# Plots
fig, (graf1, graf2) = plt.subplots(nrows = 1, ncols = 2, figsize = (9, 4), sharey = True)
grafs = (graf1, graf2)

for i in range(2):
    for j in range(2):
        grafs[j].boxplot(faltas[j][i], patch_artist = True, positions = [i + 1], 
                               boxprops = dict(facecolor = cores[i], color = cores[i + 2]),
                               capprops = dict(color = cores[i + 2]),
                               whiskerprops = dict(color = cores[i + 2]),
                               flierprops = dict(color = cores[i + 2], markeredgecolor = cores[i + 2]),
                               medianprops = dict(color = cores[i + 2]))
for i in range(2):
    grafs[i].set_title(str('Faltas de ' + local[i]))
    grafs[i].set_xticklabels(["Holandês", "Português"])

plt.show()

# Table
flt = {'holandes_casa':holandes.HF, 'portugues_casa':portugues.HF,
        'holandes_fora':holandes.AF, 'portugues_fora':portugues.AF}

faltas = pd.DataFrame(data = flt)

faltas.describe()



## Yellow and red cards

# Tables
dados1 = {"Eredivisie": [sum(holandes.HY), sum(holandes.AY), sum(holandes.HR), sum(holandes.AR)],
          "Primeira Liga": [sum(portugues.HY), sum(portugues.AY), sum(portugues.HR), sum(portugues.AR)]}

df = pd.DataFrame(dados1,
                  index =['Cartões amarelos Mandantes','Cartões amarelos Visitantes',
                          'Cartões vermelhos Mandantes','Cartões vermelhos visitantes'],)  
df




### Hypothesis testing

## Mean of goals per match

# Variable lists
gols_partida_holandes = []
gols_partida_portugues = []

# Values
for i in range(len(holandes)):
    gols_partida_holandes.append(holandes.FTHG[i] + holandes.FTAG[i])
    gols_partida_portugues.append(portugues.FTHG[i] + portugues.FTAG[i])

# Levene's test for variance
est.levene(gols_partida_holandes, gols_partida_portugues)

# Independent samples t-test
est.ttest_ind(gols_partida_holandes, gols_partida_portugues)


## Shots mean per match

# Variable lists
chutegol_partida_holandes = []
chutegol_partida_portugues = []

# Values
for i in range(len(holandes)):
    chutegol_partida_holandes.append(holandes.HST[i] + holandes.AST[i])
    chutegol_partida_portugues.append(portugues.HST[i] + portugues.AST[i])

# Levene's variance test
est.levene(chutegol_partida_holandes, chutegol_partida_portugues)

# Independent samples t-test
est.ttest_ind(chutegol_partida_holandes, chutegol_partida_portugues, equal_var = False)



## Fouls mean per match

# Variables lists
faltas_partida_holandes = []
faltas_partida_portugues = []

# Values
for i in range(len(holandes)):
    faltas_partida_holandes.append(holandes.HF[i] + holandes.AF[i])
    faltas_partida_portugues.append(portugues.HF[i] + portugues.AF[i])

# Levene's variance test
est.levene(faltas_partida_holandes, faltas_partida_portugues)

# Independent samples t-test
est.ttest_ind(faltas_partida_portugues, faltas_partida_holandes, equal_var = False)



## Odds and shots Spearman's correlation

pais = ["holandeses", "portugueses"]
liga = [holandes, portugues]

for i in range(2):
    print("Para times", pais[i], "mandantes, o teste resultou em: \n", est.spearmanr(liga[i].HS, liga[i].B365H))

    print()

    print("Para times", pais[i], "visitantes, o teste resultou em: \n", est.spearmanr(liga[i].AS, liga[i].B365A))

    print()

# correlation table
rhos = {"Mandantes": [-0.542, -0.45], "Visitantes": [-0.536, -0.43]}

corr = pd.DataFrame(rhos, index = ['Eredivisie','Primeira Liga']) 
                                  
corr



### Clusterization

## Eredivisie Odds cluster

# Home team odds
A = holandes.groupby("HomeTeam").sum().sort_values(["B365H"], ascending = True).B365H
# Away team odds
B = holandes.groupby("AwayTeam").sum().sort_values(["B365A"],ascending = True).B365A
# Variables
ere = pd.DataFrame(data={'Odds como mandante': A, 'Odds como visitante': B})
# Euclidian dendogram
dendrograma = sch.dendrogram (sch.linkage (ere, method = "ward")) 
plt.title ('Dendograma da liga holandesa') 
plt.xlabel ('Equipes') 
plt.ylabel ('Distâncias euclidianas') 
plt.show()

# Scatter plot
# Separation
hc = AgglomerativeClustering (n_clusters = 4, affinity = 'euclidean', linkage = 'ward')
# Values
X = ere.iloc[:, [0,1]].values
# Prediction
y_hc = hc.fit_predict(X)

# Plot
plt.scatter(X[y_hc==0, 0], X[y_hc==0, 1],s=80, c='limegreen', label ='Cluster 1')
plt.scatter(X[y_hc==1, 0], X[y_hc==1, 1],s=80, c='midnightblue', label ='Cluster 2')
plt.scatter(X[y_hc==2, 0], X[y_hc==2, 1], s=80,c='cyan', label ='Cluster 3')
plt.scatter(X[y_hc==3, 0], X[y_hc==3, 1],s=80, c='crimson', label ='Cluster 4')
plt.title ('Clusters das equipes(Modelo de clusterização hierárquica)') 
plt.xlabel ('Odds como mandante') 
plt.ylabel ('Odds como visitante') 
plt.legend(loc="lower right", title="")
plt.show()

# Table
ere["Cluster"] = y_hc+1
ere.sort_values(["Cluster", "Odds como mandante"], ascending = True)



## Primeira Liga Odds cluster

# Home team odds
C = portugues.groupby("HomeTeam").sum().sort_values(["B365H"], ascending = True).B365H
# Away team odds
D = portugues.groupby("AwayTeam").sum().sort_values(["B365A"],ascending = True).B365A
# Variables
nos = pd.DataFrame(data={'Odds como mandante': C, 'Odds como visitante': D})
# Euclidian dendogram
dendrograma = sch.dendrogram (sch.linkage (nos, method = "ward")) 
plt.title ('Dendograma da liga portuguesa') 
plt.xlabel ('Equipes') 
plt.ylabel ('Distâncias euclidianas') 
plt.show ()

# Scatter plot
# Separation
hc = AgglomerativeClustering (n_clusters = 4, affinity = 'euclidean', linkage = 'ward')
# Values
X = nos.iloc[:, [0,1]].values
# Prediction
y_hc = hc.fit_predict(X)

# Plot
plt.scatter(X[y_hc==0, 0], X[y_hc==0, 1],s=80, c='limegreen', label ='Cluster 1')
plt.scatter(X[y_hc==1, 0], X[y_hc==1, 1],s=80, c='midnightblue', label ='Cluster 2')
plt.scatter(X[y_hc==2, 0], X[y_hc==2, 1], s=80,c='cyan', label ='Cluster 3')
plt.scatter(X[y_hc==3, 0], X[y_hc==3, 1],s=80, c='crimson', label ='Cluster 4')
plt.title ('Clusters das equipes(Modelo de clusterização hierárquica)') 
plt.xlabel ('Odds como mandante') 
plt.ylabel ('Odds como visitante') 
plt.legend(loc="lower right", title="")
plt.show()

# Table
nos["Cluster"] = y_hc+1
nos.sort_values(["Cluster", "Odds como mandante"], ascending = True)


## Both leagues Odds cluster

# Values
ambas = nos.append(ere)
# Euclidian dendogram
dendrograma = sch.dendrogram (sch.linkage (ambas, method = "ward")) 
plt.title ('Dendograma de todas equipes') 
plt.xlabel ('Equipes') 
plt.ylabel ('Distâncias euclidianas') 
plt.show ()


# Scatter plot

# Separation
hc = AgglomerativeClustering (n_clusters = 4, affinity = 'euclidean', linkage = 'ward')
# Values
X = ambas.iloc[:, [0,1]].values
# Prediction
y_hc = hc.fit_predict(X)

# Scatter plot
plt.scatter(X[y_hc==0, 0], X[y_hc==0, 1],s=80, c='limegreen', label ='Cluster 1')
plt.scatter(X[y_hc==1, 0], X[y_hc==1, 1],s=80, c='midnightblue', label ='Cluster 2')
plt.scatter(X[y_hc==2, 0], X[y_hc==2, 1], s=80,c='cyan', label ='Cluster 3')
plt.scatter(X[y_hc==3, 0], X[y_hc==3, 1],s=80, c='crimson', label ='Cluster 4')
plt.title ('Clusters das equipes(Modelo de clusterização hierárquica)') 
plt.xlabel ('Odds como mandante') 
plt.ylabel ('Odds como visitante') 
plt.legend(loc="lower right", title="")
plt.show()

# Table
ambas["Cluster"] = y_hc+1
ambas.sort_values(["Cluster", "Odds como mandante"], ascending = True)


### Goals cluster

## Primeira Liga
# Home goals
gmc = portugues.groupby("HomeTeam").sum().sort_values(["FTHG"], ascending = False).FTHG
# Away goals
gmf = portugues.groupby("AwayTeam").sum().sort_values(["FTAG"], ascending = False).FTAG
# Home conceded goals
gsc = portugues.groupby("HomeTeam").sum().sort_values(["FTAG"], ascending = False).FTAG
# Away conceded goals
gsf = portugues.groupby("AwayTeam").sum().sort_values(["FTHG"], ascending = False).FTHG

## Eredivisie
# Home goals
gmc1 = holandes.groupby("HomeTeam").sum().sort_values(["FTHG"], ascending = False).FTHG
# Away goals
gmf1 = holandes.groupby("AwayTeam").sum().sort_values(["FTAG"], ascending = False).FTAG
# Home conceded goals
gsc1 = holandes.groupby("HomeTeam").sum().sort_values(["FTAG"], ascending = False).FTAG
# Away conceded goals
gsf1 = holandes.groupby("AwayTeam").sum().sort_values(["FTHG"], ascending = False).FTHG

# Primeira Liga goals data
nosg = pd.DataFrame(data={'Gols marcados': gmc + gmf, 'Gols sofridos': gsc+ gsf})
# Eredivisie goals data
ereg = pd.DataFrame(data={'Gols marcados': gmc1 + gmf1, 'Gols sofridos': gsc1+ gsf1})
# Both leagues data
ambasg = nosg.append(ereg)

# Euclidian dendogram
dendrogram = sch.dendrogram (sch.linkage (ambasg, method = "ward")) 
plt.title('Dendograma') 
plt.xlabel('Equipes') 
plt.ylabel('Distâncias Euclidianas') 
plt.show ()

# Scatter plot
# Separation
hc = AgglomerativeClustering (n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
# Values
X = ambasg.iloc[:, [0,1]].values
# Prediction
y_hc = hc.fit_predict(X)

# Plot
plt.scatter(X[y_hc==0, 0], X[y_hc==0, 1],s=80, c='limegreen', label ='Cluster 1')
plt.scatter(X[y_hc==1, 0], X[y_hc==1, 1],s=80, c='midnightblue', label ='Cluster 2')
plt.scatter(X[y_hc==2, 0], X[y_hc==2, 1], s=80,c='cyan', label ='Cluster 3')
plt.scatter(X[y_hc==3, 0], X[y_hc==3, 1],s=80, c='crimson', label ='Cluster 4')
plt.scatter(X[y_hc==4, 0], X[y_hc==4, 1],s=80, c='black', label ='Cluster 5')
plt.title ('Clusters das equipes(Modelo de clusterização hierárquica)') 
plt.xlabel ('Gols marcados') 
plt.ylabel ('Gols sofridos') 
plt.legend(loc="upper right", title="")
plt.show()

# Table
ambasg["Cluster"] = y_hc+1
ambasg.sort_values(["Cluster", "Gols marcados"], ascending = False)