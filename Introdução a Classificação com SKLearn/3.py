import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

uri = 'https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv'
dados = pd.read_csv(uri)

a_renomear = {
    'expected_hours' : 'horas_esperadas',
    'price' : 'preco',
    'unfinished': 'nao_finalizado'
}
dados = dados.rename(columns=a_renomear)

troca = {
    0 : 1,
    1 : 0
}
dados['finalizado'] = dados.nao_finalizado.map(troca)

# sns.scatterplot(x='horas_esperadas',y='preco', hue="finalizado", data=dados)
# sns.relplot(x='horas_esperadas',y='preco', hue="finalizado", col="finalizado", data=dados)
# plt.show()

x = dados[['horas_esperadas', 'preco']]
y = dados['finalizado']

SEED = 5
np.random.seed(SEED)
treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, random_state= SEED, test_size=0.25, stratify=y)

print(f'\nTreinaremos com {len(treino_x)} e testaremos com {len(teste_x)} elementos.\n')


modelo = LinearSVC( )
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

previsoes_de_base = np.ones(540)
acuracia = accuracy_score(teste_y, previsoes_de_base) *100
print(f'A acurácia do algoritmo de baseline foi {acuracia:.2f}%')

taxa_de_acerto = accuracy_score(teste_y, previsoes) * 100
print(f'A taxa de acerto foi de {taxa_de_acerto:.2f}%') 

x_min = teste_x.horas_esperadas.min()
x_max = teste_x.horas_esperadas.max()
y_min = teste_x.preco.min()
y_max = teste_x.preco.max()
pixels = 100

eixo_x = np.arange(x_min, x_max, (x_max - x_min)/pixels)
eixo_y = np.arange(y_min, y_max, (y_max - y_min)/pixels)

xx, yy = np.meshgrid(eixo_x, eixo_y)
pontos = np.c_[xx.ravel(), yy.ravel()]

z = modelo.predict(pontos)
z = z.reshape(xx.shape)

# Decision Boundary
plt.contourf(xx, yy, z, alpha=0.3)
plt.scatter(teste_x.horas_esperadas, teste_x.preco, c=teste_y, s=1)
plt.show()

