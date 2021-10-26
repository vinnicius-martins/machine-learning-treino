'''
features (1 sim, 0 não)
pelo longo?
perna curta?
faz auau?
'''

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

model = LinearSVC()

porco1 = [0, 1, 0]
porco2 = [0, 1, 1]
porco3 = [1, 1, 0]

cachorro1 = [0, 1, 1]
cachorro2 = [1, 0, 1]
cachorro3 = [1, 1, 1]

# 1 - porco, 0 - cachorro
treino_x = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]
treino_y = [1, 1, 1, 0, 0, 0] # labels - etiqueta

model.fit(treino_x, treino_y)

# animal_misterioso = [1,1,1]
misterio1 = [1,1,1]
misterio2 = [1,1,0]
misterio3 = [0,1,1]

testes_x = [misterio1, misterio2, misterio3]
testes_y = [0,1,1]

previsoes = model.predict(testes_x)

# corretos = (previsoes==testes_classes).sum()
# total = len(testes)
# taxa_de_acerto = corretos/total

taxa_de_acerto = accuracy_score(testes_y, previsoes)*100
print(f'{taxa_de_acerto:.2f} %')