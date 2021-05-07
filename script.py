import numpy as np
import matplotlib.pyplot as plt

#Dados gerados considerando uma função horária v(t) = 4 + 8*t + erro
tempos      = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
velocidades = np.array([14.1, 20.2, 27.2, 36.4, 47.1, 48.9, 65.2, 72.5, 74.4])

plt.figure(dpi=150)
plt.scatter(tempos, velocidades)
plt.xlabel('Tempo (s)')
plt.ylabel('Velocidade (m/s)')
plt.grid(alpha=0.3)
plt.show()

from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

#Estimação da função horária da velocidade
#Estima os coeficientes e a função horária da velocidade
reg = LinearRegression().fit(tempos.reshape(-1, 1), velocidades)
v0, a = reg.intercept_, reg.coef_[0] #coeficientes
v     = v0 + a * tempos    #função horária

print('Velocidade no instante 0: %.2f' %v0)
print('Aceleração: %.2f' %a)
print('MAE: %.2f' %mean_absolute_error(v, velocidades))
print('R²: %.2f' %r2_score(v, velocidades))

#Mostra o resultado
plt.figure(dpi=150)
plt.scatter(tempos, velocidades)
plt.plot(tempos, v, color='r')
plt.xlabel('Tempo (s)')
plt.ylabel('Velocidade (m/s)')
plt.grid(alpha=0.3)
plt.show()

#Dados dos 70 pinheiros
diametro = np.array([4.4, 4.6, 5.0, 5.1, 5.1, 5.2, 5.2, 5.5, 5.5, 5.6, 5.9, 5.9, 7.5, 7.6, 7.6, 
                     7.8, 8.0, 8.1, 8.4, 8.6, 8.9, 9.1, 9.2, 9.3, 9.3, 9.8, 9.9, 9.9, 9.9, 10.1, 
                     10.2, 10.2, 10.3, 10.4, 10.6, 11.0, 11.1, 11.2, 11.5, 11.7, 12.0, 12.2, 12.2, 
                     12.5, 12.9, 13.0, 13.1, 13.1, 13.4, 13.8, 13.8, 14.3, 14.3, 14.6, 14.8, 14.9, 
                     15.1, 15.2, 15.2, 15.3, 15.4, 15.7, 15.9, 16.0, 16.8, 17.8, 18.3, 18.3, 19.4, 23.4])

volume   = np.array([2.0, 2.2, 3.0, 4.3, 3.0, 2.9, 3.5, 3.4, 5.0, 7.2, 6.4, 5.6, 7.7, 10.3, 8.0, 12.1, 11.1, 
                     16.8, 13.6, 16.6, 20.2, 17.0, 17.7, 19.4, 17.1, 23.9, 22.0, 23.1, 22.6, 22.0, 27.0, 27.0, 
                     27.4, 25.2, 25.5, 25.8, 32.8, 35.4, 26.0, 29.0, 30.2, 28.2, 32.4, 41.3, 45.2, 31.5, 37.8, 
                     31.6, 43.1, 36.5, 43.3, 41.3, 58.9, 65.6, 59.3, 41.4, 61.5, 66.7, 68.2, 73.2, 65.9, 55.5, 
                     73.6, 65.9, 71.4, 80.2, 93.8, 97.9, 107.0, 163.5])

#Mostra a dispersão dos dados
plt.figure(dpi=150)
plt.scatter(diametro, volume)
plt.xlabel('Diâmetro (pol)')
plt.ylabel('Volume (pés³)')
plt.grid(alpha=0.3)
plt.show()

#Tenta ajustar uma reta de regressão para estimar o volume da árvore
reg = LinearRegression().fit(diametro.reshape(-1, 1), volume)
beta0, beta1 = reg.intercept_, reg.coef_[0] #coeficientes
volume_estimado = beta0 + beta1 * diametro

print('Coeficientes estimados')
print('Beta 0: %.4f' %beta0)
print('Beta 1: %.4f' %beta1)
print('MAE: %.2f' %mean_absolute_error(volume, volume_estimado))
print('R²: %.2f' %r2_score(volume, volume_estimado))

#Mostra o resultado
plt.figure(dpi=150)
plt.scatter(diametro, volume)
plt.plot(diametro, volume_estimado, color='r')
plt.xlabel('Diâmetro (pol)')
plt.ylabel('Volume (pés³)')
plt.grid(alpha=0.3)
plt.show()

#Aplicamos a solução acima para resolver o problema proposto

#Aplica a função logartmio na base de Euler as variáveis
diametro_log = np.log(diametro)
volume_log   = np.log(volume)

#Mostra os valores na escala original e logartmica
plt.figure(dpi=150, figsize=(10,5))
plt.subplot(221)
plt.title('Dados Originais')
plt.scatter(diametro, volume)
plt.xlabel('Diâmetro (pol)')
plt.ylabel('Volume (pés³)')
plt.grid(alpha=0.3)
plt.subplot(223)
plt.hist(volume, bins=10)
plt.xlabel('Volume (pés³)')
plt.subplot(222)
plt.title('Dados Transformados')
plt.scatter(diametro_log , volume_log)
plt.xlabel('Log do Diâmetro (pol)')
plt.ylabel('Log do Volume (pés³)')
plt.grid(alpha=0.3)
plt.subplot(224)
plt.hist(volume_log, bins=10)
plt.xlabel('Volume (pés³)')
plt.tight_layout()
plt.show()

#Calcula os coeficientes beta nos dados transformados
reg_log = LinearRegression().fit(diametro_log.reshape(-1, 1), volume_log)
betal0, betal1 = reg_log.intercept_, reg_log.coef_[0] #coeficientes

#Estima os volumes (y) utilizando o modelo na escala logarítmica
#Observe que aplicamos os valores originais do diâmetro (x) utilizando os coeficientes estimados no espaço transformado
volume_estimado_log = np.exp(betal0) * diametro**betal1

print('Estimação dados originais')
print('Beta 0: %.4f' %beta0)
print('Beta 1: %.4f' %beta1)
print('MAE: %.2f' %mean_absolute_error(volume, volume_estimado))
print('R²: %.2f' %r2_score(volume, volume_estimado))
print('')
print('Estimação dados transformados')
print('Beta 0: %.4f' %betal0)
print('Beta 1: %.4f' %betal1)
print('MAE: %.2f' %mean_absolute_error(volume, volume_estimado_log))
print('R²: %.2f' %r2_score(volume, volume_estimado_log))

#Formata os modelos estimados
modelo_linear = '$y = ' + str(np.round(beta0, 2)) + ' + ' + str(np.round(beta1, 2)) + 'x$'
modelo_loglog = '$y = e^{' + str(np.round(betal0, 2)) + '} x^{' + str(np.round(betal1, 2)) + '}$'

#Mostra o resultado
plt.figure(dpi=150, figsize=(10,5))
plt.subplot(121)
plt.title('Estimação Dados Originais \n' + modelo_linear)
plt.scatter(diametro, volume)
plt.plot(diametro, volume_estimado, color='r')
plt.xlabel('Diâmetro (pol)')
plt.ylabel('Volume (pés³)')
plt.grid(alpha=0.3)
plt.subplot(122)
plt.title('Estimação Dados Transformados \n' + modelo_loglog)
plt.scatter(diametro, volume)
plt.plot(diametro, volume_estimado_log, color='r')
plt.xlabel('Diâmetro (pol)')
plt.ylabel('Volume (pés³)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

def CA(y):
  '''
  Coeficiente de Assimetria
  '''
  return (1/len(y)) * np.sum(((y - np.mean(y)) / np.std(y))**3)

print('Coeficiente de Assimetria: %.2f' % CA(volume))

#Mostra a distribuição do volume
plt.figure(dpi=150)
plt.hist(volume, bins=10)
plt.xlabel('Volume (pés³)')
plt.show()

import pandas as pd

#Lê os dados do arquivo csv e removes valores NaN e valores nulos
df = pd.read_csv('Fish.csv')
df.dropna(inplace=True)
df = df[(df[['Weight','Width']] != 0).all(axis=1)] #IMPORTANTÍSSIMO remover valores nulos, porque log(0) = infinito

#variável resposta (y)
peso_peixe = np.array(df['Weight'])

#variável entrada (x)
largura_peixe = np.array(df['Width'])

#Calcula o coeficiente de assimetria
print('Coeficiente de Assimetria: %.2f' % CA(peso_peixe))

#Mostra o gráfico de dispersão e o histograma
plt.figure(dpi=150, figsize=(10,5))
plt.subplot(121)
plt.title('Dados Originais')
plt.scatter(largura_peixe, peso_peixe)
plt.ylabel('Peso (g)')
plt.xlabel('Largura (cm)')
plt.grid(alpha=0.3)
plt.subplot(122)
plt.hist(peso_peixe, bins=10)
plt.xlabel('Peso (g)')
plt.tight_layout()
plt.show()

#Import módulo para separa os dados
from sklearn.model_selection import train_test_split

#Separa os dados em treinamento e teste
largura_peixe_train, largura_peixe_test, peso_peixe_train, peso_peixe_test = train_test_split(largura_peixe, peso_peixe, test_size=0.3, random_state=1)

#Ajusta a reta de regressão linear aos dados de treino
reg = LinearRegression().fit(largura_peixe_train.reshape(-1, 1), peso_peixe_train)
beta0, beta1 = reg.intercept_, reg.coef_[0] #coeficientes

#Faz a predição no conjunto de teste
peso_peixe_estimado = reg.predict(largura_peixe_test.reshape(-1, 1))

#Calcula as métricas no conjunto de teste
print('Resultado - Dados originais')
print('Beta 0: %.4f' %beta0)
print('Beta 1: %.4f' %beta1)
print('MAE: %.2f' %mean_absolute_error(peso_peixe_test , peso_peixe_estimado))
print('R²: %.2f' %r2_score(peso_peixe_test , peso_peixe_estimado))

#Aplica a transformação Logaritmica
largura_peixe_train_log = np.log(largura_peixe_train) 
peso_peixe_train_log    = np.log(peso_peixe_train)
largura_peixe_test_log  = np.log(largura_peixe_test) 
peso_peixe_test_log     = np.log(peso_peixe_test)

#Ajusta o modelo de regressão aos dados de treino na escala logarítmica
reg_log = LinearRegression().fit(largura_peixe_train_log.reshape(-1, 1), peso_peixe_train_log)
betal0, betal1 = reg_log.intercept_, reg_log.coef_[0] #coeficientes

#Faz a predição
peso_peixe_estimado_log = np.exp(betal0) * largura_peixe_test**betal1

#Calcula as métricas no conjunto de teste
print('')
print('Resultado - Dados transformados')
print('Beta 0: %.4f' %betal0)
print('Beta 1: %.4f' %betal1)
print('MAE: %.2f' %mean_absolute_error(peso_peixe_test, peso_peixe_estimado_log))
print('R²: %.2f' %r2_score(peso_peixe_test, peso_peixe_estimado_log))

#Ordena os dados para melhor representação gráfica
df_ord = pd.DataFrame({'Largura' : largura_peixe_test, 'Peso' : peso_peixe_estimado_log})
df_ord = df_ord.sort_values(by=['Largura'])

#Mostra o resultado
plt.figure(dpi=150)
plt.title('Estimação')
plt.scatter(largura_peixe_train, peso_peixe_train, alpha=0.5, label='Treino')
plt.scatter(largura_peixe_test, peso_peixe_test, alpha=0.8, label='Teste')
plt.plot(largura_peixe_test, peso_peixe_estimado, color='r', label='Linear')
plt.plot(df_ord['Largura'], df_ord['Peso'], color='g', label='Logarítmico')
plt.xlabel('Peso (g)')
plt.ylabel('Largura (cm)')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

#Lê os dados do arquivo csv e removes valores NaN
df = pd.read_csv('healthcare-dataset-stroke-data.csv')
df.dropna(inplace=True)

#variável resposta (y)
nivel_glicose = np.array(df['avg_glucose_level'])

#variável entrada (x)
imc = np.array(df['bmi'])

#Calcula o coeficiente de assimetria
print('Coeficiente de Assimetria: %.2f' % CA(nivel_glicose))

#Mostra o gráfico de dispersão e o histograma
plt.figure(dpi=150, figsize=(10,5))
plt.subplot(121)
plt.title('Dados Originais')
plt.scatter(imc, nivel_glicose, alpha=0.8)
plt.ylabel('Nível Glicose (mg/dL)')
plt.xlabel('IMC (kg/m²)')
plt.grid(alpha=0.3)
plt.subplot(122)
plt.hist(nivel_glicose, bins=10)
plt.xlabel('Nível Glicose (mg/dL)')
plt.tight_layout()
plt.show()

#Separa os dados em treinamento e teste
imc_train, imc_test, nivel_glicose_train, nivel_glicose_test = train_test_split(imc, nivel_glicose, test_size=0.3, random_state=0)

#Ajusta a reta de regressão aos dados de treino
reg = LinearRegression().fit(imc_train.reshape(-1, 1), nivel_glicose_train)
beta0, beta1 = reg.intercept_, reg.coef_[0] #coeficientes

#Faz a predição
nivel_glicose_estimado = reg.predict(imc_test.reshape(-1, 1))

#Calcula as métricas no conjunto de teste
print('Resultado - Dados originais')
print('Beta 0: %.4f' %beta0)
print('Beta 1: %.4f' %beta1)
print('MAE: %.2f' %mean_absolute_error(nivel_glicose_test , nivel_glicose_estimado))
print('R²: %.2f' %r2_score(nivel_glicose_test , nivel_glicose_estimado))

#Aplica a transformação Logaritmica
imc_train_log           = np.log(imc_train) 
nivel_glicose_train_log = np.log(nivel_glicose_train)
imc_test_log            = np.log(imc_test) 
nivel_glicose_test_log  = np.log(nivel_glicose_test)

#Ajusta o modelo de regressão aos dados de treino na escala logarítmica
reg_log = LinearRegression().fit(imc_train_log.reshape(-1, 1), nivel_glicose_train_log)
betal0, betal1 = reg_log.intercept_, reg_log.coef_[0] #coeficientes

#Faz a predição
nivel_glicose_estimado_log = np.exp(betal0) * imc_test**betal1

#Calcula as métricas no conjunto de teste
print('')
print('Resultado - Dados transformados')
print('Beta 0: %.4f' %betal0)
print('Beta 1: %.4f' %betal1)
print('MAE: %.2f' %mean_absolute_error(nivel_glicose_test, nivel_glicose_estimado_log))
print('R²: %.2f' %r2_score(nivel_glicose_test, nivel_glicose_estimado_log))