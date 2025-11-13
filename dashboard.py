import streamlit as st
import numpy as np
import pandas as pd
import joblib

# === Carregar modelo ===
dados = joblib.load("MLP/modelos/MultiLayerPerceptron_NaoLinear/modelo_RM_nlinear_v1_16.33.pkl")
modelo = dados['modelo']
scaler_variaveis = dados['scaler_variavel']
scaler_target = dados['scaler_target']
mape = dados['MAPE']

st.title("Simulador")
st.subheader("Modelo: Multi Layer Perceptron - Rede Neural")
st.caption("""O modelo utilizado é uma Rede Neural Artificial do tipo Multi Layer Perceptron (MLP) com duas camadas ocultas.
           Ele aprende relações não lineares entre as variáveis de entrada e a variável alvo, ajustando automaticamente seus parâmetros para reduzir o erro nas previsões.
           O uso da função de ativação ReLU e da técnica de Dropout ajuda o modelo a capturar padrões complexos sem perder capacidade de generalização.""")
st.markdown("Preencha os dados da nova localidade.")

col1, col2 = st.columns(2)
with col1:
    anos = st.number_input("Anos de Atividade", min_value=1, step=1)
    populacao = st.number_input("População: ", min_value=1)
    fluxoPassantes = st.number_input("Fluxo de Passantes por Semana: ", min_value=1)
    densidadeDemografica = st.number_input("Densidade Demográfica: ", min_value=1)
    rendaMediaDomiciliar = st.number_input("Renda Média Domiciliar (R$): ", min_value=1)
    peaDia = st.number_input("PEA Dia: ", min_value=1)
    superiorCompleto = st.number_input("Superior Completo: ", min_value=1)
with col2:
    potencialConsumoTotal = st.number_input("Potencial de Consumo Total (R$)", min_value=1)
    potencialMedioDomicilio = st.number_input("Potencial de Consumo Médio por Domicilio (R$):", min_value=1)
    homens = st.number_input("População de Homens: ", min_value=1)
    mulheres = st.number_input("População de Mulheres: ", min_value=1)
    domiciliosPorFaixaMoradores = st.number_input("Domicilios por Faixa de Moradores: ", min_value=1)
    trabalhadores = st.number_input("Trabalhadores: ", min_value=1)
    tipoLoja = st.radio(
        "Tipo de Loja:",
        ["Mega", "Rua"],
        index=None,
    )

mapa_tipo_loja = {
"Mega": 0,
"Rua": 1
}

if st.button("Estimar"):
    entrada = pd.DataFrame({
        "Anos de Atividade": [anos],
        "Potencial de consumo médio por domicílio": [potencialMedioDomicilio],
        "Potencial de Consumo Total": [potencialConsumoTotal],
        "PEA Dia": [peaDia],
        "Trabalhadores": [trabalhadores],
        "População Mulheres": [mulheres],
        "Renda média domiciliar": [rendaMediaDomiciliar],
        "Densidade demográfica": [densidadeDemografica],
        "População": [populacao],
        "População Homens": [homens],
        "    Superior completo": [superiorCompleto],
        "Domicílios por faixa de moradores": [domiciliosPorFaixaMoradores],
        "Fluxo de Passantes": [fluxoPassantes],
        "tipo_loja_encoder": [mapa_tipo_loja.get(tipoLoja, None)],
    })

    entrada_scaled = scaler_variaveis.transform(entrada)
    previsao = modelo.predict(entrada_scaled)
    previsao = scaler_target.inverse_transform(previsao)
    previsao_valor = float(np.ravel(previsao)[0])

    print(previsao)

    st.success(f"Estimado anual: **R$ {previsao_valor:,.2f}**")
    st.caption(f"Erro médio do Modelo: {mape:,.2f}%")
    st.caption("Baseado em modelo Multi Layer Perceptron - Rede Neural, treinado em 12/11/2025.")
