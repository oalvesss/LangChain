from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import Field, BaseModel
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

class Destino(BaseModel):
    cidade:str = Field("A cidade recomndada para visitar")
    motivo:str = Field("motivo pelo qual é interessante visitar essa cidade")

class Restaurantes(BaseModel):
    cidade:str = Field("A cidade recomndada para visitar")
    restaurantes:str = Field("Restaurantes recomendados na cidade")

parseador_distino = JsonOutputParser(pydantic_object= Destino)
parseador_restaurante = JsonOutputParser(pydantic_object= Restaurantes)

prompt_cidade = PromptTemplate(
    template="""
    Sugira uma cidade dado o meu interesse por {interesse}.
    {formato_de_saida}
    """,
    input_variables=["interesse"],
    partial_variables={"formato_de_saida" : parseador_distino.get_format_instructions()}
)

prompt_restaurantes = PromptTemplate(
    template="""
    Sugira restaurantes populares entre locais em {cidade}
    {formato_de_saida}
    """,
    partial_variables={"formato_de_saida" : parseador_restaurante.get_format_instructions()}
)

prompt_cultura = PromptTemplate(
    template="sugira atividades e locais culturais em {cidade}"
)



modelo = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0,
    api_key=api_key
)

cadeia_1 = prompt_cidade | modelo | parseador_distino
cadeia_2 = prompt_restaurantes | modelo | parseador_restaurante
cadeia_3 = prompt_cultura | modelo | StrOutputParser()

cadeia = (cadeia_1 | cadeia_2 | cadeia_3)

resposta = cadeia.invoke(
    {
        "interesse" : "praias"
    }
)

print(resposta)