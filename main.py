import asyncio
import os

import openai

from dotenv import load_dotenv
import layout

# Classes
view = layout.view

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

models_availible = {
    "1": "gpt-3.5-turbo",
    "2": "gpt-4"
}

async def get_openai_response(prompt: str, model: str) -> tuple:
    """
    Chama a API do OpenAI (nova interface) em modo "assíncrono" usando asyncio.to_thread.
    Retorna a resposta e o modelo utilizado.
    """

    def sync_completion():

        return openai.ChatCompletion.create(

            model=model,

            messages=[{"role": "user", "content": prompt}],

            max_tokens=100

        )

    response = await asyncio.to_thread(sync_completion)

    content = response.choices[0].message["content"]

    return model, content


async def main():

    print("Bem-vindo ao sistema de configuração de modelos!")

    while True:

        config = [] # Ira conter a multiplicacao dos modelos que seram usados, assim, posteriormente sendo passado assincronamente.

        view.main_menu()

        while True:

            escolha = input("Escolha o modelo (1 ou 2): ")

            if escolha.lower() == "done":
                break

            if escolha not in models_availible:
                print("Opção inválida. Escolha 1 para gpt-3.5-turbo ou 2 para gpt-4.")
                continue

            try:

                count = int(input("Digite o número de instâncias: "))
                if count < 1:
                    print("O número de instâncias deve ser pelo menos 1.")
                    continue

                model = models_availible[escolha]

                config.extend([model] * count)

            except ValueError:

                print("Por favor, insira um número válido.")

        if not config:

            print("Nenhum modelo configurado. Saindo.")

            break

        print(f"\nConfiguração aplicada: {config}")

        while True:

            user_input = input("\nDigite sua frase (ou 'exit' para sair da configuração): ")

            if user_input.lower() == "exit":

                print("Saindo da configuração atual...")

                break

            tasks = [

                asyncio.create_task(get_openai_response(user_input, model))

                for model in config

            ]

            print("\nExecutando chamadas para os modelos...")

            for coro in asyncio.as_completed(tasks):

                try:

                    model, resposta = await coro

                    print(f"Resposta do modelo {model}: {resposta}")

                except Exception as e:

                    print("Erro:", e)

if __name__ == "__main__":

    asyncio.run(main()) 