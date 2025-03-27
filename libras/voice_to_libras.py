import speech_recognition as sr
import cv2
import imageio  # Para carregar GIFs
import time

# Dicionário que mapeia palavras para GIFs de sinais de Libras
sign_dict = {
    "olá": "signs/ola.gif",
    "obrigado": "signs/obrigado.gif",
    "tchau": "signs/tchau.gif",
    "sim": "signs/sim.gif",
    "não": "signs/nao.gif"
}

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Fale algo...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio, language="pt-BR")
            print(f"Você disse: {text}")
            return text.lower()
        except sr.UnknownValueError:
            print("Não entendi o que você disse.")
        except sr.RequestError:
            print("Erro na conexão com o Google Speech-to-Text.")
    return None

def show_gif(word):
    if word in sign_dict:
        gif_path = sign_dict[word]
        gif = imageio.mimread(gif_path)  # Carregar frames do GIF
        frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in gif]

        for _ in range(3):  # Repetir o GIF algumas vezes
            for frame in frames:
                cv2.imshow("Sinal de Libras", frame)
                if cv2.waitKey(100) & 0xFF == ord('q'):  # Pressione 'q' para sair
                    cv2.destroyAllWindows()
                    return

        time.sleep(1)  # Pequena pausa antes de fechar a janela
        cv2.destroyAllWindows()
    else:
        print(f"Nenhum sinal disponível para '{word}'")

if __name__ == "__main__":
    while True:
        spoken_text = recognize_speech()
        if spoken_text:
            show_gif(spoken_text)
