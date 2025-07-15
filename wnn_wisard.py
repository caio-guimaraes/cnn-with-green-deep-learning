import random
from sklearn.metrics import accuracy_score

def zero_pad(input_bits, address_size):
    padded = list(input_bits)
    while len(padded) % address_size != 0:
        padded.append(0)
    return padded

class Discriminator:
    def __init__(self, num_rams, address_size, input_size):
        self.rams = []
        for _ in range(num_rams):
            indices = random.sample(range(input_size), address_size)
            self.rams.append({'ram': {}, 'indices': indices})

    def train(self, input_bits):
        for ram_data in self.rams:
            bits = [input_bits[i] for i in ram_data['indices']]
            address = ''.join(str(b) for b in bits)
            ram = ram_data['ram']
            ram[address] = ram.get(address, 0) + 1

    def predict(self, input_bits):
        score = 0
        for ram_data in self.rams:
            bits = [input_bits[i] for i in ram_data['indices']]
            address = ''.join(str(b) for b in bits)
            score += ram_data['ram'].get(address, 0)  # Soma contagem (pode ser 0)
        return score

class WiSARD:
    def __init__(self, input_size, address_size):
        self.address_size = address_size
        self.input_size = input_size
        self.discriminators = {}

    def train(self, input_bits, label):
        padded_bits = zero_pad(input_bits, self.address_size)
        num_rams = len(padded_bits) // self.address_size
        if label not in self.discriminators:
            self.discriminators[label] = Discriminator(
                num_rams=num_rams,
                address_size=self.address_size,
                input_size=len(padded_bits)
            )
        self.discriminators[label].train(padded_bits)

    def predict(self, input_bits):
        padded_bits = zero_pad(input_bits, self.address_size)
        scores = {
            label: disc.predict(padded_bits)
            for label, disc in self.discriminators.items()
        }
        best_label = max(scores.items(), key=lambda x: x[1])[0]
        return best_label

    def fit(self, X_train, y_train, epochs=1):
        """Treina o modelo WiSARD com múltiplas épocas."""
        for epoch in range(epochs):
            for input_bits, label in zip(X_train, y_train):
                self.train(input_bits, label)

    def predict_batch(self, X_test):
        """Recebe uma lista de entradas binárias e retorna uma lista de predições."""
        return [self.predict(x) for x in X_test]

    def predict_bleaching(self, input_bits):
        padded_bits = zero_pad(input_bits, self.address_size)
        scores = {
            label: disc.predict(padded_bits)
            for label, disc in self.discriminators.items()
        }

        max_score = max(scores.values())
        bleach = max_score

        while bleach >= 0:
            candidates = [label for label, score in scores.items() if score >= bleach]
            if len(candidates) == 1:
                return candidates[0]
            bleach -= 1

        # Empate completo: retornar aleatoriamente
        return random.choice(list(scores.keys()))

    def predict_batch_bleaching(self, X_test):
        return [self.predict_bleaching(x) for x in X_test]
