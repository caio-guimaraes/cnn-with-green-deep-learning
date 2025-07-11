import os
import shutil
import xml.etree.ElementTree as ET

# Caminhos
xml_path = 'dataset2/xmls/Camera1/Set05/vehicles.xml'
base_dir = 'dataset2/Camera1/Set05/classes_carros'  # Substitua pelo caminho correto
dest_dir = 'data_test'

# Parse do XML
tree = ET.parse(xml_path)
root = tree.getroot()

# Placas já processadas
placas_processadas = set()

# Itera sobre veículos
for vehicle in root.findall('.//vehicle'):
    placa = vehicle.get('placa')
    color = vehicle.get('color')

    if not placa or not color:
        continue

    placa = placa.lower()
    color = color.lower()

    if placa in placas_processadas:
        continue

    placas_processadas.add(placa)

    placa_dir = os.path.join(base_dir, placa)
    color_dir = os.path.join(dest_dir, color)

    if not os.path.isdir(placa_dir):
        print(f'[Ignorado] Pasta não encontrada para: {placa}')
        continue

    # Cria a pasta da cor se não existir
    os.makedirs(color_dir, exist_ok=True)

    # Lista de arquivos da pasta, pegando apenas arquivos .jpg
    imagens = [f for f in os.listdir(placa_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not imagens:
        print(f'[Ignorado] Nenhuma imagem na pasta: {placa}')
        continue

    # Ordena e seleciona a primeira
    imagens.sort()
    imagem_origem = os.path.join(placa_dir, imagens[0])
    imagem_destino = os.path.join(color_dir, f'{placa}.jpg')  # renomeia com a placa

    shutil.copy2(imagem_origem, imagem_destino)
    print(f'Copiado: {placa} → {color}/{placa}.jpg')
