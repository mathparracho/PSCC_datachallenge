import gzip
import nibabel as nib
import os

def descomprimir_nii_gz(caminho_arquivo_gz, caminho_saida):
    # Abrir o arquivo .nii.gz e descomprimir
    for filename in os.listdir(caminho_arquivo_gz):
        print(filename[:-3])
        with gzip.open(caminho_arquivo_gz + "/" + filename, 'rb') as f_gz:
            conteudo_gz = f_gz.read()

        # Salvar o conteúdo descomprimido em um arquivo temporário
        caminho_temporario = caminho_saida + filename[:-3]
        with open(caminho_temporario, 'wb') as f_temp:
            f_temp.write(conteudo_gz)

    # Ler o arquivo NIfTI descomprimido usando nibabe

    # Remover o arquivo temporário
    #os.remove(caminho_temporario)

    return img

# Exemplo de uso:
caminho_arquivo_gz = '/tsi/data_education/data_challenge/test/volume/'
caminho_saida = '/home/ids/ext-1437/project/data/test/volume/'

imagem_nii = descomprimir_nii_gz(caminho_arquivo_gz, caminho_saida)
