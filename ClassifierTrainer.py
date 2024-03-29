from NN_structure import model_teamClassifer
from imagesDataset import EquiposDataset
from torch.utils.data.sampler import SubsetRandomSampler
import sys
import json
import torch
import numpy as np


def main():
    in_images = sys.argv[1]
    in_labels = sys.argv[2]
    out_classifier = sys.argv[3]
    epochs = int(sys.argv[4])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    json_labels = json.load(open(in_labels, 'r'))
    trainset = EquiposDataset(
        directory=in_images, labels=json_labels)

    train_size = len(trainset)
    indexes = list(range(train_size))

    batch_size = 12
    minimum_validation_loss = np.inf

    train_index = indexes[int(np.floor(train_size*0.1)):]
    test_index = indexes[:int(np.floor(train_size*0.1))]
    red = model_teamClassifer()
    if device == 'cuda':
        red = red.to(device='cuda')
        print("USANDO CUDA")
    else:
        red = red.to('cpu')

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=red.parameters(), lr=0.09)
    train_sampler = SubsetRandomSampler(train_index)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=8, drop_last=True, num_workers=0, sampler=train_sampler)
    validation_sampler = SubsetRandomSampler(test_index)
    validation_loader = torch.utils.data.DataLoader(
        trainset, batch_size=8, shuffle=False, drop_last=True, num_workers=0)
    for epoch in range(1, epochs+1):
        suma_train_loss = 0
        suma_valid_loss = 0

        # training de la red
        red.train()
        elemento = 1
        for (data, target) in train_loader:
            print(target[0])
            optimizer.zero_grad()
            output = red(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            suma_train_loss += loss.item()*data.size(0)
            print(f"Aprendiendo de elemento {elemento}")
            data = None
            target = None
            output = None

        # validacion de la red
        print("="*100)
        print(f"Entrenamiento epoca {epoch} finalizado, comenzando validacion")
        red.eval()
        for (data, target) in validation_loader:
            output = red(data)
            loss = criterion(output, target)
            suma_valid_loss += loss.item()*data.size(0)
            data = None
            target = None
            output = None

        valid_loss = suma_valid_loss/(len(validation_loader)*batch_size)
        train_loss = suma_train_loss/(len(train_loader)*batch_size)

        print(
            f'Epoch {epoch}\t Training Loss: {train_loss}\t Validation Loss:{valid_loss}')

        # Guardando el modelo cada vez que la perdida de validación decrementa.
        if valid_loss <= minimum_validation_loss:
            fails = 0
            print(
                f'Perdida de validacion bajo de {round(minimum_validation_loss, 6)} a {round(valid_loss, 6)}')
            torch.save(red.state_dict(), out_classifier)
            minimum_validation_loss = valid_loss
            print('Guardando modelo')
        else:
            # si las fallas llega a 10, se cierra el programa y se guarda el modelo
            fails += 1
            if fails >= 50:
                print('Perdida no ha bajado. Manteniendo ultimo modelo guardado')
                torch.save(red.state_dict(), out_classifier)
                minimum_validation_loss = valid_loss
                exit(0)

    print("Return")


if __name__ == "__main__":
    main()
