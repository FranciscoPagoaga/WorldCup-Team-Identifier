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
        directory=in_images, labels=json_labels, size={733, 565})

    train_size = len(trainset)
    indexes = list(range(train_size))

    batch_size = 8

    train_index = indexes[int(np.floor(train_size*0.1)):]
    test_index = indexes[:int(np.floor(train_size*0.1))]
    red = model_teamClassifer(ratio_height=733, ratio_width=565, out=32)

    if device == 'cuda':
        red = red.to(device='cuda')
        print("USANDO CUDA")
    else:
        red = red.to('cpu')

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=red.parameters(), lr=0.09)
    train_sampler = SubsetRandomSampler(train_index)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, sampler=train_sampler, num_workers=0)
    validation_sampler = SubsetRandomSampler(test_index)
    validation_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, sampler=validation_sampler, num_workers=0)
    for epoch in range(1, epochs+1):
        suma_train_loss = 0
        suma_valid_loss = 0

        # training de la red
        red.train()
        for batch_index, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            if device == 'cuda':
                data = data.cuda()
                target = target.cuda()
                output = red(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                suma_train_loss += loss.item()*data.size(0)

        # validacion de la red
        red.eval()
        for batch_index, (data, target) in enumerate(validation_loader):
            if device == 'cuda':
                data = data.cuda()
                target = target.cuda()

            output = red(data)
            loss = criterion(output, target)
            suma_valid_loss += loss.item()*data.size(0)

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
