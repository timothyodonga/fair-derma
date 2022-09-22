import torchvision


def get_transform(mode):

    transforms_train = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(size=(224, 224)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomRotation(degrees=(90, 90)),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]
    )

    transforms_test = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(size=(224, 224)),
            # torchvision.transforms.RandomRotation(degrees = (90,90)),
            # torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]
    )

    transforms_explain = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(size=(224, 224)),
            torchvision.transforms.ToTensor(),
        ]
    )

    if mode == "train":
        return transforms_train
    elif mode == "test":
        return transforms_test
    elif mode == "explain":
        return transforms_explain
