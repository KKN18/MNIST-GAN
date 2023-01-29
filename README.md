# MNIST-GAN
AI model generating digit image based on MNIST dataset

## Environment 
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-kwYmLlPhvN4JkkzKKhpx7pz5iqyqiz1#scrollTo=Ka8rQBYrZyXG)

## Dataset
### MNIST dataset
<img width="500" src="https://user-images.githubusercontent.com/63842546/214065889-45168edd-e111-458f-9ecf-a99abfaae632.png"/>

## Model
### Generator

```python
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(z_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, data_length)
```

###

```python
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(data_length, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)
```

## Result
### Epoch 50
<img width="300" src="https://user-images.githubusercontent.com/63842546/215310037-62cccbf9-5450-46c0-a7d5-dd7629bfe243.png"/>

